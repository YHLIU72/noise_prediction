"""
PIMBCN 训练脚本（2026-06-10 损失函数 V2 重构版）

损失函数变更:
- 移除 Cosine (全正 dB 值下退化为常数)、LogMSE (双重对数无意义)、OASPL (与MSE冗余)
- 新增线性域峰值 MSE (dB→声压比, 自动侧重高峰值区域)
- 保留 dB-MSE (核心) + Sobolev 梯度 (平滑约束)
- 损失权重: MSE(5.0) + LinearPeak(2.0) + Sobolev(3.0)

超参数保持不变: lr=3e-4, T_0=200, wd=1e-3, batch=8, freq_bins=1246
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from tqdm import tqdm

from PIMBCN_net0610_lossV2 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0610_lossV2 import PIMBCNDataset


def train_model(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    torch.backends.cudnn.benchmark = True

    # ================= 超参数 =================
    batch_size = 8
    epochs = 50000

    head_learning_rate = 3e-4
    shared_learning_rate = 3e-4
    freq_bins = 1246
    weight_decay_val = 1e-3

    # ================= 断点续训 =================
    start_epoch = 0
    best_val_loss = float('inf')
    scaler_state_dict = None

    if resume_path is not None and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        save_dir = os.path.dirname(resume_path)
        run_name = os.path.basename(os.path.dirname(save_dir))
        writer = SummaryWriter(f'runs/{run_name}')
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', -1) + 1
        scaler_state_dict = checkpoint.get('scaler_state_dict', None)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f'pi_mbcn_lossV2_{timestamp}_epochs{epochs}_bs{batch_size}_wd{weight_decay_val}'
        save_dir = f'runs/{run_name}/models'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{run_name}')
        print(f"从头训练 (损失V2)，时间戳: {timestamp}")

    # ================= 数据集 =================
    data_directory = "F:\\lyh\\paddlespeech\\csvdata333"

    train_dataset = PIMBCNDataset(
        directory_path=data_directory, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False,
        augment=True
    )
    val_dataset = train_dataset.get_validation_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    print(f"训练样本: {num_train_samples}, 验证样本: {num_val_samples}")

    # ================= 模型与损失 =================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    loss_wrapper = PhysicsLossWrapper().to(device)

    print(f"损失权重 V2: MSE={loss_wrapper.weight_mse}, "
          f"LinearPeak={loss_wrapper.weight_linear}, "
          f"Sobolev={loss_wrapper.weight_grad}")

    # ================= 优化器与调度器 =================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.heads.parameters(), 'lr': head_learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay_val)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-7
    )

    # ================= 混合精度 =================
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    amp_enabled = scaler is not None

    if resume_path is not None and os.path.exists(resume_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)

    compile_enabled = False
    if compile_enabled and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("当前运行于 Windows 原生环境，已关闭动态图编译，采用全量显存 Eager 高速计算。")

    # ================= 预分配损失累加器 (精简为3项) =================
    epoch_train_loss = torch.zeros(1, device=device)
    epoch_train_mse = torch.zeros(1, device=device)
    epoch_train_grad = torch.zeros(1, device=device)
    epoch_train_linear = torch.zeros(1, device=device)
    train_batch_count = 0

    epoch_val_loss = torch.zeros(1, device=device)
    epoch_val_mse = torch.zeros(1, device=device)
    epoch_val_grad = torch.zeros(1, device=device)
    epoch_val_linear = torch.zeros(1, device=device)
    val_batch_count = 0

    # ================= 训练循环 =================
    epoch_bar = tqdm(range(start_epoch, epochs), desc='Training Epochs')
    for epoch in epoch_bar:
        model.train()
        loss_wrapper.train()

        epoch_train_loss.zero_()
        epoch_train_mse.zero_()
        epoch_train_grad.zero_()
        epoch_train_linear.zero_()
        train_batch_count = 0

        for inputs, types, modes, _, _, target_spectrum in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            target_spectrum = target_spectrum.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.amp.autocast('cuda'):
                    pred_spectrum = model(inputs, modes, types)
                    loss, l_mse, l_grad, l_linear = loss_wrapper(pred_spectrum, target_spectrum)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_spectrum = model(inputs, modes, types)
                loss, l_mse, l_grad, l_linear = loss_wrapper(pred_spectrum, target_spectrum)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_loss += loss.detach()
            epoch_train_mse += l_mse.detach()
            epoch_train_grad += l_grad.detach()
            epoch_train_linear += l_linear.detach()
            train_batch_count += 1

        scheduler.step()

        # ================= 验证 =================
        model.eval()
        loss_wrapper.eval()
        epoch_val_loss.zero_()
        epoch_val_mse.zero_()
        epoch_val_grad.zero_()
        epoch_val_linear.zero_()
        val_batch_count = 0

        with torch.no_grad():
            for inputs, types, modes, _, _, target_spectrum in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                types = types.to(device, non_blocking=True)
                modes = modes.to(device, non_blocking=True)
                target_spectrum = target_spectrum.to(device, non_blocking=True)

                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        pred_spectrum = model(inputs, modes, types)
                        val_loss, val_mse, val_grad, val_linear = loss_wrapper(pred_spectrum, target_spectrum)
                else:
                    pred_spectrum = model(inputs, modes, types)
                    val_loss, val_mse, val_grad, val_linear = loss_wrapper(pred_spectrum, target_spectrum)

                epoch_val_loss += val_loss.detach()
                epoch_val_mse += val_mse.detach()
                epoch_val_grad += val_grad.detach()
                epoch_val_linear += val_linear.detach()
                val_batch_count += 1

        # ================= 日志 =================
        avg_train = (epoch_train_loss / train_batch_count).item()
        avg_train_mse = (epoch_train_mse / train_batch_count).item()
        avg_train_grad = (epoch_train_grad / train_batch_count).item()
        avg_train_linear = (epoch_train_linear / train_batch_count).item()
        avg_val = (epoch_val_loss / val_batch_count).item()
        avg_val_mse = (epoch_val_mse / val_batch_count).item()
        avg_val_grad = (epoch_val_grad / val_batch_count).item()
        avg_val_linear = (epoch_val_linear / val_batch_count).item()

        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_db', avg_train_mse, epoch + 1)
        writer.add_scalar('Loss/train_grad', avg_train_grad, epoch + 1)
        writer.add_scalar('Loss/train_linear_peak', avg_train_linear, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        writer.add_scalar('Loss/val_mse_db', avg_val_mse, epoch + 1)
        writer.add_scalar('Loss/val_grad', avg_val_grad, epoch + 1)
        writer.add_scalar('Loss/val_linear_peak', avg_val_linear, epoch + 1)

        if (epoch + 1) % 10 == 0:
            epoch_bar.set_postfix({
                'Train': f'{avg_train:.4f}',
                'dB-MSE': f'{avg_train_mse:.4f}',
                'Val': f'{avg_val:.4f}'
            })

        # ================= 模型保存 =================
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'input_mean': torch.from_numpy(train_dataset.input_mean),
                'input_std': torch.from_numpy(train_dataset.input_std),
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PI-MBCN 损失V2 训练脚本')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_model(resume_path=args.resume)
