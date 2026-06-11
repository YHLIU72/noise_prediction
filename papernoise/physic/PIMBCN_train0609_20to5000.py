"""
PIMBCN 训练脚本（2026-06-09 频率范围适配版：20~5000Hz）

修复与优化更新：
1. 解决 Windows Triton 编译崩溃：安全回退至原生 Eager 高速执行模式。
2. 修复 PyTorch AMP 警告：使用最新的 torch.amp 接口替换弃用的 torch.cuda.amp。
3. 极速吞吐：保留全量显存缓存 (In-Memory VRAM Caching) 与纯异步 GPU 累加。
4. 小样本特化超参数：Batch Size (8), Weight Decay (5e-3), 物理损失非对称初始化。

变更说明 (2026-06-09):
- 频率范围由 0~10000Hz 改为 20~5000Hz，freq_bins 由 2501 改为 1246。
- 对应导入 0609 版模型与数据集。
"""
import os

# 解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from tqdm import tqdm

# 导入 20~5000Hz 适配版模型和数据集
from PIMBCN_net0609_20to5000 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0609_20to5000 import PIMBCNDataset


def train_model(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    torch.backends.cudnn.benchmark = True

    # ================= 核心：小样本特化超参数 (重平衡版) =================
    batch_size = 8
    epochs = 50000

    head_learning_rate = 3e-4       # 降低 lr, 更稳定的小样本收敛
    shared_learning_rate = 3e-4
    freq_bins = 1246  # 20~5000Hz, 间隔4Hz, 共1246点

    # 适中的权重衰减, 避免过度压缩参数
    weight_decay_val = 1e-3

    # ================= 断点续训逻辑 =================
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
        run_name = f'pi_mbcn_20to5000_{timestamp}_epochs{epochs}_bs{batch_size}_wd{weight_decay_val}'
        save_dir = f'runs/{run_name}/models'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{run_name}')
        print(f"从头训练 (20~5000Hz)，时间戳: {timestamp}")

    # ================= 数据集全量预载入 =================
    data_directory = "F:\\lyh\\paddlespeech\\csvdata333"

    # 使用增强版数据集，启用数据增强
    train_dataset = PIMBCNDataset(
        directory_path=data_directory, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False,
        augment=True  # 启用数据增强
    )
    val_dataset = train_dataset.get_validation_dataset()  # 验证集自动禁用增强

    # 注意：由于使用数据增强，不能预载入显存（每次需要重新生成增强数据）
    # 改用 DataLoader 动态加载
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    print(f"训练样本: {num_train_samples}, 验证样本: {num_val_samples}")

    # ================= 模型构建与物理权重干预 =================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    loss_wrapper = PhysicsLossWrapper().to(device)

    # 重平衡损失权重: Sobolev(4.0) > MSE(5.0) > Cosine(1.5) = LogMSE(1.5) > OASPL(1.0)
    print(f"损失权重: MSE={loss_wrapper.weight_mse}, Cosine={loss_wrapper.weight_cosine}, "
          f"OASPL={loss_wrapper.weight_oaspl}, Gradient={loss_wrapper.weight_grad}, "
          f"LogMSE={loss_wrapper.weight_logmse}")

    # ================= 优化器与调度器 =================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.heads.parameters(), 'lr': head_learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay_val)
    
    # 延长重启周期, 给模型充分收敛时间 (T_0=200)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-7
    )

    # ================= 混合精度 (已修复 PyTorch 2.x 警告) =================
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    amp_enabled = scaler is not None

    if resume_path is not None and os.path.exists(resume_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)

    # ================= 编译模式 (已针对 Windows 禁用) =================
    compile_enabled = False  # 强制禁用，规避 Windows Triton 寻址崩溃
    if compile_enabled and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("当前运行于 Windows 原生环境，已关闭动态图编译，采用全量显存 Eager 高速计算。")

    # ================= 预分配 GPU 损失累加器 =================
    # 训练损失累加器
    epoch_train_loss = torch.zeros(1, device=device)
    epoch_train_mse = torch.zeros(1, device=device)
    epoch_train_cosine = torch.zeros(1, device=device)
    epoch_train_oaspl = torch.zeros(1, device=device)
    epoch_train_grad = torch.zeros(1, device=device)
    epoch_train_logmse = torch.zeros(1, device=device)
    train_batch_count = 0
    # 验证损失累加器（独立）
    epoch_val_loss = torch.zeros(1, device=device)
    epoch_val_mse = torch.zeros(1, device=device)
    epoch_val_cosine = torch.zeros(1, device=device)
    epoch_val_oaspl = torch.zeros(1, device=device)
    epoch_val_grad = torch.zeros(1, device=device)
    epoch_val_logmse = torch.zeros(1, device=device)
    val_batch_count = 0

    # ================= 训练循环（支持数据增强） =================
    epoch_bar = tqdm(range(start_epoch, epochs), desc='Training Epochs')
    for epoch in epoch_bar:
        model.train()
        loss_wrapper.train()

        epoch_train_loss.zero_()
        epoch_train_mse.zero_()
        epoch_train_cosine.zero_()
        epoch_train_oaspl.zero_()
        epoch_train_grad.zero_()
        epoch_train_logmse.zero_()
        train_batch_count = 0

        # 使用 DataLoader 动态加载（支持数据增强）
        for inputs, types, modes, _, _, target_spectrum in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            target_spectrum = target_spectrum.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                # 使用最新的 torch.amp.autocast 接口
                with torch.amp.autocast('cuda'):
                    pred_spectrum = model(inputs, modes, types)
                    loss, l_mse, l_cos, l_oaspl, l_grad, l_logmse = loss_wrapper(pred_spectrum, target_spectrum)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_spectrum = model(inputs, modes, types)
                loss, l_mse, l_cos, l_oaspl, l_grad, l_logmse = loss_wrapper(pred_spectrum, target_spectrum)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # GPU 内纯异步累加
            epoch_train_loss += loss.detach()
            epoch_train_mse += l_mse.detach()
            epoch_train_cosine += l_cos.detach()
            epoch_train_oaspl += l_oaspl.detach()
            epoch_train_grad += l_grad.detach()
            epoch_train_logmse += l_logmse.detach()
            train_batch_count += 1

        scheduler.step()

        # ================= 验证阶段 =================
        model.eval()
        loss_wrapper.eval()
        epoch_val_loss.zero_()
        epoch_val_mse.zero_()
        epoch_val_cosine.zero_()
        epoch_val_oaspl.zero_()
        epoch_val_grad.zero_()
        epoch_val_logmse.zero_()
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
                        val_loss, val_mse, val_cos, val_oaspl, val_grad, val_logmse = loss_wrapper(pred_spectrum, target_spectrum)
                else:
                    pred_spectrum = model(inputs, modes, types)
                    val_loss, val_mse, val_cos, val_oaspl, val_grad, val_logmse = loss_wrapper(pred_spectrum, target_spectrum)

                # 验证损失独立累加
                epoch_val_loss += val_loss.detach()
                epoch_val_mse += val_mse.detach()
                epoch_val_cosine += val_cos.detach()
                epoch_val_oaspl += val_oaspl.detach()
                epoch_val_grad += val_grad.detach()
                epoch_val_logmse += val_logmse.detach()
                val_batch_count += 1

        # ================= 日志记录 =================
        avg_train = (epoch_train_loss / train_batch_count).item()
        avg_train_mse = (epoch_train_mse / train_batch_count).item()
        avg_train_cosine = (epoch_train_cosine / train_batch_count).item()
        avg_train_oaspl = (epoch_train_oaspl / train_batch_count).item()
        avg_train_grad = (epoch_train_grad / train_batch_count).item()
        avg_train_logmse = (epoch_train_logmse / train_batch_count).item()
        avg_val = (epoch_val_loss / val_batch_count).item()
        avg_val_mse = (epoch_val_mse / val_batch_count).item()
        avg_val_cosine = (epoch_val_cosine / val_batch_count).item()
        avg_val_oaspl = (epoch_val_oaspl / val_batch_count).item()
        avg_val_grad = (epoch_val_grad / val_batch_count).item()
        avg_val_logmse = (epoch_val_logmse / val_batch_count).item()

        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec', avg_train_mse, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec', avg_train_cosine, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl', avg_train_oaspl, epoch + 1)
        writer.add_scalar('Loss/train_grad', avg_train_grad, epoch + 1)
        writer.add_scalar('Loss/train_logmse', avg_train_logmse, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        writer.add_scalar('Loss/val_mse_spec', avg_val_mse, epoch + 1)
        writer.add_scalar('Loss/val_cosine_spec', avg_val_cosine, epoch + 1)
        writer.add_scalar('Loss/val_mse_oaspl', avg_val_oaspl, epoch + 1)
        writer.add_scalar('Loss/val_grad', avg_val_grad, epoch + 1)
        writer.add_scalar('Loss/val_logmse', avg_val_logmse, epoch + 1)

        # 轻量化刷新进度条
        if (epoch + 1) % 10 == 0:
            epoch_bar.set_postfix({
                'Train': f'{avg_train:.4f}',
                'Spec': f'{avg_train_mse:.4f}',
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
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PI-MBCN 20~5000Hz 训练脚本')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_model(resume_path=args.resume)
