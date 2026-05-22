"""
PIMBCN 训练脚本（2026-05-21 优化版）

基于 V514 训练脚本，优化点：
- 数据加载: num_workers=4 + pin_memory=True
- 混合精度: torch.cuda.amp.GradScaler + autocast（预期 1.5-2x 加速，显存减半）
- 模型编译: torch.compile()（预期 1.3-1.5x 加速）
- 断点续训: --resume 参数支持中断后继续训练
- 使用优化版数据类 PIMBCN_data_0521 和网络 PIMBCN_net0521
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from tqdm import tqdm

from PIMBCN_net0521 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0521 import PIMBCNDataset


def train_model(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # ================= 训练超参数 =================
    batch_size = 16
    epochs = 10000

    head_learning_rate = 5e-4
    shared_learning_rate = 5e-4
    freq_bins = 2501

    # ================= 断点续训 vs 从头训练 =================
    start_epoch = 0
    best_val_loss = float('inf')
    scaler_state_dict = None

    if resume_path is not None and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # 解析路径，提取 run_name 和 save_dir
        save_dir = os.path.dirname(resume_path)
        run_dir = os.path.dirname(save_dir)
        run_name = os.path.basename(run_dir)

        writer = SummaryWriter(f'runs/{run_name}')
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', -1) + 1
        scaler_state_dict = checkpoint.get('scaler_state_dict', None)
        print(f"可续训至第 {checkpoint.get('epoch', -1) + 1} 轮，"
              f"最佳验证损失: {best_val_loss:.4f}，"
              f"已训练 {start_epoch} 轮")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f'pi_mbcn_hvac_MTL_{timestamp}_epochs{epochs}_bs{batch_size}_lr{head_learning_rate}_opt0521'
        save_dir = f'runs/{run_name}/models'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{run_name}')
        print(f"从头训练，时间戳: {timestamp}")

    # ================= 数据加载（优化版） =================
    data_directory = "F:\\lyh\\paddlespeech\\csvdata333"

    train_dataset = PIMBCNDataset(
        directory_path=data_directory,
        input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2,
        val_split=0.2, is_validation=False
    )

    val_dataset = train_dataset.get_validation_dataset()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ================= 模型构建 =================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    loss_wrapper = PhysicsLossWrapper().to(device)

    # ================= 优化器 =================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.heads.parameters(), 'lr': head_learning_rate},
        {'params': loss_wrapper.parameters(), 'lr': head_learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7
    )

    # ================= AMP 混合精度 =================
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    amp_enabled = scaler is not None
    if amp_enabled:
        print("启用 AMP 混合精度训练")

    # ================= 加载检查点状态 =================
    compile_enabled = hasattr(torch, 'compile')
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        loss_wrapper.load_state_dict(checkpoint['loss_wrapper_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print("已加载模型、损失模块、优化器、调度器、AMP 状态")

    # ================= torch.compile =================
    # 2026-05-21: compile 暂禁用，PIMBCN_net0521 的 TransposeConvBlock 残差连接
    # 在 compile trace 阶段存在维度广播不匹配 (16 vs 32)，等待模型修复后重新启用
    compile_enabled = False
    if compile_enabled:
        print("启用 torch.compile() 加速...")
        model = torch.compile(model)

    # 计算图保存
    sample_input = torch.randn(1, 3).to(device)
    sample_mode = torch.tensor([0], dtype=torch.long).to(device)
    sample_type = torch.tensor([0], dtype=torch.long).to(device)
    try:
        writer.add_graph(model, (sample_input, sample_mode, sample_type))
    except Exception as e:
        print(f"跳过计算图保存: {e}")

    # ================= 训练循环 =================
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_wrapper.train()
        total_train_loss = 0.0
        total_mse_spec_loss = 0.0
        total_cosine_spec_loss = 0.0
        total_mse_oaspl_loss = 0.0

        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for batch_idx, (inputs, types, modes, _, _, target_spectrum) in enumerate(train_bar):
            inputs = inputs.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            target_spectrum = target_spectrum.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- AMP 前向 ---
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    pred_spectrum = model(inputs, modes, types)
                    loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl = loss_wrapper(
                        pred_spectrum, target_spectrum)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_spectrum = model(inputs, modes, types)
                loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl = loss_wrapper(
                    pred_spectrum, target_spectrum)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item()
            total_mse_spec_loss += loss_mse_spec.item()
            total_cosine_spec_loss += loss_cosine_spec.item()
            total_mse_oaspl_loss += loss_mse_oaspl.item()

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse_spec': f'{loss_mse_spec.item():.4f}',
                'cosine': f'{loss_cosine_spec.item():.4f}',
            })

        scheduler.step()

        # ================= 验证阶段 =================
        model.eval()
        loss_wrapper.eval()
        val_mse_loss = val_spectrum_loss = 0.0

        val_bar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] Validation', leave=False)
        with torch.no_grad():
            for inputs, types, modes, _, _, target_spectrum in val_bar:
                inputs = inputs.to(device, non_blocking=True)
                types = types.to(device, non_blocking=True)
                modes = modes.to(device, non_blocking=True)
                target_spectrum = target_spectrum.to(device, non_blocking=True)

                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        pred_spectrum = model(inputs, modes, types)
                else:
                    pred_spectrum = model(inputs, modes, types)

                spectrum_loss = F.mse_loss(pred_spectrum, target_spectrum).item()
                val_spectrum_loss += spectrum_loss
                val_mse_loss += spectrum_loss

        # ================= 日志与 Tensorboard =================
        num_batches = len(train_loader)
        avg_train = total_train_loss / num_batches
        avg_mse_spec = total_mse_spec_loss / num_batches
        avg_cosine_spec = total_cosine_spec_loss / num_batches
        avg_mse_oaspl = total_mse_oaspl_loss / num_batches
        avg_val = val_mse_loss / len(val_loader)

        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec_loss', avg_mse_spec, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec_loss', avg_cosine_spec, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl', avg_mse_oaspl, epoch + 1)
        writer.add_scalar('Loss/val_spectrum', avg_val, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)

        current_lr_shared = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[2]['lr']
        writer.add_scalar('Learning_Rate/shared_trunk', current_lr_shared, epoch + 1)
        writer.add_scalar('Learning_Rate/task_heads', current_lr_head, epoch + 1)

        with torch.no_grad():
            weights = torch.exp(-loss_wrapper.log_vars)
            writer.add_scalar('Dynamic_Weights/MSE_Weight', weights[0].item(), epoch + 1)
            writer.add_scalar('Dynamic_Weights/Cosine_Weight', weights[1].item(), epoch + 1)
            writer.add_scalar('Dynamic_Weights/OASPL_Weight', weights[2].item(), epoch + 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.4f} | "
                  f"Val MSE: {avg_val:.4f} | LR: {current_lr_head:.2e}")

        # ================= 保存检查点（含完整训练状态） =================
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_dict = {
                'model_state_dict': model.state_dict(),
                'loss_wrapper_state_dict': loss_wrapper.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))

        # 每 500 轮额外保存一个带轮数标记的检查点，防止 best_model 很久不更新
        if (epoch + 1) % 500 == 0:
            save_dict = {
                'model_state_dict': model.state_dict(),
                'loss_wrapper_state_dict': loss_wrapper.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))

    print("模型训练完成！")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PI-MBCN 训练脚本 (优化版)')
    parser.add_argument('--resume', type=str, default=None,
                        help='继续训练的检查点路径，如: runs/xxx/models/best_model.pth')
    args = parser.parse_args()

    train_model(resume_path=args.resume)
