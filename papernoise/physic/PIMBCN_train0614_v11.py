"""
PIMBCN 训练脚本（2026-06-14 V11: 方案1 — 频率加权损失）

=== 基于 V4 训练策略，新增频率加权损失 ===
[V11] 损失函数对低频段(20-100Hz)给予 1.6~2.5× 更高权重
  - 引导模型优先学习低频段频谱形状
  - 同时记录加权/未加权 MSE 对比

=== V4 训练策略（全部保留）===
[架构] 1共享Head, EMA+warmup+T_0=150+梯度累积
[增强] 弱增强 (噪声0.02/频谱0.8dB/频移[-3,4])

超参数: lr=3e-4, wd=1e-3, batch=8, accum=2(等效16), epochs=50000
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch, copy, time, argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from PIMBCN_net0614_v11 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0614_v11 import PIMBCNDataset


def train_model(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    torch.backends.cudnn.benchmark = True

    # ===================== 超参数 =====================
    batch_size = 8
    accum_steps = 2
    epochs = 50000
    head_learning_rate = 3e-4
    shared_learning_rate = 3e-4
    freq_bins = 1246          # 20~5000Hz, 1246点 (间隔4Hz)
    weight_decay_val = 1e-3

    # ===================== 断点续训 =====================
    start_epoch = 0; best_val_loss = float('inf'); scaler_state_dict = None
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
        run_name = f'pi_mbcn_v11_{timestamp}'
        save_dir = f'runs/{run_name}/models'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{run_name}')
        print(f"从头训练 (V11: 频率加权损失, 20~5000Hz), 时间戳: {timestamp}")

    # ===================== 数据集 =====================
    data_directory = "F:\\lyh\\paddlespeech\\csvdata333"
    train_dataset = PIMBCNDataset(
        directory_path=data_directory, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False, augment=True)
    val_dataset = train_dataset.get_validation_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=2, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=2, persistent_workers=True)
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    print(f"频率范围: 20~5000Hz, 频点数: {freq_bins}")
    print(f"梯度累积: {accum_steps}步, 等效batch={batch_size * accum_steps}")
    print(f"[V11] 频率加权损失已启用: 低频(20Hz)权重≈2.5×, 高频(5000Hz)权重≈0.5×")

    # ===================== 模型 + EMA =====================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    loss_wrapper = PhysicsLossWrapper(freq_bins=freq_bins).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters(): p.requires_grad_(False)
    ema_decay = 0.999
    print(f"损失权重: MSE(频率加权)={loss_wrapper.weight_mse}, "
          f"LinearPeak={loss_wrapper.weight_linear}, Sobolev={loss_wrapper.weight_grad}")

    # ===================== 优化器 + 调度器 =====================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.mode_embed.parameters(), 'lr': shared_learning_rate},
        {'params': model.type_embed.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_head.parameters(), 'lr': head_learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2, eta_min=1e-7)

    # ===================== LR warmup =====================
    warmup_epochs = 5; warmup_steps = warmup_epochs * len(train_loader)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]; global_step = 0

    # ===================== 混合精度 =====================
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    amp_enabled = scaler is not None
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and scaler_state_dict is not None:
            scaler.load_state_dict(scaler_state_dict)
    print("已关闭动态图编译。")

    # ===================== 损失累加器 =====================
    epoch_train_loss = torch.zeros(1, device=device)
    epoch_train_mse = torch.zeros(1, device=device)
    epoch_train_grad = torch.zeros(1, device=device)
    epoch_train_linear = torch.zeros(1, device=device)
    epoch_val_loss = torch.zeros(1, device=device)
    epoch_val_mse = torch.zeros(1, device=device)
    epoch_val_grad = torch.zeros(1, device=device)
    epoch_val_linear = torch.zeros(1, device=device)
    # [V11] 额外记录未加权的纯 MSE (用于对比加权效果)
    epoch_train_mse_raw = torch.zeros(1, device=device)
    epoch_val_mse_raw = torch.zeros(1, device=device)

    # ===================== 训练循环 =====================
    epoch_bar = tqdm(range(start_epoch, epochs), desc='Training Epochs')
    for epoch in epoch_bar:
        model.train(); loss_wrapper.train()
        epoch_train_loss.zero_(); epoch_train_mse.zero_()
        epoch_train_grad.zero_(); epoch_train_linear.zero_()
        epoch_train_mse_raw.zero_()
        train_batch_count = 0; optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, types, modes, _, _, target) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            modes = modes.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg, blr in zip(optimizer.param_groups, base_lrs):
                    pg['lr'] = blr * lr_scale

            if amp_enabled:
                with torch.amp.autocast('cuda'):
                    pred = model(inputs, modes, types)
                    loss, l_mse, l_grad, l_linear = loss_wrapper(pred, target)
                scaler.scale(loss / accum_steps).backward()
            else:
                pred = model(inputs, modes, types)
                loss, l_mse, l_grad, l_linear = loss_wrapper(pred, target)
                (loss / accum_steps).backward()

            # [V11] 额外计算纯 MSE (不加权, 用于对比)
            with torch.no_grad():
                raw_mse = F.mse_loss(pred, target)

            epoch_train_loss += loss.detach()
            epoch_train_mse += l_mse.detach()        # 频率加权 MSE
            epoch_train_grad += l_grad.detach()
            epoch_train_linear += l_linear.detach()
            epoch_train_mse_raw += raw_mse.detach()   # 纯 MSE
            train_batch_count += 1; global_step += 1

            if (batch_idx + 1) % accum_steps == 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        if global_step >= warmup_steps: scheduler.step()

        # ---- 验证 ----
        ema_model.eval(); loss_wrapper.eval()
        epoch_val_loss.zero_(); epoch_val_mse.zero_()
        epoch_val_grad.zero_(); epoch_val_linear.zero_()
        epoch_val_mse_raw.zero_()
        val_batch_count = 0
        with torch.no_grad():
            for inputs, types, modes, _, _, target in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                types = types.to(device, non_blocking=True)
                modes = modes.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        pred = ema_model(inputs, modes, types)
                        v_loss, v_mse, v_grad, v_linear = loss_wrapper(pred, target)
                else:
                    pred = ema_model(inputs, modes, types)
                    v_loss, v_mse, v_grad, v_linear = loss_wrapper(pred, target)
                raw_mse = F.mse_loss(pred, target)   # [V11] 纯 MSE
                epoch_val_loss += v_loss.detach()
                epoch_val_mse += v_mse.detach()
                epoch_val_grad += v_grad.detach()
                epoch_val_linear += v_linear.detach()
                epoch_val_mse_raw += raw_mse.detach()
                val_batch_count += 1

        # ---- 日志 ----
        avg_train = (epoch_train_loss / train_batch_count).item()
        avg_train_mse = (epoch_train_mse / train_batch_count).item()
        avg_train_mse_raw = (epoch_train_mse_raw / train_batch_count).item()
        avg_train_grad = (epoch_train_grad / train_batch_count).item()
        avg_train_linear = (epoch_train_linear / train_batch_count).item()
        avg_val = (epoch_val_loss / val_batch_count).item()
        avg_val_mse = (epoch_val_mse / val_batch_count).item()
        avg_val_mse_raw = (epoch_val_mse_raw / val_batch_count).item()
        avg_val_grad = (epoch_val_grad / val_batch_count).item()
        avg_val_linear = (epoch_val_linear / val_batch_count).item()

        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_weighted', avg_train_mse, epoch + 1)     # [V11] 加权
        writer.add_scalar('Loss/train_mse_raw', avg_train_mse_raw, epoch + 1)     # [V11] 纯MSE
        writer.add_scalar('Loss/train_grad', avg_train_grad, epoch + 1)
        writer.add_scalar('Loss/train_linear_peak', avg_train_linear, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        writer.add_scalar('Loss/val_mse_weighted', avg_val_mse, epoch + 1)         # [V11] 加权
        writer.add_scalar('Loss/val_mse_raw', avg_val_mse_raw, epoch + 1)         # [V11] 纯MSE
        writer.add_scalar('Loss/val_grad', avg_val_grad, epoch + 1)
        writer.add_scalar('Loss/val_linear_peak', avg_val_linear, epoch + 1)

        if (epoch + 1) % 10 == 0:
            epoch_bar.set_postfix({'Train': f'{avg_train:.4f}', 'MSE-w': f'{avg_train_mse:.4f}',
                                   'MSE-r': f'{avg_train_mse_raw:.4f}', 'Val': f'{avg_val:.4f}'})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_dict = {'model_state_dict': ema_model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'epoch': epoch, 'best_val_loss': best_val_loss,
                         'input_mean': torch.from_numpy(train_dataset.input_mean),
                         'input_std': torch.from_numpy(train_dataset.input_std)}
            if scaler is not None: save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！"); writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PI-MBCN V11 训练脚本 (频率加权损失)')
    parser.add_argument('--resume', type=str, default=None)
    train_model(resume_path=parser.parse_args().resume)
