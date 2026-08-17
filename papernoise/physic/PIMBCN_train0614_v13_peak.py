"""
V13 峰值增强微调 (Peak-Enhanced Fine-tuning)
- 基于 V13 best_model 续训
- 大幅提高 LinearPeakMSE 权重 (2.0→20.0) 以改善峰值低估
- 降低 Sobolev 梯度权重 (3.0→1.5) 允许更尖锐的峰值
- 较低学习率 (1e-4) 稳定微调
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch, copy, time
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from PIMBCN_net0614_v13 import PI_MBCN, FourierFeatureEmbedding, SharedDecoderBody, SharedTaskHead
from PIMBCN_data_0614_v13 import PIMBCNDataset
from PIMBCN_net0614_v13 import PhysicsLossWrapper as OriginalLossWrapper

# ============================================================
# 峰值增强损失包装器 (继承自原版，仅修改权重)
# ============================================================
class PeakEnhancedLossWrapper(torch.nn.Module):
    """峰值增强版: LinearPeak权重 2.0→20.0, Sobolev 3.0→1.5"""
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0
        self.weight_linear = 20.0    # ★ 10× 增强峰值约束
        self.weight_grad = 1.5       # ★ 减半, 允许更尖锐的峰值
        self.ms_w2 = 0.3
        self.ms_w4 = 0.1

    def _multiscale_mse(self, pred, target):
        loss = F.mse_loss(pred, target)
        p2 = F.avg_pool1d(pred.unsqueeze(1), 2, 2).squeeze(1)
        t2 = F.avg_pool1d(target.unsqueeze(1), 2, 2).squeeze(1)
        loss = loss + self.ms_w2 * F.mse_loss(p2, t2)
        p4 = F.avg_pool1d(pred.unsqueeze(1), 4, 4).squeeze(1)
        t4 = F.avg_pool1d(target.unsqueeze(1), 4, 4).squeeze(1)
        loss = loss + self.ms_w4 * F.mse_loss(p4, t4)
        return loss

    def _sobolev_gradient_loss(self, pred, target):
        return F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

    def _linear_peak_loss(self, pred_db, target_db):
        ref = torch.max(target_db.detach(), dim=1, keepdim=True)[0]
        pred_lin = torch.pow(10.0, (pred_db - ref) / 20.0)
        target_lin = torch.pow(10.0, (target_db - ref) / 20.0)
        return F.mse_loss(pred_lin, target_lin)

    def forward(self, pred, target):
        loss_mse = self._multiscale_mse(pred, target)
        loss_linear = self._linear_peak_loss(pred, target)
        loss_grad = self._sobolev_gradient_loss(pred, target)
        total = self.weight_mse * loss_mse + self.weight_linear * loss_linear + self.weight_grad * loss_grad
        return total, loss_mse, loss_grad, loss_linear


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    torch.backends.cudnn.benchmark = True

    # ===================== 超参数 =====================
    batch_size = 8
    accum_steps = 2
    epochs = 10000           # 微调轮数
    learning_rate = 1e-4     # ★ 降低学习率 (原3e-4)
    freq_bins = 1246
    weight_decay_val = 5e-4  # ★ 略增正则化防过拟合
    ema_decay = 0.999

    # ===================== 加载V13 checkpoint =====================
    resume_path = r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_v13_20260615_141745\models\best_model.pth"
    print(f"从V13最优模型续训: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f'pi_mbcn_v13_peak_{timestamp}'
    save_dir = f'runs/{run_name}/models'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(f'runs/{run_name}')
    
    start_epoch = 0
    best_val_loss = float('inf')

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
    print(f"频率: 20~5000Hz 对数采样, 频点数: {freq_bins}")

    # ===================== 模型 + EMA =====================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载V13权重
    loss_wrapper = PeakEnhancedLossWrapper().to(device)     # ★ 新损失权重
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters(): p.requires_grad_(False)
    
    print(f"★ 峰值增强损失: MSE={loss_wrapper.weight_mse}, "
          f"LinearPeak={loss_wrapper.weight_linear} (原2.0→20.0), "
          f"Sobolev={loss_wrapper.weight_grad} (原3.0→1.5)")

    # ===================== 优化器 + 调度器 =====================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': learning_rate},
        {'params': model.mode_embed.parameters(), 'lr': learning_rate},
        {'params': model.type_embed.parameters(), 'lr': learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': learning_rate},
        {'params': model.shared_head.parameters(), 'lr': learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay_val)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-7)

    # ===================== LR warmup =====================
    warmup_epochs = 10  # ★ 更长warmup, 稳定过渡
    warmup_steps = warmup_epochs * len(train_loader)
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    global_step = 0

    # ===================== 混合精度 =====================
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    amp_enabled = scaler is not None

    # ===================== 训练循环 =====================
    epoch_bar = tqdm(range(start_epoch, epochs), desc='Peak Fine-tuning')
    for epoch in epoch_bar:
        model.train(); loss_wrapper.train()
        train_loss_sum = 0.0; train_mse_sum = 0.0
        train_grad_sum = 0.0; train_linear_sum = 0.0
        train_count = 0; optimizer.zero_grad(set_to_none=True)

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

            train_loss_sum += loss.item()
            train_mse_sum += l_mse.item()
            train_grad_sum += l_grad.item()
            train_linear_sum += l_linear.item()
            train_count += 1; global_step += 1

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
        val_loss_sum = 0.0; val_mse_sum = 0.0
        val_grad_sum = 0.0; val_linear_sum = 0.0
        val_count = 0
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
                val_loss_sum += v_loss.item()
                val_mse_sum += v_mse.item()
                val_grad_sum += v_grad.item()
                val_linear_sum += v_linear.item()
                val_count += 1

        # ---- 日志 ----
        avg_train_loss = train_loss_sum / train_count
        avg_val_loss = val_loss_sum / val_count
        avg_train_mse = train_mse_sum / train_count
        avg_val_mse = val_mse_sum / val_count

        writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        writer.add_scalar('Loss/train_mse_db', avg_train_mse, epoch)
        writer.add_scalar('Loss/val_mse_db', avg_val_mse, epoch)
        writer.add_scalar('Loss/train_linear_peak', train_linear_sum/train_count, epoch)
        writer.add_scalar('Loss/val_linear_peak', val_linear_sum/val_count, epoch)
        writer.add_scalar('Loss/train_grad', train_grad_sum/train_count, epoch)
        writer.add_scalar('Loss/val_grad', val_grad_sum/val_count, epoch)

        epoch_bar.set_postfix({
            'T_loss': f'{avg_train_loss:.2f}', 'V_loss': f'{avg_val_loss:.2f}',
            'V_mse': f'{avg_val_mse:.3f}',
        })

        # ---- 保存最佳模型 ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dict = {
                'model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'best_val_loss': best_val_loss,
                'input_mean': torch.from_numpy(train_dataset.input_mean),
                'input_std': torch.from_numpy(train_dataset.input_std),
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
            epoch_bar.set_postfix({
                'T_loss': f'{avg_train_loss:.2f}', 'V_loss': f'{avg_val_loss:.2f}',
                'V_mse': f'{avg_val_mse:.3f}', '★': 'SAVED'
            })

    writer.close()
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {save_dir}/best_model.pth")


if __name__ == "__main__":
    train_model()
