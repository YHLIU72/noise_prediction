"""
PIMBCN 训练脚本（Windows 极速吞吐 + 小样本超参数特化版）

修复与优化更新：
1. 解决 Windows Triton 编译崩溃：安全回退至原生 Eager 高速执行模式。
2. 修复 PyTorch AMP 警告：使用最新的 torch.amp 接口替换弃用的 torch.cuda.amp。
3. 极速吞吐：保留全量显存缓存 (In-Memory VRAM Caching) 与纯异步 GPU 累加。
4. 小样本特化超参数：Batch Size (8), Weight Decay (5e-3), 物理损失非对称初始化。
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

# 请确保模型文件名为 PIMBCN_net0529.py (即小样本高鲁棒版)
from PIMBCN_net0529 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0529 import PIMBCNDataset


def load_full_dataset_to_gpu(dataset, device):
    """将极小数据集一次性完整拉入显存，实现内存局部性极致加速"""
    print(f"正在将数据集全量预载入显存 ({len(dataset)} 条)...")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for inputs, types, modes, _, _, target_spectrum in loader:
        return (
            inputs.to(device, non_blocking=True),
            types.to(device, non_blocking=True),
            modes.to(device, non_blocking=True),
            target_spectrum.to(device, non_blocking=True)
        )


def train_model(resume_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    torch.backends.cudnn.benchmark = True

    # ================= 核心：小样本特化超参数 =================
    batch_size = 8
    epochs = 50000

    head_learning_rate = 5e-4
    shared_learning_rate = 5e-4
    freq_bins = 2501
    
    # 激进的权重衰减 (L2 正则化)，逼迫网络抛弃冗余参数
    weight_decay_val = 5e-3

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
        run_name = f'pi_mbcn_WinFast_{timestamp}_epochs{epochs}_bs{batch_size}_wd{weight_decay_val}'
        save_dir = f'runs/{run_name}/models'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/{run_name}')
        print(f"从头训练，时间戳: {timestamp}")

    # ================= 数据集全量预载入 =================
    data_directory = "F:\\lyh\\paddlespeech\\csvdata333"

    train_dataset = PIMBCNDataset(
        directory_path=data_directory, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False
    )
    val_dataset = train_dataset.get_validation_dataset()

    train_X, train_T, train_M, train_Y = load_full_dataset_to_gpu(train_dataset, device)
    val_X, val_T, val_M, val_Y = load_full_dataset_to_gpu(val_dataset, device)

    num_train_samples = train_X.size(0)
    num_val_samples = val_X.size(0)
    num_batches = num_train_samples // batch_size
    num_val_batches = max(1, num_val_samples // batch_size)

    # ================= 模型构建与物理权重干预 =================
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    loss_wrapper = PhysicsLossWrapper().to(device)

    # 非对称损失权重初始化：强制初期关注 MSE 与 Sobolev 梯度
    # prec = exp(-log_var)，所以 log_var 越小，权重越大
    # 当前配置: MSE(7.39) > Sobolev(2.72) > Cosine(1) = OASPL(1)
    if start_epoch == 0:
        with torch.no_grad():
            loss_wrapper.log_vars.data = torch.tensor([-2.0, 0.0, 0.0, -1.0], device=device)
            print("已注入物理先验：提高 MSE(7.39x) 与 Sobolev 梯度(2.72x)的初始拟合权重。")

    # ================= 优化器与调度器 =================
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.heads.parameters(), 'lr': head_learning_rate},
        {'params': loss_wrapper.parameters(), 'lr': head_learning_rate},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay_val)
    
    # 缩短重启周期 (T_0=20)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    # ================= 混合精度 (已修复 PyTorch 2.x 警告) =================
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    amp_enabled = scaler is not None

    if resume_path is not None and os.path.exists(resume_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        loss_wrapper.load_state_dict(checkpoint['loss_wrapper_state_dict'])
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
    epoch_train_loss = torch.zeros(1, device=device)
    epoch_mse_spec = torch.zeros(1, device=device)
    epoch_cosine_spec = torch.zeros(1, device=device)
    epoch_mse_oaspl = torch.zeros(1, device=device)
    epoch_grad_loss = torch.zeros(1, device=device)
    epoch_val_loss = torch.zeros(1, device=device)

    # ================= 极速训练循环 =================
    epoch_bar = tqdm(range(start_epoch, epochs), desc='Training Epochs')
    for epoch in epoch_bar:
        model.train()
        loss_wrapper.train()

        epoch_train_loss.zero_()
        epoch_mse_spec.zero_()
        epoch_cosine_spec.zero_()
        epoch_mse_oaspl.zero_()
        epoch_grad_loss.zero_()

        # GPU 生成随机排列索引
        indices = torch.randperm(num_train_samples, device=device)

        # 修复：遍历所有训练样本，跳过 batch size < 2 的情况（BatchNorm 需要至少2个样本）
        for i in range(0, num_train_samples - batch_size + 1, batch_size):
            batch_idx = indices[i:i + batch_size]
            inputs = train_X[batch_idx]
            types = train_T[batch_idx]
            modes = train_M[batch_idx]
            target_spectrum = train_Y[batch_idx]

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                # 使用最新的 torch.amp.autocast 接口
                with torch.amp.autocast('cuda'):
                    pred_spectrum = model(inputs, modes, types)
                    loss, l_mse, l_cos, l_oaspl, l_grad = loss_wrapper(pred_spectrum, target_spectrum)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_spectrum = model(inputs, modes, types)
                loss, l_mse, l_cos, l_oaspl, l_grad = loss_wrapper(pred_spectrum, target_spectrum)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # GPU 内纯异步累加
            epoch_train_loss += loss.detach()
            epoch_mse_spec += l_mse.detach()
            epoch_cosine_spec += l_cos.detach()
            epoch_mse_oaspl += l_oaspl.detach()
            epoch_grad_loss += l_grad.detach()

        scheduler.step()

        # ================= 极速验证阶段 =================
        model.eval()
        loss_wrapper.eval()
        epoch_val_loss.zero_()

        with torch.no_grad():
            # 修复：遍历所有验证样本（包括不能被 batch_size 整除的尾部）
            for i in range(0, num_val_samples, batch_size):
                inputs = val_X[i:i + batch_size]
                types = val_T[i:i + batch_size]
                modes = val_M[i:i + batch_size]
                target_spectrum = val_Y[i:i + batch_size]

                if amp_enabled:
                    with torch.amp.autocast('cuda'):
                        pred_spectrum = model(inputs, modes, types)
                        val_loss, _, _, _, _ = loss_wrapper(pred_spectrum, target_spectrum)
                else:
                    pred_spectrum = model(inputs, modes, types)
                    val_loss, _, _, _, _ = loss_wrapper(pred_spectrum, target_spectrum)

                # 修复：使用与训练一致的 PhysicsLossWrapper 计算验证损失
                epoch_val_loss += val_loss

        # ================= 日志记录 =================
        avg_train = (epoch_train_loss / num_batches).item()
        avg_mse_spec = (epoch_mse_spec / num_batches).item()
        avg_cosine_spec = (epoch_cosine_spec / num_batches).item()
        avg_mse_oaspl = (epoch_mse_oaspl / num_batches).item()
        avg_grad = (epoch_grad_loss / num_batches).item()
        avg_val = (epoch_val_loss / num_val_batches).item()

        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec', avg_mse_spec, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec', avg_cosine_spec, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl', avg_mse_oaspl, epoch + 1)
        writer.add_scalar('Loss/train_grad', avg_grad, epoch + 1)
        writer.add_scalar('Loss/val_spectrum', avg_val, epoch + 1)

        # 轻量化刷新进度条
        if (epoch + 1) % 10 == 0:
            epoch_bar.set_postfix({
                'Train': f'{avg_train:.4f}',
                'Spec': f'{avg_mse_spec:.4f}',
                'Val': f'{avg_val:.4f}'
            })

        # ================= 模型保存 =================
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

    print("模型训练完成！")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PI-MBCN 极速吞吐训练脚本 (Windows 兼容终极版)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_model(resume_path=args.resume)