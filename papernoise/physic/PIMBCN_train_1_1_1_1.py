import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from PIMBCN_net_1_1_1_1 import PI_MBCN, physics_informed_loss_fn
from PIMBCN_data import PIMBCNDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # 架构优化后显存占用大幅降低，提升批次大小以加速收敛并稳定梯度
    batch_size = 16  
    epochs = 1500 
    
    # 定义基础学习率
    head_learning_rate = 5e-4   # 任务头的学习率
    shared_learning_rate = 1e-4 # 共享主干的学习率
    freq_bins = 2501
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f'pi_mbcn_hvac_MTL_{timestamp}_epochs{epochs}_bs{batch_size}_lr{head_learning_rate}'
    writer = SummaryWriter(f'runs/{run_name}')
    
    best_val_loss = float('inf')
    save_dir = f'runs/{run_name}/models'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"训练时间戳: {timestamp}")
    
    data_directory = "E:\\lyh\\paddlespeech\\csvdata333"
    
    train_dataset = PIMBCNDataset(
        directory_path=data_directory,
        input_cols=[4, 5, 6], 
        oaspl_col=11, octave_col=12, spectrum_col=13, 
        type_col=3, mode_col=2, 
        val_split=0.2, is_validation=False
    )
    
    val_dataset = train_dataset.get_validation_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 实例化重构后的参数共享多分支网络
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    
    # ================= 核心修改：解耦学习率设置 =================
    # 为共享模块与独立任务头分配不同的优化步长
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': shared_learning_rate},
        {'params': model.shared_decoder_body.parameters(), 'lr': shared_learning_rate},
        {'params': model.heads.parameters(), 'lr': head_learning_rate}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    
    # 学习率调度器：余弦退火配合 Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    
    # 尝试保存计算图
    sample_input = torch.randn(1, 3).to(device)
    sample_mode = torch.tensor([0], dtype=torch.long).to(device)
    sample_type = torch.tensor([0], dtype=torch.long).to(device)
    try:
        writer.add_graph(model, (sample_input, sample_mode, sample_type))
    except Exception as e:
        print(f"跳过计算图保存: {e}")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_mse_spec_loss = 0.0
        total_cosine_spec_loss = 0.0
        total_mse_oaspl_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for batch_idx, (inputs, types, modes, _, _, target_spectrum) in enumerate(train_bar):
            inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
            target_spectrum = target_spectrum.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播：已内置动态掩码路由
            pred_spectrum = model(inputs, modes, types)
            
            # 计算融合了物理信息约束的损失
            loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl = physics_informed_loss_fn(
                pred_spectrum, target_spectrum
            )
            
            loss.backward()
            
            # ================= 核心修改：物理引导模型的数值稳定性保护 =================
            # 引入全局梯度裁剪，防止指数项计算导致的异常梯度突变
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_mse_spec_loss += loss_mse_spec.item()
            total_cosine_spec_loss += loss_cosine_spec.item()
            total_mse_oaspl_loss += loss_mse_oaspl.item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'mse_spec': f'{loss_mse_spec.item():.4f}',
                'cosine': f'{loss_cosine_spec.item():.4f}'
            })
            
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_mse_loss = val_spectrum_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] Validation', leave=False)
        with torch.no_grad():
            for inputs, types, modes, _, _, target_spectrum in val_bar:
                inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
                target_spectrum = target_spectrum.to(device)
                
                pred_spectrum = model(inputs, modes, types)
                
                spectrum_loss = F.mse_loss(pred_spectrum, target_spectrum).item()
                
                val_spectrum_loss += spectrum_loss
                val_mse_loss += spectrum_loss
                
        # 记录与统计
        avg_train = total_train_loss / len(train_loader)
        avg_train_mse_spec_loss = total_mse_spec_loss / len(train_loader)
        avg_train_cosine_spec_loss = total_cosine_spec_loss / len(train_loader)
        avg_train_mse_oaspl_loss = total_mse_oaspl_loss / len(train_loader)
        avg_spectrum_val = val_spectrum_loss / len(val_loader)
        avg_val = val_mse_loss / len(val_loader)
        
        # Tensorboard 曲线绘制
        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec_loss', avg_train_mse_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec_loss', avg_train_cosine_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl', avg_train_mse_oaspl_loss, epoch + 1)
        writer.add_scalar('Loss/val_spectrum', avg_spectrum_val, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        
        # 动态记录当前多组学习率情况
        current_lr_shared = optimizer.param_groups[0]['lr']
        current_lr_head = optimizer.param_groups[2]['lr']
        writer.add_scalar('Learning_Rate/shared_trunk', current_lr_shared, epoch + 1)
        writer.add_scalar('Learning_Rate/task_heads', current_lr_head, epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.4f} | Val MSE: {avg_val:.4f} | LR (Head): {current_lr_head:.2e}")
        
        # 最佳模型持久化
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！")
    writer.close()

if __name__ == "__main__":
    train_model()