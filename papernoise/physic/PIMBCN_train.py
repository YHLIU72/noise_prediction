import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from PIMBCN_net import PI_MBCN, physics_informed_loss_fn
from PIMBCN_data import PIMBCNDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    batch_size = 8 
    epochs = 200
    learning_rate = 1e-3
    freq_bins = 2501
    
    # 获取当前训练时间
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    writer = SummaryWriter(f'runs/pi_mbcn_hvac_{timestamp}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dir_csvdata333')
    
    best_val_loss = float('inf')
    save_dir = f'runs/pi_mbcn_hvac_{timestamp}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dir_csvdata333/models'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"训练时间戳: {timestamp}")
    
    # 请替换为实际路径
    data_directory = "E:\\lyh\\paddlespeech\\csvdata333"
    
    train_dataset = PIMBCNDataset(
        directory_path=data_directory,
        input_cols=[4, 5, 6], 
        oaspl_col=11, octave_col=12, spectrum_col=13, 
        type_col=3, mode_col=2, 
        val_split=0.2, is_validation=False
    )
    
    # 获取 RPM 反归一化参数，准备送入 GPU
    rpm_mean_val, rpm_std_val = train_dataset.get_rpm_norm_params()
    rpm_mean = torch.tensor(rpm_mean_val, dtype=torch.float32, device=device)
    rpm_std = torch.tensor(rpm_std_val, dtype=torch.float32, device=device)
    print(f"提取RPM反归一化参数 => 均值: {rpm_mean_val:.2f}, 标准差: {rpm_std_val:.2f}")
    
    val_dataset = train_dataset.get_validation_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
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
        total_phy_loss = 0.0
        total_mse_spec_loss = 0.0
        total_cosine_spec_loss = 0.0
        total_mse_oaspl_loss = 0.0
        total_mse_octave_loss = 0.0
        batch_alphas = [] 
        
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for batch_idx, (inputs, types, modes, target_oaspl, target_octave, target_spectrum) in enumerate(train_bar):
            inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
            target_oaspl = target_oaspl.to(device)
            target_octave = target_octave.to(device)
            target_spectrum = target_spectrum.to(device)
            
            inputs.requires_grad_(True)
            optimizer.zero_grad()
            
            pred_oaspl, pred_octave, pred_spectrum, alpha = model(inputs, modes, types)
            batch_alphas.append(alpha.mean().item()) # 提取整个批次的平均alpha
            
            # 【修复点：将均值和标准差送入物理损失函数】
            loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl, loss_mse_octave, loss_physics = physics_informed_loss_fn(
                pred_oaspl, pred_octave, pred_spectrum, alpha, 
                target_oaspl, target_octave, target_spectrum, 
                inputs, rpm_mean, rpm_std, lambda_phy=2.0
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_mse_spec_loss += loss_mse_spec.item()
            total_cosine_spec_loss += loss_cosine_spec.item()
            total_mse_oaspl_loss += loss_mse_oaspl.item()
            total_mse_octave_loss += loss_mse_octave.item()
            total_phy_loss += loss_physics.item()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 'phy_loss': f'{loss_physics.item():.4f}', 
                                  'mse_spec_loss': f'{loss_mse_spec.item():.4f}', 
                                  'cosine_spec_loss': f'{loss_cosine_spec.item():.4f}', 
                                  'mse_oaspl_loss': f'{loss_mse_oaspl.item():.4f}', 
                                  'mse_octave_loss': f'{loss_mse_octave.item():.4f}'})
            
        scheduler.step()
        epoch_avg_alpha = sum(batch_alphas) / len(batch_alphas) if batch_alphas else 6.0
        
        model.eval()
        val_mse_loss = val_oaspl_loss = val_octave_loss = val_spectrum_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] Validation', leave=False)
        with torch.no_grad():
            for inputs, types, modes, target_oaspl, target_octave, target_spectrum in val_bar:
                inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
                target_oaspl, target_octave, target_spectrum = target_oaspl.to(device), target_octave.to(device), target_spectrum.to(device)
                
                pred_oaspl, pred_octave, pred_spectrum, _ = model(inputs, modes, types)
                
                oaspl_loss = F.mse_loss(pred_oaspl, target_oaspl).item()
                octave_loss = F.mse_loss(pred_octave, target_octave).item()
                spectrum_loss = F.mse_loss(pred_spectrum, target_spectrum).item()
                
                val_oaspl_loss += oaspl_loss
                val_octave_loss += octave_loss
                val_spectrum_loss += spectrum_loss
                val_mse_loss += (spectrum_loss + octave_loss) / 2
                
        avg_train = total_train_loss / len(train_loader)
        avg_train_mse_spec_loss = total_mse_spec_loss / len(train_loader)
        avg_train_cosine_spec_loss = total_cosine_spec_loss / len(train_loader)
        avg_train_mse_oaspl_loss = total_mse_oaspl_loss / len(train_loader)
        avg_train_mse_octave_loss = total_mse_octave_loss / len(train_loader)
        avg_phy = total_phy_loss / len(train_loader)
        avg_oaspl_val = val_oaspl_loss / len(val_loader)
        avg_octave_val = val_octave_loss / len(val_loader)
        avg_spectrum_val = val_spectrum_loss / len(val_loader)
        avg_val = val_mse_loss / len(val_loader)
        
        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec_loss', avg_train_mse_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec_loss', avg_train_cosine_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl_loss', avg_train_mse_oaspl_loss, epoch + 1)
        writer.add_scalar('Loss/train_mse_octave_loss', avg_train_mse_octave_loss, epoch + 1)
        writer.add_scalar('Loss/phy_penalty', avg_phy, epoch + 1)
        writer.add_scalar('Loss/val_oaspl', avg_oaspl_val, epoch + 1)
        writer.add_scalar('Loss/val_octave', avg_octave_val, epoch + 1)
        writer.add_scalar('Loss/val_spectrum', avg_spectrum_val, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        writer.add_scalar('Physics/alpha_avg', epoch_avg_alpha, epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.4f} | Phy Loss: {avg_phy:.4f} | Val MSE: {avg_val:.4f} | Avg Alpha: {epoch_avg_alpha:.3f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'model_state_dict': model.state_dict(), 'alpha_avg': epoch_avg_alpha}, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！")
    writer.close()

if __name__ == "__main__":
    train_model()