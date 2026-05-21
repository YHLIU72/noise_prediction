import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from PIMBCN_net_1_1_1 import PI_MBCN, physics_informed_loss_fn
from PIMBCN_data import PIMBCNDataset

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    batch_size = 4  # 模型参数增加，减小batch_size防止显存溢出
    epochs = 300  # 增加训练轮数，给更大的模型更多训练时间
    learning_rate = 5e-4  # 降低学习率，适配更大的模型
    freq_bins = 2501
    

    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/pi_mbcn_hvac_{timestamp}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dir_csvdata333')
    
    best_val_loss = float('inf')
    save_dir = f'runs/pi_mbcn_hvac_{timestamp}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_dir_csvdata333/models'
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    
    # 优化器设置：使用AdamW，适度正则化
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器：余弦退火，配合warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    
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
            
            pred_spectrum = model(inputs, modes, types)
            
            # 计算损失
            loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl = physics_informed_loss_fn(
                pred_spectrum, target_spectrum
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_mse_spec_loss += loss_mse_spec.item()
            total_cosine_spec_loss += loss_cosine_spec.item()
            total_mse_oaspl_loss += loss_mse_oaspl.item()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}', 
                                  'mse_spec': f'{loss_mse_spec.item():.4f}',
                                  'cosine': f'{loss_cosine_spec.item():.4f}'})
            
        scheduler.step()
        
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
                
        avg_train = total_train_loss / len(train_loader)
        avg_train_mse_spec_loss = total_mse_spec_loss / len(train_loader)
        avg_train_cosine_spec_loss = total_cosine_spec_loss / len(train_loader)
        avg_train_mse_oaspl_loss = total_mse_oaspl_loss / len(train_loader)
        avg_spectrum_val = val_spectrum_loss / len(val_loader)
        avg_val = val_mse_loss / len(val_loader)
        
        writer.add_scalar('Loss/train_total', avg_train, epoch + 1)
        writer.add_scalar('Loss/train_mse_spec_loss', avg_train_mse_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_cosine_spec_loss', avg_train_cosine_spec_loss, epoch + 1)
        writer.add_scalar('Loss/train_mse_oaspl', avg_train_mse_oaspl_loss, epoch + 1)
        writer.add_scalar('Loss/val_spectrum', avg_spectrum_val, epoch + 1)
        writer.add_scalar('Loss/val_total', avg_val, epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train:.4f} | Val MSE: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_dir, 'best_model.pth'))

    print("模型训练完成！")
    writer.close()

if __name__ == "__main__":
    train_model()