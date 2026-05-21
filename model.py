#导入库
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def load_data_from_csv(csv_path, n_mels=80, fft_window=1280, hop_length=320):
    """
    从CSV文件加载音频路径和工况参数，生成梅尔频谱特征
    
    参数:
        csv_path (str): CSV文件路径
        n_mels (int): 梅尔滤波器数量
        fft_window (int): FFT窗口大小
        hop_length (int): 帧移大小
    
    返回:
        tuple: (param_list, mel_list)
            param_list: 形状为(N, 3)的工况参数列表
            mel_list: 形状为(N, n_mels, T)的梅尔频谱列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    param_list = []
    mel_list = []
    
    # 遍历CSV中的每个音频文件
    for idx, row in df.iterrows():
        try:
            # 提取音频路径和工况参数（第五、六、七列）
            audio_path = row.iloc[0]  # 第一列：音频文件路径
            params = row.iloc[4:7].values.astype(float)  # 第五至七列：工况参量
            
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=None)
            
            # 生成梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=fft_window,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # 转换为分贝刻度
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 添加到列表
            param_list.append(params)
            mel_list.append(mel_spec_db)
            
            if idx % 10 == 0:
                print(f"已处理 {idx+1}/{len(df)} 个音频文件")
                
        except Exception as e:
            print(f"处理文件 {row.iloc[0]} 时出错: {str(e)}")
            continue
    
    return np.array(param_list), np.array(mel_list), sr

# 定义模型
class MelDataset(Dataset):
    def __init__(self, param_list, mel_list, mel_mean=None, mel_std=None, scaler=None):
        self.params = torch.FloatTensor(param_list)  # (N, 3)
        self.mels = torch.FloatTensor(mel_list)      # (N, n_mels, T)
        
        # 标准化参数
        self.scaler = StandardScaler() if scaler is None else scaler
        if scaler is None:
            self.params = torch.FloatTensor(self.scaler.fit_transform(self.params))
        else:
            self.params = torch.FloatTensor(self.scaler.transform(self.params))
        #  新增：标准化梅尔频谱到 [-1,1]
        if mel_mean is None or mel_std is None:
            # 训练集：计算均值和标准差
            self.mel_mean = self.mels.mean()
            self.mel_std = self.mels.std()
        else:
            # 验证集：复用训练集的均值和标准差
            self.mel_mean = mel_mean
            self.mel_std = mel_std
        
        # 归一化到 [-1,1]（假设分布近似正态，超出范围的会被截断）
        self.mels = (self.mels - self.mel_mean) / (self.mel_std + 1e-8)  # 标准化
        self.mels = torch.clamp(self.mels, -1, 1)  # 截断极端值
        
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        return self.params[idx], self.mels[idx].unsqueeze(0)  # (3), (1, n_mels, T)
    
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_mels=80, T=52):
        super().__init__()
        self.n_mels = n_mels
        self.T = T
        
        # 将条件 + 噪声投影到初始特征图
        self.projection = nn.Sequential(
            nn.Linear(3 + latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        # 转置卷积上采样
        self.upconv = nn.Sequential(
            # 输入: (512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # -> (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 调整最后一层转置卷积参数以输出 (1, 80, 52)
            nn.ConvTranspose2d(32, 1, kernel_size=(17, 1), stride=(1, 1), padding=(0, 6)),  # -> (1, 80, 52)
            nn.Tanh()
        )
    
    def forward(self, z, c):
        # z: (batch, latent_dim), c: (batch, 3)
        x = torch.cat([z, c], dim=1)  # (batch, latent_dim + 3)
        x = self.projection(x)
        x = x.view(-1, 512, 4, 4)     # (batch, 512, 4, 4)
        x = self.upconv(x)            # (batch, 1, n_mels, T)
        return x
class Discriminator(nn.Module):
    def __init__(self, n_mels=80, T=52):
        super().__init__()
        
        # 梅尔谱的卷积处理
        self.conv = nn.Sequential(
            # 输入: (1, n_mels, T)
            nn.Conv2d(1, 32, 4, 2, 1),  # -> (32, n_mels//2, T//2)
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (64, n_mels//4, T//4)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1), # -> (128, n_mels//8, T//8)
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1), # -> (256, n_mels//16, T//16)
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 条件处理
        self.condition_fc = nn.Linear(3, n_mels * T // 16)
        
        # 最终判别
        self.final = nn.Sequential(
            nn.Linear(256 * (n_mels//16) * (T//16) + n_mels * T // 16, 1),
            # 注意：WGAN-GP 不使用 Sigmoid！
        )
    
    def forward(self, x, c):
        # x: (batch, 1, n_mels, T), c: (batch, 3)
        x_feat = self.conv(x)  # (batch, 256, n_mels//16, T//16)
        x_feat = x_feat.view(x_feat.size(0), -1)  # (batch, 256 * (n_mels//16) * (T//16))
        
        c_proj = self.condition_fc(c)  # (batch, n_mels * T // 16)
        
        x_combined = torch.cat([x_feat, c_proj], dim=1)
        return self.final(x_combined)  # (batch, 1)
    
# 超参数
latent_dim = 100
batch_size = 32
lr = 0.0002
epochs = 1000
lambda_gp = 10  # WGAN-GP 梯度惩罚系数
lambda_l1 = 100  # L1 Loss 权重

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

# 优化器
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# 新增：学习率调度器（当验证损失不再改善时降低学习率）
scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
    opt_G, 
    mode='min',          # 监控指标最小化（验证L1损失）
    factor=0.5,          # 学习率衰减因子（降低50%）
    patience=10,         # 10个epoch无改善则衰减
    min_lr=1e-6       # 最小学习率
    # verbose=True         # 打印学习率调整信息
)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
    opt_D, 
    mode='min', 
    factor=0.5, 
    patience=10, 
    min_lr=1e-6
    # verbose=True
)

# 日志记录
writer = SummaryWriter("runs/experiment_1")

# 数据加载
param_list, mel_list, sr = load_data_from_csv(
    "MAR2 EVA2.csv",  # 替换为实际CSV文件路径
    n_mels=80, 
    fft_window=1280, 
    hop_length=320
)
# 新增：划分训练集和验证集（80%训练，20%验证）
param_train, param_val, mel_train, mel_val = train_test_split(
    param_list, mel_list, 
    test_size=0.2,  # 验证集比例
    random_state=42  # 固定随机种子确保可复现
)

# 新增：创建训练集和验证集数据集（共享训练集的标准化参数）
train_dataset = MelDataset(param_train, mel_train)  # 训练集：拟合标准化器
val_dataset = MelDataset(param_val, mel_val, 
                          mel_mean=train_dataset.mel_mean, 
                          mel_std=train_dataset.mel_std, 
                          scaler=train_dataset.scaler)  # 验证集：使用训练集的标准化器

# 新增：创建训练集和验证集数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集不打乱

# dataset = MelDataset(param_list, mel_list)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(epochs):
    G.train()  # 启用训练模式（BatchNorm等层生效）
    D.train()
    train_d_loss = 0.0
    train_g_loss = 0.0
    for i, (real_c, real_mel) in enumerate(train_dataloader):
        real_c = real_c.to(device)
        real_mel = real_mel.to(device)
        batch_size = real_c.size(0)
        
        for _ in range(5):  # 判别器训练5次
            # ---- 训练判别器 D ----
            opt_D.zero_grad()
            
            # 真实样本
            real_validity = D(real_mel, real_c)
            
            # 生成样本
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_mel = G(z, real_c)
            fake_validity = D(fake_mel.detach(), real_c)
            
            # 梯度惩罚 (WGAN-GP)
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolated = (alpha * real_mel + (1 - alpha) * fake_mel.detach()).requires_grad_(True)
            interpolated_validity = D(interpolated, real_c)
            
            # 计算梯度惩罚
            grad = torch.autograd.grad(
                outputs=interpolated_validity,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interpolated_validity),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
            grad_penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
            
            # 判别器损失
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + grad_penalty
            d_loss.backward()
            opt_D.step()

        
        # ---- 训练生成器 G ----
        opt_G.zero_grad()
        
        fake_validity = D(fake_mel, real_c)
        g_loss = -torch.mean(fake_validity)
        
        # L1 Loss
        l1_loss = nn.L1Loss()(fake_mel, real_mel)
        
        # 总损失
        total_g_loss = g_loss + lambda_l1 * l1_loss
        total_g_loss.backward()
        opt_G.step()

        # 累计损失
        train_d_loss += d_loss.item() * batch_size
        train_g_loss += total_g_loss.item() * batch_size

        # 记录损失
        writer.add_scalar("D Loss", d_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar("G Loss", total_g_loss.item(), epoch * len(train_dataloader) + i)
        writer.add_scalar("G L1 Loss", l1_loss.item(), epoch * len(train_dataloader) + i)
        
        # 打印训练信息
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {total_g_loss.item():.4f}")
            
    # 计算训练集平均损失
    train_d_loss_avg = train_d_loss / len(train_dataloader.dataset)
    train_g_loss_avg = train_g_loss / len(train_dataloader.dataset)
    # 记录训练集平均损失
    writer.add_scalar("Train D Loss", train_d_loss_avg, epoch)
    writer.add_scalar("Train G Loss", train_g_loss_avg, epoch)
    # ---- 验证阶段 ----
    G.eval()  # 启用评估模式（BatchNorm等层固定）
    D.eval()
    val_l1_loss = 0.0
    
    with torch.no_grad():  # 关闭梯度计算，节省显存和计算资源
        for real_c, real_mel in val_dataloader:
            real_c = real_c.to(device)
            real_mel = real_mel.to(device)
            
            # 生成验证集样本
            z = torch.randn(real_c.size(0), latent_dim).to(device)
            fake_mel = G(z, real_c)
            
            # 计算验证集L1损失（生成质量指标）
            val_l1_loss += nn.L1Loss()(fake_mel, real_mel).item() * real_c.size(0)
             # 新增：绘制对比图（每100个epoch保存一次，取第一个batch的第一个样本）
            if (epoch % 100 == 0 or epoch == epochs-1):
                # 创建保存目录
                os.makedirs("mel_comparison", exist_ok=True)
                
                # 选择第一个样本（移除批次和通道维度）
                real_sample = real_mel[0].cpu().squeeze().numpy()  # (n_mels, T)
                fake_sample = fake_mel[0].cpu().squeeze().numpy()  # (n_mels, T)
                # ---- 新增：逆归一化，映射回原始分贝范围 ----
                real_sample = real_sample * train_dataset.mel_std.item() + train_dataset.mel_mean.item()  # 真实样本逆归一化
                fake_sample = fake_sample * train_dataset.mel_std.item() + train_dataset.mel_mean.item()  # 生成样本逆归一化
                # print (real_sample.shape, fake_sample.shape)
                # exit(0)
                # 绘制对比图
                plt.figure(figsize=(12, 8))
                
                # 真实梅尔频谱
                plt.subplot(2, 1, 1)
                librosa.display.specshow(
                    real_sample, 
                    x_axis='time', 
                    y_axis='mel', 
                    sr=sr,  # 使用数据加载时的采样率
                    hop_length=320
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Epoch {epoch} - Real Mel Spectrogram')
                
                # 生成梅尔频谱
                plt.subplot(2, 1, 2)
                librosa.display.specshow(
                    fake_sample, 
                    x_axis='time', 
                    y_axis='mel', 
                    sr=sr, 
                    hop_length=320
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Epoch {epoch} - Generated Mel Spectrogram')
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(f"mel_comparison/epoch_{epoch}_comparison.png")
                plt.close()
    
    # 计算验证集平均损失
    val_l1_loss_avg = val_l1_loss / len(val_dataset)
    writer.add_scalar("Val L1 Loss", val_l1_loss_avg, epoch)
    # 更新学习率调度器（基于验证集L1损失）
    scheduler_G.step(val_l1_loss_avg)
    scheduler_D.step(val_l1_loss_avg)

    print(f"[Epoch {epoch}/{epochs}] Train D Loss: {train_d_loss_avg:.4f}, Train G Loss: {train_g_loss_avg:.4f}, Val L1 Loss: {val_l1_loss_avg:.4f}")

    
writer.close()