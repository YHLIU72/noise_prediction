import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    
    return np.array(param_list), np.array(mel_list)
# -------------------------- 数据预处理 --------------------------
class ConditionDataset(Dataset):

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


# -------------------------- 转置卷积生成网络 --------------------------
class MelGenerator(nn.Module):
    def __init__(self, n_mels=80, T=52, latent_dim=64):
        """
        基于转置卷积的梅尔频谱生成网络
        :param n_mels: 梅尔滤波器数量（梅尔频谱高度）
        :param T: 梅尔频谱时间帧数（梅尔频谱宽度）
        :param latent_dim: 工况参数投影后的隐空间维度
        """
        super().__init__()
        self.n_mels = n_mels
        self.T = T
        
        # 1. 工况参数投影：将3维参数映射到隐空间并reshape为初始特征图
        self.condition_proj = nn.Sequential(
            nn.Linear(3, latent_dim * 4 * 4),  # 3维参数 → [latent_dim*4*4]
            nn.BatchNorm1d(latent_dim * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        # 2. 转置卷积上采样模块：逐步扩大特征图尺寸至 [n_mels, T]
        self.upconv_blocks = nn.Sequential(
            # 输入特征图：[latent_dim, 4, 4]
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # → [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # → [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # → [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 输出层：调整 kernel/stride/padding 以匹配目标尺寸 [1, n_mels, T]
            nn.ConvTranspose2d(32, 1, 
                              kernel_size=(17, 5),  # 高度方向 kernel=17：64→80 (64+17-1=80)；宽度方向 kernel=5：64→52 (64+5-1-2*padding=52)
                              stride=(1, 1), 
                              padding=(0, 8)),      # 宽度方向 padding=8：(64 + 2*8 -5)/1 +1 = 52
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )

    def forward(self, condition_params):
        """
        :param condition_params: 标准化后的工况参数，形状 [batch_size, 3]
        :return: 生成的梅尔频谱，形状 [batch_size, 1, n_mels, T]
        """
        # 1. 投影工况参数并reshape为初始特征图
        x = self.condition_proj(condition_params)  # [batch_size, latent_dim*4*4]
        x = x.view(-1, 64, 4, 4)  # [batch_size, latent_dim=64, 4, 4]
        
        # 2. 转置卷积上采样生成梅尔频谱
        mel_out = self.upconv_blocks(x)  # [batch_size, 1, n_mels, T]
        
        # 3. （可选）将 [-1,1] 映射回原始分贝范围（推理时使用）
        # mel_out = mel_out * self.mel_std + self.mel_mean
        
        return mel_out

# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # -------------------------- 1. 配置训练参数 --------------------------
    batch_size = 32
    lr = 0.0002  # 学习率（与GAN训练保持一致，便于对比）
    epochs = 100  # 训练轮次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    save_path = "trained_generator.pth"  # 模型保存路径

    # -------------------------- 2. 数据加载与预处理 --------------------------
    # 加载CSV数据（工况参数+真实梅尔频谱）
    param_list, mel_list = load_data_from_csv(
        "MAR2 EVA2.csv",  # 替换为实际CSV文件路径
        n_mels=80, 
        fft_window=1280, 
        hop_length=320
    )
    # 划分训练集和验证集（80%训练，20%验证）
    param_train, param_val, mel_train, mel_val = train_test_split(
        param_list, mel_list, 
        test_size=0.2, 
        random_state=42
    )
    # 创建数据集（共享训练集的标准化参数）
    train_dataset = ConditionDataset(param_train, mel_train)
    val_dataset = ConditionDataset(
        param_val, mel_val, 
        mel_mean=train_dataset.mel_mean, 
        mel_std=train_dataset.mel_std, 
        scaler=train_dataset.scaler
    )
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------- 3. 初始化模型、损失函数和优化器 --------------------------
    generator = MelGenerator(n_mels=80, T=52).to(device)  # 生成器移到设备
    criterion = nn.L1Loss()  # L1损失：适合保留梅尔频谱细节
    optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)  # Adam优化器，常用参数组合
    )

    # -------------------------- 4. 训练循环 --------------------------
    best_val_loss = float('inf')  # 记录最佳验证损失（用于保存最优模型）
    for epoch in range(epochs):
        # -------------------------- 训练阶段 --------------------------
        generator.train()  # 启用训练模式（BatchNorm生效）
        train_loss = 0.0
        for batch_idx, (params, real_mels) in enumerate(train_dataloader):
            # 数据移到设备
            params = params.to(device)  # 工况参数 [batch_size, 3]
            real_mels = real_mels.to(device)  # 真实梅尔频谱 [batch_size, 1, 80, 52]
            
            # 生成器前向传播：输入工况参数，生成梅尔频谱
            fake_mels = generator(params)  # [batch_size, 1, 80, 52]
            
            # 计算损失（生成频谱 vs 真实频谱）
            loss = criterion(fake_mels, real_mels)
            
            # 反向传播与参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新生成器参数
            
            # 累计训练损失
            train_loss += loss.item() * params.size(0)  # 乘以batch_size，后续求平均
            
            # 打印批次训练信息
            if batch_idx % 5 == 0:  # 每5个batch打印一次
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx+1}/{len(train_dataloader)}] "
                      f"Train Loss: {loss.item():.4f}")

        # 计算训练集平均损失
        train_loss_avg = train_loss / len(train_dataset)

        # -------------------------- 验证阶段 --------------------------
        generator.eval()  # 启用评估模式（BatchNorm固定）
        val_loss = 0.0
        with torch.no_grad():  # 关闭梯度计算，节省显存
            for params, real_mels in val_dataloader:
                params = params.to(device)
                real_mels = real_mels.to(device)
                
                # 生成验证集梅尔频谱
                fake_mels = generator(params)
                # 计算验证损失
                loss = criterion(fake_mels, real_mels)
                val_loss += loss.item() * params.size(0)
        
        # 计算验证集平均损失
        val_loss_avg = val_loss / len(val_dataset)

        # -------------------------- 结果记录与模型保存 --------------------------
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}")
        
        # 保存验证损失最低的模型（避免过拟合）
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "scaler": train_dataset.scaler,  # 保存工况参数标准化器
                "mel_mean": train_dataset.mel_mean,  # 保存梅尔频谱均值
                "mel_std": train_dataset.mel_std    # 保存梅尔频谱标准差
            }, save_path)
            print(f"√ 最佳模型已保存至 {save_path} (Val Loss: {best_val_loss:.4f})")

    print("训练完成！")

    # -------------------------- 5. 生成示例（训练后） --------------------------
    # 加载最优模型
    checkpoint = torch.load(save_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    
    # 从验证集中取一个样本生成梅尔频谱
    with torch.no_grad():
        # 取验证集第一个样本的工况参数（已标准化）
        sample_params, sample_real_mel = val_dataset[0]  # params: [3], real_mel: [1,80,52]
        sample_params = sample_params.unsqueeze(0).to(device)  # [1,3]
        
        # 生成梅尔频谱
        sample_fake_mel = generator(sample_params)  # [1,1,80,52]
        
        # 打印生成结果信息
        print("\n生成示例：")
        print(f"工况参数（标准化后）：{sample_params.cpu().numpy()[0]}")
        print(f"生成梅尔频谱形状：{sample_fake_mel.shape}")
        print(f"生成梅尔频谱值范围（归一化后）：[{sample_fake_mel.min():.2f}, {sample_fake_mel.max():.2f}]")

