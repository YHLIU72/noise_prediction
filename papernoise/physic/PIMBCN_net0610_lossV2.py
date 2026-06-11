"""
PIMBCN 网络模型（2026-06-10 损失函数 V2 重构版）

损失函数重构:
- 移除 Cosine 相似度（全正 dB 值下退化为常数, 梯度 ≈ 0）
- 移除 LogMSE（dB 已是对数域, 双重对数无物理意义）
- 移除 OASPL 损失（与 MSE 高度冗余, 不提供新监督信号）
- 新增线性域峰值 MSE: dB→声压比, 自动侧重高峰值区域
- 保留 dB-MSE (核心) + Sobolev 梯度 (平滑约束)

新损失组成: MSE(5.0) + 线性峰值MSE(2.0) + Sobolev(3.0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================= 优化 1: 降维版傅里叶特征编码器 (增强抗过拟合) =================
class FourierFeatureEmbedding(nn.Module):
    """小样本适用的傅里叶特征映射，降低基底维度，强化正则化"""
    def __init__(self, input_dim=3, mapping_size=32, scale=3.0, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = (2.0 * math.pi * x) @ self.B
        x_ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.encoder(x_ff)

# ================= 无分支转置卷积块 =================
class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.residual_proj = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = self.residual_proj(x)
        x = F.gelu(self.bn(self.conv(x)))
        return x + residual

# ================= 通道注意力 =================
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ================= 优化 2: 轻量化解码器主干 =================
class SharedDecoderBody(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.seq_len_init = 16
        self.channels_init = 64

        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(64, 64),   # 16 -> 32
            TransposeConvBlock(64, 32),   # 32 -> 64
            TransposeConvBlock(32, 32),   # 64 -> 128
            TransposeConvBlock(32, 16),   # 128 -> 256
            TransposeConvBlock(16, 8),    # 256 -> 512
            TransposeConvBlock(8, 8),     # 512 -> 1024
        ])
        self.attention = SEBlock1D(channel=8, reduction=2)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        x_seq = self.attention(x_seq)
        return x_seq  # [B, 8, 1024]

# ================= 优化 3: 强正则化任务头 =================
class HighFidelityTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):
        super().__init__()
        self.freq_bins = freq_bins

        self.refine = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16, 1024]),
            nn.GELU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(16, 8, kernel_size=3, padding=1)
        )
        
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, freq_bins),
        )
        
        self.broadband_path = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=5, padding=2)
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        env = self.broadband_path(x_seq)
        peaks = self.tonal_peak_path(x_seq)
        spectrum = (env + peaks).squeeze(1)
        return spectrum

# ================= 主模型 =================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246):
        super().__init__()
        self.num_modes = num_modes
        self.num_types = num_types

        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=256)

        self.heads = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                self.heads[f"mode_{m}_type_{t}"] = HighFidelityTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        shared_hidden = self.shared_encoder(x)
        shared_features = self.shared_decoder_body(shared_hidden)

        batch_size = x.size(0)
        spectrum_out = torch.zeros(
            batch_size, self.heads["mode_0_type_0"].freq_bins,
            device=x.device, dtype=shared_features.dtype
        )

        combo_idx = mode_idx * self.num_types + type_idx
        unique_combos = torch.unique(combo_idx)

        for combo in unique_combos:
            m = (combo // self.num_types).long().item()
            t = (combo % self.num_types).long().item()
            mask = (combo_idx == combo)
            branch_features = shared_features[mask]
            spectrum_out[mask] = self.heads[f"mode_{m}_type_{t}"](branch_features).to(spectrum_out.dtype)

        return spectrum_out

# ================= 物理损失 V2: 精简三项 =================
class PhysicsLossWrapper(nn.Module):
    """
    损失函数 V2 — 移除无效项, 新增线性域峰值约束
    
    三项损失:
    1. dB-MSE (5.0):   核心逐点精度, 在 dB 域直接优化
    2. 线性峰值MSE (2.0): dB→线性声压比, 自动侧重高峰值区域
    3. Sobolev 梯度 (3.0):  约束频谱平滑性, 防止高频振荡
    """
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0      # dB 域 MSE
        self.weight_linear = 2.0   # 线性域峰值 MSE (替代 LogMSE)
        self.weight_grad = 3.0     # Sobolev 梯度平滑

    def _sobolev_gradient_loss(self, pred, target):
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        return F.mse_loss(diff_pred, diff_target)

    def _linear_peak_loss(self, pred_db, target_db):
        """
        dB → 线性声压比, 自动侧重高峰值:
        - 60dB 处 1dB 误差 → 线性域 Δ≈115
        - 20dB 处 1dB 误差 → 线性域 Δ≈1.2
        约 100:1 的权重差异, 自然聚焦高声压区 (物理上更重要的区域)
        """
        # 用 target 的峰值做参考, 确保数值在 float16 安全范围内
        ref = torch.max(target_db.detach(), dim=1, keepdim=True)[0]
        pred_lin = torch.pow(10.0, (pred_db - ref) / 20.0)
        target_lin = torch.pow(10.0, (target_db - ref) / 20.0)
        return F.mse_loss(pred_lin, target_lin)

    def forward(self, pred_spectrum, target_spectrum):
        loss_mse = F.mse_loss(pred_spectrum, target_spectrum)
        loss_linear = self._linear_peak_loss(pred_spectrum, target_spectrum)
        loss_grad = self._sobolev_gradient_loss(pred_spectrum, target_spectrum)

        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_linear * loss_linear +
            self.weight_grad * loss_grad
        )

        return total_loss, loss_mse, loss_grad, loss_linear
