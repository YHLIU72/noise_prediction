"""
PIMBCN 网络模型（2026-06-10 最终优化版）

=== 相对原始版 (0601_ncjianshao) 的变更 ===

[模型架构]
- 解码器去最后一层 deconv: [B,4,2048]→[B,8,1024], 中间维度减半
- freq_proj: Linear(2048,1246)→Linear(1024,512)+Linear(512,1246), 渐进投影
- TransposeConv: BatchNorm1d→InstanceNorm1d, 小 batch=8 统计量更稳定
- 频谱输出: 2501点(0~10kHz)→1246点(20~5kHz @4Hz)

[损失函数 V2]
- 移除 Cosine: 全正 dB 值下退化为常数, 梯度≈0
- 移除 LogMSE: dB 已是对数域, log(dB) 双重对数无物理意义
- 移除 OASPL: 与 dB-MSE 高度冗余
- 新增线性峰值MSE: dB→声压比(10^(dB/20)), 自动侧重高峰值
- 新组成: MSE(5.0) + LinearPeak(2.0) + Sobolev(3.0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 傅里叶特征编码器
# ============================================================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=32, scale=3.0, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.to(self.B.dtype)  # AMP float16 安全
        x_proj = (2.0 * math.pi * x) @ self.B
        x_ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.encoder(x_ff)


# ============================================================================
# 转置卷积块 (InstanceNorm 版: 小 batch 统计量无偏)
# ============================================================================
class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.residual_proj = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = self.residual_proj(x)
        x = F.gelu(self.norm(self.conv(x)))
        return x + residual


# ============================================================================
# 通道注意力
# ============================================================================
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


# ============================================================================
# 轻量化解码器 (6 层 deconv, 输出 [B,8,1024])
# ============================================================================
class SharedDecoderBody(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.seq_len_init = 16
        self.channels_init = 64
        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(), nn.Dropout(0.5),
        )
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(64, 64),  # 16→32
            TransposeConvBlock(64, 32),  # 32→64
            TransposeConvBlock(32, 32),  # 64→128
            TransposeConvBlock(32, 16),  # 128→256
            TransposeConvBlock(16, 8),   # 256→512
            TransposeConvBlock(8, 8),    # 512→1024
        ])
        self.attention = SEBlock1D(channel=8, reduction=2)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        return self.attention(x_seq)  # [B, 8, 1024]


# ============================================================================
# 任务头: 渐进投影 + 宽带/窄带双路径
# ============================================================================
class HighFidelityTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16, 1024]), nn.GELU(), nn.Dropout1d(0.3),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, freq_bins),
        )
        self.broadband_path = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=5, padding=2),
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=1), nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=1),
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        env = self.broadband_path(x_seq)
        peaks = self.tonal_peak_path(x_seq)
        return (env + peaks).squeeze(1)


# ============================================================================
# 主模型 (52 个独立任务头)
# ============================================================================
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
        shared_features = self.shared_decoder_body(self.shared_encoder(x))
        batch_size = x.size(0)
        spectrum_out = torch.zeros(
            batch_size, self.heads["mode_0_type_0"].freq_bins,
            device=x.device, dtype=shared_features.dtype
        )
        combo_idx = mode_idx * self.num_types + type_idx
        for combo in torch.unique(combo_idx):
            m = (combo // self.num_types).long().item()
            t = (combo % self.num_types).long().item()
            mask = (combo_idx == combo)
            spectrum_out[mask] = self.heads[f"mode_{m}_type_{t}"](shared_features[mask])
        return spectrum_out


# ============================================================================
# 物理损失 V2: dB-MSE + 线性峰值MSE + Sobolev 梯度
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    """
    三项损失:
      dB-MSE (5.0):      逐点 dB 精度
      线性峰值MSE (2.0):  dB→声压比, 自然侧重高峰值 (60dB处1dB误差≈115倍于20dB处)
      Sobolev 梯度 (3.0): 频谱平滑约束
    """
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0
        self.weight_linear = 2.0
        self.weight_grad = 3.0

    def _sobolev_gradient_loss(self, pred, target):
        return F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

    def _linear_peak_loss(self, pred_db, target_db):
        ref = torch.max(target_db.detach(), dim=1, keepdim=True)[0]
        pred_lin = torch.pow(10.0, (pred_db - ref) / 20.0)
        target_lin = torch.pow(10.0, (target_db - ref) / 20.0)
        return F.mse_loss(pred_lin, target_lin)

    def forward(self, pred, target):
        loss_mse = F.mse_loss(pred, target)
        loss_linear = self._linear_peak_loss(pred, target)
        loss_grad = self._sobolev_gradient_loss(pred, target)
        total = self.weight_mse * loss_mse + self.weight_linear * loss_linear + self.weight_grad * loss_grad
        return total, loss_mse, loss_grad, loss_linear
