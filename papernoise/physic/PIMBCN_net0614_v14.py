"""
PIMBCN 网络模型（2026-06-14 V14: 方案3 — 修复V10增强 + 损失重校准）

=== 基于 V10 架构（60~5000Hz, 1236 bins），修复V10失败因素 ===
[V14] 损失权重重新校准:
  - MSE: 5.0→6.0 (补偿dB尺度缩小, 60~5000Hz的dB值整体更低)
  - LinearPeak: 2.0→3.0 (60~5000Hz峰值位置变化, 需更强线性域约束)
  - Sobolev: 3.0→1.5 (V10的val_grad=0.414已优于V4=0.508, 释放优化空间)

[数据侧修复] 频移增强: np.roll→np.pad(reflect) (见 PIMBCN_data_0614_v14.py)

[保留] V4/V10架构: 1共享Head, 嵌入注入, 多分辨率损失, IN, EMA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 傅里叶特征编码器 (scale=1.0)
# ============================================================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=32, scale=1.0, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = (2.0 * math.pi * x) @ self.B
        return self.encoder(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1))


# ============================================================================
# 转置卷积块 (InstanceNorm)
# ============================================================================
class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.residual_proj = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = self.residual_proj(x)
        return F.gelu(self.norm(self.conv(x))) + residual


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
        return x * self.fc(y).view(b, c, 1).expand_as(x)


# ============================================================================
# 解码器主干
# ============================================================================
class SharedDecoderBody(nn.Module):
    def __init__(self, input_dim=288, channels_init=64, seq_len_init=16):
        super().__init__()
        self.channels_init = channels_init
        self.seq_len_init = seq_len_init
        self.fc_expand = nn.Sequential(
            nn.Linear(input_dim, channels_init * seq_len_init),
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
# 共享任务头 (freq_bins=1236 适配 60~5000Hz)
# ============================================================================
class SharedTaskHead(nn.Module):
    def __init__(self, freq_bins=1236):
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(8, 24, kernel_size=3, padding=1),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout1d(0.3),
            nn.Conv1d(24, 8, kernel_size=3, padding=1),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, freq_bins),
        )
        self.broadband_path = nn.Sequential(
            nn.Conv1d(8, 6, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(6, 1, kernel_size=5, padding=2),
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(8, 6, kernel_size=1), nn.GELU(),
            nn.Conv1d(6, 1, kernel_size=1),
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        return (self.broadband_path(x_seq) + self.tonal_peak_path(x_seq)).squeeze(1)


# ============================================================================
# 主模型: 1个共享Head (V4架构, 适配 60~5000Hz)
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1236, embed_dim=16):
        super().__init__()
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(input_dim=256 + 2 * embed_dim)
        self.shared_head = SharedTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)
        m_emb = self.mode_embed(mode_idx)
        t_emb = self.type_embed(type_idx)
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)
        features = self.shared_decoder_body(cond)
        return self.shared_head(features)


# ============================================================================
# 物理损失 V4 + [V14] 损失权重重校准
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    """
    [V14] 损失权重重校准 — 针对 60~5000Hz 截断频谱优化:
      - MSE ↑ 5.0→6.0: 60~5000Hz的dB值整体比20~5000Hz低约3-5dB,
        MSE绝对值变小, 需提高权重以维持梯度尺度
      - LinearPeak ↑ 2.0→3.0: 截断后峰值位置变化, 需更强线性域约束
      - Sobolev ↓ 3.0→1.5: V10的val_grad=0.414已大幅优于V4=0.508,
        降低梯度约束释放优化空间给MSE
    """
    def __init__(self):
        super().__init__()
        self.weight_mse = 6.0      # [V14] 5.0→6.0
        self.weight_linear = 3.0   # [V14] 2.0→3.0
        self.weight_grad = 1.5     # [V14] 3.0→1.5
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


# ============================================================================
# 参数统计
# ============================================================================
if __name__ == "__main__":
    model = PI_MBCN()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V14 参数总量: {total:,}, 可训练: {trainable:,}")
    print(f"频率范围: 60~5000Hz, 频点数: 1236")
    print(f"损失权重: MSE={6.0}, LinearPeak={3.0}, Sobolev={1.5}")
