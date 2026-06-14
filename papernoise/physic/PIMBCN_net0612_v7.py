"""
PIMBCN 网络模型（2026-06-12 V7: 回归基线增强版）

=== 核心设计 ===
[V7] V4弱增强 + V5架构增强 + 理性损失权重 = 最佳组合
  - 损失: 5:10:3 (MSE主导 + 适度LinearPeak + Sobolev约束)
  - 无DropPath: V6分析证明DropPath在小批量下有害
  - 保留V5架构: embed_dim=32, channels=128, FreqPE, refine=48
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 频率位置编码 (保留V5改进)
# ============================================================================
class FrequencyPositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, max_len=5000):
        super().__init__()
        freqs = torch.linspace(20, max_len, seq_len).unsqueeze(1)
        freqs_norm = (torch.log10(freqs) - 1.3) / 2.4
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(freqs_norm * div_term)
        pe[:, 1::2] = torch.cos(freqs_norm * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[2], :].transpose(1, 2)


# ============================================================================
# 傅里叶特征编码器 (保留V5)
# ============================================================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=32, scale=1.0, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15),
        )

    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = (2.0 * math.pi * x) @ self.B
        return self.encoder(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1))


# ============================================================================
# 转置卷积块 — [V7] 无DropPath
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
# 通道注意力 (保留V5)
# ============================================================================
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=2):
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
# 共享解码器主干 — [V7] 无DropPath, 保留V5其余架构
# ============================================================================
class SharedDecoderBody(nn.Module):
    def __init__(self, input_dim=320, channels_init=128, seq_len_init=16):
        super().__init__()
        self.channels_init = channels_init
        self.seq_len_init = seq_len_init
        self.fc_expand = nn.Sequential(
            nn.Linear(input_dim, channels_init * seq_len_init),
            nn.GELU(), nn.Dropout(0.3),
        )
        # [V7] 无DropPath — 6层纯净转置卷积
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(128, 128),  # 16→32
            TransposeConvBlock(128, 64),   # 32→64
            TransposeConvBlock(64,  64),   # 64→128
            TransposeConvBlock(64,  32),   # 128→256
            TransposeConvBlock(32,  16),   # 256→512
            TransposeConvBlock(16,  16),   # 512→1024
        ])
        self.attention = SEBlock1D(channel=16, reduction=2)
        self.freq_pos = FrequencyPositionalEncoding(seq_len=1024, d_model=16)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        x_seq = self.attention(x_seq)
        x_seq = self.freq_pos(x_seq)
        return x_seq


# ============================================================================
# 共享任务头 (保留V5)
# ============================================================================
class SharedTaskHead(nn.Module):
    def __init__(self, freq_bins=1246, in_channels=16):
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=3, padding=1),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout1d(0.2),
            nn.Conv1d(48, in_channels, kernel_size=3, padding=1),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, freq_bins),
        )
        self.broadband_path = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=1), nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        return (self.broadband_path(x_seq) + self.tonal_peak_path(x_seq)).squeeze(1)


# ============================================================================
# 主模型
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246, embed_dim=32):
        super().__init__()
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        decoder_input_dim = 256 + 2 * embed_dim
        self.shared_decoder_body = SharedDecoderBody(input_dim=decoder_input_dim, channels_init=128)
        self.shared_head = SharedTaskHead(freq_bins=freq_bins, in_channels=16)

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)
        m_emb = self.mode_embed(mode_idx)
        t_emb = self.type_embed(type_idx)
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)
        features = self.shared_decoder_body(cond)
        return self.shared_head(features)


# ============================================================================
# 物理损失 — [V7] 5:10:3 权重 (MSE主导 + 适度线性峰值)
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0         # [V7] 恢复V4主导
        self.weight_linear = 10.0     # [V7] 折中: V4=2(太弱) vs V5=300(过强) → 10
        self.weight_grad = 3.0        # [V7] 恢复V4
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
