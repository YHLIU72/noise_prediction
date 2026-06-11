"""
PIMBCN 网络模型（2026-06-11 V3: 嵌入注入 + 多分辨率损失）

=== 相对 0610_final 的增量变更 ===
[策略1] 梯度累积 → 训练脚本侧实现
[策略2] 多分辨率损失: 原生 + 2×下采样 + 4×下采样 MSE
[策略3] Mode/Type 嵌入: 解码器感知工况, 16维可学习嵌入注入
[辅助] LayerNorm 修正: [16,1024]→1024 (逐通道归一化)
[辅助] 傅里叶 scale: 3.0→1.0 (适配平滑物理函数)
[保留] IN归一化 + 损失V2 (dB-MSE + LinearPeak + Sobolev)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 傅里叶特征编码器 (scale=1.0 温和版)
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
# 解码器主干 (输入维度含 mode/type 嵌入: 256+16+16=288)
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
# 任务头 (LayerNorm 修正: 逐通道归一化)
# ============================================================================
class HighFidelityTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout1d(0.3),  # 修正: [16,1024]→1024
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
        return (self.broadband_path(x_seq) + self.tonal_peak_path(x_seq)).squeeze(1)


# ============================================================================
# 主模型: 新增 mode/type 嵌入注入解码器
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246, embed_dim=16):
        super().__init__()
        self.num_modes = num_modes
        self.num_types = num_types
        self.embed_dim = embed_dim

        # 可学习的工况嵌入
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)

        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        # 解码器输入 = encoder输出(256) + mode嵌入(16) + type嵌入(16) = 288
        self.shared_decoder_body = SharedDecoderBody(input_dim=256 + 2 * embed_dim)

        self.heads = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                self.heads[f"mode_{m}_type_{t}"] = HighFidelityTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)                             # [B, 256]
        m_emb = self.mode_embed(mode_idx)                           # [B, 16]
        t_emb = self.type_embed(type_idx)                           # [B, 16]
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)            # [B, 288]
        shared_features = self.shared_decoder_body(cond)             # [B, 8, 1024]

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
# 物理损失 V3: 多分辨率 MSE + 线性峰值 + Sobolev
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    """
    损失组成:
      多分辨率 dB-MSE (5.0):  原生(1.0) + 2×下采样(0.3) + 4×下采样(0.1)
      线性峰值MSE (2.0):      dB→声压比, 侧重高峰值
      Sobolev 梯度 (3.0):     频谱平滑约束
    """
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0
        self.weight_linear = 2.0
        self.weight_grad = 3.0
        # 多分辨率内部权重
        self.ms_w1 = 1.0   # 原生分辨率
        self.ms_w2 = 0.3   # 2× 下采样
        self.ms_w4 = 0.1   # 4× 下采样

    def _multiscale_mse(self, pred, target):
        """原生 + 2× + 4× 下采样 MSE, 聚焦宽带趋势"""
        loss = F.mse_loss(pred, target)
        # 2× 下采样
        p2 = F.avg_pool1d(pred.unsqueeze(1), 2, 2).squeeze(1)
        t2 = F.avg_pool1d(target.unsqueeze(1), 2, 2).squeeze(1)
        loss = loss + self.ms_w2 * F.mse_loss(p2, t2)
        # 4× 下采样
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
