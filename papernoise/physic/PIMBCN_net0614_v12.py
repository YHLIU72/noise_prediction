"""
PIMBCN 网络模型（2026-06-14 V12: 方案2 — 低频专项Head）

=== 基于 V4 架构，新增低频专用分支 ===
  核心思想: 在统一预测全频段的基础上，增加一个专门处理低频段(20-200Hz)的分支。
  低频分支使用大卷积核以捕获低频段的宽泛结构，输出50个低频点的精细化预测。
  损失函数增加低频专项 MSE 项，强制模型关注低频段精度。

=== 相对 V4 的变更 ===
[新增] LowFreqBranch: 独立低频分支
  - 大卷积核 (kernel=15, groups=4) 专门提取低频特征
  - 输出前50个频点 (20~200Hz) 的精细化预测
  - 与主Head的低频部分做残差融合

[新增] 低频损失项: weight_lowfreq=2.0
  - 总损失 = 5*MSE + 2*LinearPeak + 3*Grad + 2*LowFreqMSE

[保留] V4 全部设计: 1共享Head, 5:2:3基础损失, embed=16, ch=64, Dropout 0.5
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
# [V12 新增] 低频专项分支
# ============================================================================
class LowFreqBranch(nn.Module):
    """
    专门处理 20~200Hz 低频段的分支 (对应前 ~50 个频点).
    使用大卷积核 + 分组卷积, 捕获低频段的宽泛频谱结构.
    输入: 解码器特征 [B, 8, 1024]
    输出: 低频精细化预测 [B, 50] (20~200Hz, 残差加到主Head低频部分)
    """
    def __init__(self, low_bins=50):
        super().__init__()
        # 大卷积核提取低频趋势特征
        self.low_extract = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=15, padding=7, groups=4),  # 分组卷积, 大感受野
            nn.InstanceNorm1d(8), nn.GELU(),
            nn.Conv1d(8, 4, kernel_size=11, padding=5, groups=2),
            nn.InstanceNorm1d(4), nn.GELU(),
        )
        # 投影到低频频点: 1024 序列 → low_bins 低频点
        self.low_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),                     # 压缩时序: 1024→128
            nn.Flatten(start_dim=1),                        # [B, 4*128] = [B, 512]
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, low_bins),                       # → [B, 50]
        )

    def forward(self, x_seq):
        # x_seq: [B, 8, 1024]
        feat = self.low_extract(x_seq)   # [B, 4, 1024]
        return self.low_proj(feat)       # [B, 50]


# ============================================================================
# 共享任务头 (与 V4 相同)
# ============================================================================
class SharedTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):
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
# 主模型: 1共享Head + 低频分支 [V12]
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246, embed_dim=16, low_bins=50):
        super().__init__()
        self.low_bins = low_bins
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(input_dim=256 + 2 * embed_dim)
        self.shared_head = SharedTaskHead(freq_bins=freq_bins)
        self.lowfreq_branch = LowFreqBranch(low_bins=low_bins)  # [V12] 低频分支

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)
        m_emb = self.mode_embed(mode_idx)
        t_emb = self.type_embed(type_idx)
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)
        features = self.shared_decoder_body(cond)            # [B, 8, 1024]

        # 主Head预测全频段
        pred_full = self.shared_head(features)               # [B, 1246]

        # [V12] 低频分支预测低频精细化修正
        pred_low_residual = self.lowfreq_branch(features)    # [B, 50]

        # 残差融合: 主Head前50个频点 + 低频分支修正
        pred_full[:, :self.low_bins] = pred_full[:, :self.low_bins] + pred_low_residual

        return pred_full


# ============================================================================
# 物理损失 V4 + 低频专项损失 [V12]
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    """
    [V12] 增加低频专项 MSE 损失项
    总损失 = weight_mse * MSE + weight_linear * LinearPeak
             + weight_grad * Grad + weight_lowfreq * LowFreqMSE
    """
    def __init__(self, low_bins=50):
        super().__init__()
        self.weight_mse = 5.0
        self.weight_linear = 2.0
        self.weight_grad = 3.0
        self.weight_lowfreq = 2.0    # [V12] 低频专项损失权重
        self.low_bins = low_bins
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

    def _lowfreq_mse(self, pred, target):
        """[V12] 仅计算低频段 (前 low_bins 个频点) 的 MSE"""
        return F.mse_loss(pred[:, :self.low_bins], target[:, :self.low_bins])

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
        loss_lowfreq = self._lowfreq_mse(pred, target)        # [V12] 低频专项

        total = (self.weight_mse * loss_mse
                 + self.weight_linear * loss_linear
                 + self.weight_grad * loss_grad
                 + self.weight_lowfreq * loss_lowfreq)        # [V12]
        return total, loss_mse, loss_grad, loss_linear, loss_lowfreq


# ============================================================================
# 参数统计
# ============================================================================
if __name__ == "__main__":
    model = PI_MBCN()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V12 参数总量: {total:,}, 可训练: {trainable:,}")
    lf_params = sum(p.numel() for p in model.lowfreq_branch.parameters())
    print(f"  其中低频分支参数: {lf_params:,}")
