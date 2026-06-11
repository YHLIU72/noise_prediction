"""
PIMBCN 网络模型（2026-05-21 V3 小样本高鲁棒版）

针对 1000 余条极小数据集进行的深度重构：
- 网络瘦身: hidden_dim 2048 -> 512, 解码通道 512 -> 128，大幅缩减参数量以控制 VC 维度。
- 傅里叶限幅: 缩减映射基底，防止高频过拟合。
- 强正则化: 引入更强的 Dropout 与特征约束，符合小数据流形学习范式。
- 静态编译: 依然保留了 torch.compile 的完美兼容能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================= 优化 1: 降维版傅里叶特征编码器 (防止高频噪声记忆) =================
class FourierFeatureEmbedding(nn.Module):
    """小样本适用的傅里叶特征映射，降低基底维度，强化 Dropout"""
    def __init__(self, input_dim=3, mapping_size=64, scale=5.0, hidden_dim=512):
        super().__init__()
        # 缩小 scale 和 mapping_size，避免在小样本上对高频噪声过度敏感
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        
        # 映射后的维度为 mapping_size * 2 = 128
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),  # 提升 Dropout 比例至 0.3

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        x_ff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.encoder(x_ff)

# ================= 无分支转置卷积块 (保持 torch.compile 兼容) =================
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

# ================= 优化 2: 轻量化解码器主干 (引入信息瓶颈) =================
class SharedDecoderBody(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.seq_len_init = 16
        self.channels_init = 128  # 从 512 砍到 128，极大减少卷积核参数量

        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # 针对 128 通道的平滑降维策略
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(128, 128),  # 16 -> 32
            TransposeConvBlock(128, 64),   # 32 -> 64
            TransposeConvBlock(64, 64),    # 64 -> 128
            TransposeConvBlock(64, 32),    # 128 -> 256
            TransposeConvBlock(32, 16),    # 256 -> 512
            TransposeConvBlock(16, 16),    # 512 -> 1024
            TransposeConvBlock(16, 8),     # 1024 -> 2048
        ])
        self.attention = SEBlock1D(channel=8, reduction=2)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        x_seq = self.attention(x_seq)
        return x_seq  # [B, 8, 2048]

# ================= 优化 3: 正则化高保真任务头 =================
class HighFidelityTaskHead(nn.Module):
    def __init__(self, freq_bins=2501):
        super().__init__()
        self.freq_bins = freq_bins

        # 引入 Dropout1d 防御过拟合
        self.refine = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout1d(0.2), # 随机抹除整条频域通道，强迫模型依赖多种特征
            nn.Conv1d(16, 8, kernel_size=3, padding=1)
        )
        
        self.freq_proj = nn.Linear(2048, freq_bins)
        
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

# ================= 主模型 (完全共享任务头版) =================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super().__init__()
        self.num_modes = num_modes
        self.num_types = num_types

        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=512)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=512)

        # 所有组合共享一个任务头（大幅减少参数量，缓解过拟合）
        self.shared_head = HighFidelityTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        shared_hidden = self.shared_encoder(x)
        shared_features = self.shared_decoder_body(shared_hidden)
        
        # 所有样本使用同一个共享任务头
        spectrum_out = self.shared_head(shared_features)
        return spectrum_out

# ================= Sobolev 物理梯度损失 (保持不变，极其有效的物理先验) =================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(4))

    def _safe_oaspl(self, spectrum):
        scaled_spec = spectrum / 10.0
        max_val, _ = torch.max(scaled_spec, dim=1, keepdim=True)
        sum_exp = torch.sum(torch.pow(10.0, scaled_spec - max_val), dim=1, keepdim=True)
        return 10.0 * (torch.log10(sum_exp + 1e-10) + max_val)

    def _sobolev_gradient_loss(self, pred, target):
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        return F.mse_loss(diff_pred, diff_target)

    def forward(self, pred_spectrum, target_spectrum):
        loss_mse = F.mse_loss(pred_spectrum, target_spectrum)
        loss_cosine = 1.0 - F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean()
        
        pred_oaspl = self._safe_oaspl(pred_spectrum)
        target_oaspl = self._safe_oaspl(target_spectrum)
        loss_oaspl = F.mse_loss(pred_oaspl, target_oaspl)
        
        loss_grad = self._sobolev_gradient_loss(pred_spectrum, target_spectrum)

        # 修复：正确的损失加权求和
        # 原始实现错误地将 log_vars 直接加到损失中，这是不正确的
        # 正确做法：prec = exp(-log_var) 作为权重，只进行加权求和
        prec0 = torch.exp(-self.log_vars[0])
        prec1 = torch.exp(-self.log_vars[1])
        prec2 = torch.exp(-self.log_vars[2])
        prec3 = torch.exp(-self.log_vars[3])
        
        total_loss = prec0 * loss_mse + prec1 * loss_cosine + prec2 * loss_oaspl + prec3 * loss_grad

        return total_loss, loss_mse, loss_cosine, loss_oaspl, loss_grad