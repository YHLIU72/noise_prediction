"""
PIMBCN 网络模型（2026-06-09 频率范围适配版：20~5000Hz）

针对 1000 余条极小数据集进行的深度重构：
- 网络瘦身: hidden_dim 2048 -> 512, 解码通道 512 -> 128，大幅缩减参数量以控制 VC 维度。
- 傅里叶限幅: 缩减映射基底，防止高频过拟合。
- 强正则化: 引入更强的 Dropout 与特征约束，符合小数据流形学习范式。
- 频率范围: 从原始 0~10000Hz (2501点) 截取 20~5000Hz (1246点, 间隔4Hz)。

变更说明 (2026-06-09):
- freq_bins 默认值由 2501 改为 1246，对应 20~5000Hz @4Hz 间隔。
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
        # 进一步缩小 scale 和 mapping_size，降低高频拟合能力
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        
        # 映射后的维度为 mapping_size * 2 = 64
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # LayerNorm 在小批量更稳定
            nn.GELU(),
            nn.Dropout(0.5),  # 大幅提升 Dropout 至 0.5

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
        # 防止 Python float64(math.pi) 提升 dtype，显式转为模型精度
        x = x.to(self.B.dtype)
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

# ================= 优化 2: 轻量化解码器主干 (匹配1246点输出, 缩减中间膨胀) =================
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

        # 去掉最后一层转置卷积 (8→4, 1024→2048)，使中间维度从2048降至1024
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

# ================= 优化 3: 强正则化任务头 (适配 [B,8,1024] 输入) =================
class HighFidelityTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):  # 20~5000Hz, 1246点 @4Hz
        super().__init__()
        self.freq_bins = freq_bins

        # 残差精炼: 输入输出均为 [B, 8, 1024]
        self.refine = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16, 1024]),
            nn.GELU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(16, 8, kernel_size=3, padding=1)
        )
        
        # 渐进投影: 1024 → 512 → freq_bins (减少直接投影参数量)
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, freq_bins),
        )
        
        # 宽带包络 + 窄带峰值 双路径
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
        x_seq = self.freq_proj(x_seq)  # [B,8,1024]→[B,8,512]→[B,8,freq_bins]
        env = self.broadband_path(x_seq)
        peaks = self.tonal_peak_path(x_seq)
        spectrum = (env + peaks).squeeze(1)
        return spectrum

# ================= 主模型 (保留多任务头版) =================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246):
        super().__init__()
        self.num_modes = num_modes
        self.num_types = num_types

        # 减少隐藏层维度以控制模型容量
        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=256)

        # 保留 52 个独立任务头
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

# ================= Sobolev 物理梯度损失 (重平衡权重版) =================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 重平衡损失权重: 增强频谱平滑约束, 降低逐点MSE主导
        # Sobolev(4.0) > MSE(5.0) > Cosine(1.5) = LogMSE(1.5) > OASPL(1.0)
        self.weight_mse = 5.0       # 谱 MSE (dB域)
        self.weight_cosine = 1.5    # Cosine 形状约束
        self.weight_oaspl = 1.0     # OASPL 总声压级
        self.weight_grad = 4.0      # Sobolev 梯度平滑 (↑增强)
        self.weight_logmse = 1.5    # 对数域 MSE (关注低幅值频段)

    def _safe_oaspl(self, spectrum):
        scaled_spec = spectrum / 10.0
        max_val, _ = torch.max(scaled_spec, dim=1, keepdim=True)
        sum_exp = torch.sum(torch.pow(10.0, scaled_spec - max_val), dim=1, keepdim=True)
        return 10.0 * (torch.log10(sum_exp + 1e-7) + max_val)  # 1e-7 安全于 AMP float16

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

        # 对数域 MSE: 关注低幅值频段, 更贴合声学感知
        eps = 1e-7
        loss_logmse = F.mse_loss(
            torch.log(torch.clamp(pred_spectrum, min=eps)),
            torch.log(torch.clamp(target_spectrum, min=eps))
        )

        # 使用重平衡权重计算总损失
        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_cosine * loss_cosine +
            self.weight_oaspl * loss_oaspl +
            self.weight_grad * loss_grad +
            self.weight_logmse * loss_logmse
        )

        return total_loss, loss_mse, loss_cosine, loss_oaspl, loss_grad, loss_logmse
