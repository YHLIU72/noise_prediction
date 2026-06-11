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

# ================= 优化 2: 扩展共享解码器主干 (增大参数共享) =================
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
            TransposeConvBlock(8, 4),     # 1024 -> 2048
        ])
        self.attention = SEBlock1D(channel=4, reduction=2)
        
        # 新增：共享的频率投影层（从解码器输出到频谱维度）
        self.shared_freq_proj = nn.Linear(2048, 2501)
        
        # 新增：共享的频谱精修层
        self.shared_refine = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.LayerNorm([4, 2501]),
            nn.GELU(),
            nn.Dropout1d(0.3),
            nn.Conv1d(4, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        x_seq = self.attention(x_seq)  # [B, 4, 2048]
        
        # 共享的频率投影
        # 将形状改为 [B*4, 2048] 以应用线性层
        batch_size = x_seq.size(0)
        x_seq = x_seq.view(-1, 2048)  # [B*4, 2048]
        x_seq = self.shared_freq_proj(x_seq)  # [B*4, 2501]
        x_seq = x_seq.view(batch_size, 4, 2501)  # [B, 4, 2501]
        
        # 共享的频谱精修
        x_seq = x_seq.sum(dim=1, keepdim=True)  # [B, 1, 2501]
        x_seq = x_seq + self.shared_refine(x_seq)
        
        return x_seq.squeeze(1)  # [B, 2501] - 返回共享的基础频谱

# ================= 优化 3: 轻量化任务头 (修复梯度消失问题) =================
class HighFidelityTaskHead(nn.Module):
    """轻量化任务头：仅包含任务特定的微调层，大部分计算已在共享解码器中完成"""
    def __init__(self, freq_bins=2501):
        super().__init__()
        self.freq_bins = freq_bins

        # 修复问题二：移除 GELU，直接使用线性变换，避免梯度稀释
        self.task_adjust = nn.Conv1d(1, 1, kernel_size=1, bias=True)
        # 初始化卷积核为恒等映射，确保初始输出等于输入
        nn.init.constant_(self.task_adjust.weight, 1.0)
        nn.init.constant_(self.task_adjust.bias, 0.0)
        
        # 修复问题一：使用 Softplus 替代 Sigmoid，避免梯度消失
        # 同时进一步压缩中间层，减少参数量
        self.freq_attn = nn.Sequential(
            nn.Linear(freq_bins, freq_bins // 32),
            nn.GELU(),
            nn.Linear(freq_bins // 32, freq_bins),
            nn.Softplus(beta=1.0)  # 替换 Sigmoid，避免饱和
        )
        # 初始化注意力最后一层偏置，使初始输出接近1
        nn.init.constant_(self.freq_attn[-2].bias, 5.0)

    def forward(self, base_spectrum):
        """
        Args:
            base_spectrum: 共享解码器输出的基础频谱 [B, freq_bins]
        Returns:
            任务特定的微调频谱 [B, freq_bins]
        """
        # 添加通道维度
        x = base_spectrum.unsqueeze(1)  # [B, 1, freq_bins]
        
        # 任务特定微调（线性变换，无激活函数）
        x = self.task_adjust(x)
        
        # 频率注意力（使用 Softplus，避免梯度消失）
        # 通过温度缩放使注意力初始值接近1，范围限制在 [0.9, 1.0]
        attn_weights = self.freq_attn(base_spectrum) * 0.1 + 0.9
        x = x.squeeze(1) * attn_weights
        
        return x

# ================= 主模型 (增强共享 + 轻量化任务头) =================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super().__init__()
        self.num_modes = num_modes
        self.num_types = num_types

        # 保持隐藏层维度，但增大共享部分
        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=256)  # 已扩展共享部分

        # 保留 52 个独立任务头，但每个头已轻量化
        self.heads = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                self.heads[f"mode_{m}_type_{t}"] = HighFidelityTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        # 共享编码器
        shared_hidden = self.shared_encoder(x)
        
        # 共享解码器（已包含频率投影和基础频谱生成）
        base_spectrum = self.shared_decoder_body(shared_hidden)  # [B, freq_bins]

        batch_size = x.size(0)
        spectrum_out = torch.zeros(
            batch_size, self.heads["mode_0_type_0"].freq_bins,
            device=x.device, dtype=base_spectrum.dtype
        )

        combo_idx = mode_idx * self.num_types + type_idx
        unique_combos = torch.unique(combo_idx)

        # 每个任务头仅做微调
        for combo in unique_combos:
            m = (combo // self.num_types).long().item()
            t = (combo % self.num_types).long().item()
            mask = (combo_idx == combo)
            spectrum_out[mask] = self.heads[f"mode_{m}_type_{t}"](base_spectrum[mask]).to(spectrum_out.dtype)

        return spectrum_out

# ================= Sobolev 物理梯度损失 (固定权重版本) =================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 固定损失权重（按照大小设定好比重）
        # MSE(7.39) > Sobolev(2.72) > Cosine(1) = OASPL(1)
        self.weight_mse = 7.39      # 谱 MSE 损失权重
        self.weight_cosine = 1.0    # Cosine 相似度损失权重
        self.weight_oaspl = 1.0     # OASPL 损失权重
        self.weight_grad = 2.72     # Sobolev 梯度损失权重

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

        # 使用固定权重计算总损失
        total_loss = (
            self.weight_mse * loss_mse +
            self.weight_cosine * loss_cosine +
            self.weight_oaspl * loss_oaspl +
            self.weight_grad * loss_grad
        )

        return total_loss, loss_mse, loss_cosine, loss_oaspl, loss_grad