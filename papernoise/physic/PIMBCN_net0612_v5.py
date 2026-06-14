"""
PIMBCN 网络模型（2026-06-12 V5: 架构增强+损失重校准）

=== 相对 V4 的变更 ===
[P1] Dropout全面降低: 0.5→0.2, 缓解train/val不一致
[P2] 嵌入维度16→32: 强化工况区分能力
[P2] 解码器通道64→128: 提升模型容量, 改善高频段
[P2] refine通道24→32: 进一步增强
[P3] 新增频率位置编码: 帮助模型感知频率轴关系
[P3] 新增 StochasticDepth(DropPath): 解码器中随机丢弃层, 强力正则化
[P0] 损失权重重校准: MSE=1.0, LinearPeak=300, Grad=1.0
  - 三者实际贡献接近 (0.56 : 0.24 : 0.52)
  - 解决V4中LinearPeak几乎不起作用的问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 随机深度 (Stochastic Depth / DropPath)
# ============================================================================
class DropPath(nn.Module):
    """按概率随机丢弃整个残差分支 (训练时), 推理时恒等"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x / keep_prob * random_tensor


# ============================================================================
# 正弦位置编码 (用于频率轴感知)
# ============================================================================
class FrequencyPositionalEncoding(nn.Module):
    """为频谱序列添加正弦位置编码, 帮助模型感知频率轴"""
    def __init__(self, seq_len, d_model, max_len=5000):
        super().__init__()
        # 使用对数频率刻度, 更贴近人耳感知
        freqs = torch.linspace(20, max_len, seq_len).unsqueeze(1)  # [L, 1]
        # 取对数后归一化到[0, 1]
        freqs_norm = (torch.log10(freqs) - 1.3) / 2.4  # log10(20)=1.3, log10(5000)=3.7
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(freqs_norm * div_term)
        pe[:, 1::2] = torch.cos(freqs_norm * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d_model]

    def forward(self, x):
        # x: [B, C, L], pe: [1, L, C] → 广播相加
        return x + self.pe[:, :x.shape[2], :].transpose(1, 2)


# ============================================================================
# 傅里叶特征编码器
# ============================================================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=32, scale=1.0, hidden_dim=256):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),   # [V5] 0.5→0.2
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),   # [V5] 0.5→0.2
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15),  # [V5] 0.3→0.15
        )

    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = (2.0 * math.pi * x) @ self.B
        return self.encoder(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1))


# ============================================================================
# 转置卷积块 (InstanceNorm + DropPath)
# ============================================================================
class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 drop_path=0.0):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.residual_proj = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        return self.drop_path(F.gelu(self.norm(self.conv(x)))) + residual


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
# 解码器主干 [V5] 通道加倍 + DropPath + 频率位置编码
# ============================================================================
class SharedDecoderBody(nn.Module):
    def __init__(self, input_dim=320, channels_init=128, seq_len_init=16):
        # [V5] input_dim: 256 + 2*32 = 320 (嵌入维度增大)
        # [V5] channels_init: 64→128
        super().__init__()
        self.channels_init = channels_init
        self.seq_len_init = seq_len_init
        self.fc_expand = nn.Sequential(
            nn.Linear(input_dim, channels_init * seq_len_init),
            nn.GELU(), nn.Dropout(0.3),   # [V5] 0.5→0.3
        )
        # [V5] 逐层递增的 DropPath 率 (深层更强正则)
        drop_paths = [0.0, 0.05, 0.1, 0.15, 0.2, 0.2]
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(128, 128, drop_path=drop_paths[0]),  # 16→32
            TransposeConvBlock(128, 64,  drop_path=drop_paths[1]),  # 32→64
            TransposeConvBlock(64,  64,  drop_path=drop_paths[2]),  # 64→128
            TransposeConvBlock(64,  32,  drop_path=drop_paths[3]),  # 128→256
            TransposeConvBlock(32,  16,  drop_path=drop_paths[4]),  # 256→512
            TransposeConvBlock(16,  16,  drop_path=drop_paths[5]),  # 512→1024
        ])
        # [V5] 注意力通道数随解码器输出变化: 最终16通道, reduction=2→8
        self.attention = SEBlock1D(channel=16, reduction=2)
        # [V5新增] 频率位置编码
        self.freq_pos = FrequencyPositionalEncoding(seq_len=1024, d_model=16)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        x_seq = self.attention(x_seq)         # [B, 16, 1024]
        x_seq = self.freq_pos(x_seq)          # [V5] 注入频率位置信息
        return x_seq


# ============================================================================
# 共享任务头 [V5] 适配16通道 + refine增强
# ============================================================================
class SharedTaskHead(nn.Module):
    def __init__(self, freq_bins=1246, in_channels=16):
        # [V5] in_channels: 8→16 (随解码器通道加倍)
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=3, padding=1),    # [V5] 16→48 (原8→24)
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout1d(0.2),       # [V5] 0.3→0.2
            nn.Conv1d(48, in_channels, kernel_size=3, padding=1),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),  # [V5] 0.3→0.2
            nn.Linear(512, freq_bins),
        )
        self.broadband_path = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=7, padding=3), nn.GELU(),  # [V5] 6→8
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=1), nn.GELU(),              # [V5] 6→8
            nn.Conv1d(8, 1, kernel_size=1),
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        return (self.broadband_path(x_seq) + self.tonal_peak_path(x_seq)).squeeze(1)


# ============================================================================
# 主模型: 1个共享Head [V5] 嵌入维度32
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246, embed_dim=32):
        # [V5] embed_dim: 16→32
        super().__init__()
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)

        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        decoder_input_dim = 256 + 2 * embed_dim                         # [V5] 256+64=320
        self.shared_decoder_body = SharedDecoderBody(input_dim=decoder_input_dim, channels_init=128)
        self.shared_head = SharedTaskHead(freq_bins=freq_bins, in_channels=16)  # [V5] 16通道

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)                              # [B, 256]
        m_emb = self.mode_embed(mode_idx)                            # [B, 32]
        t_emb = self.type_embed(type_idx)                            # [B, 32]
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)             # [B, 320]
        features = self.shared_decoder_body(cond)                    # [B, 16, 1024]
        return self.shared_head(features)                            # [B, 1246]


# ============================================================================
# 物理损失 V5: 权重重校准 + 多分辨率MSE + 线性峰值 + Sobolev
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # [V5] 权重重校准: 使三者实际贡献接近
        #   V4评估中: MSE约0.56, LinearPeak约0.0008, Grad约0.52
        #   V4贡献: 5×0.56=2.8, 2×0.0008=0.0016, 3×0.52=1.56 → LinearPeak几乎为0
        #   V5目标: 1×0.56=0.56, 300×0.0008=0.24, 1×0.52=0.52 → 三者均衡
        self.weight_mse = 1.0
        self.weight_linear = 300.0
        self.weight_grad = 1.0
        self.ms_w2 = 0.3   # 2× 下采样权重
        self.ms_w4 = 0.1   # 4× 下采样权重

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
# 参数统计 (供参考)
# ============================================================================
if __name__ == "__main__":
    model = PI_MBCN()
    total = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.shared_head.parameters())
    embed_params = sum(p.numel() for p in model.mode_embed.parameters()) + \
                   sum(p.numel() for p in model.type_embed.parameters())
    print(f"模型总参数: {total/1e6:.2f}M")
    print(f"  Encoder+Decoder: {(total-head_params-embed_params)/1e6:.2f}M")
    print(f"  SharedHead: {head_params/1e6:.2f}M")
    print(f"  嵌入: {embed_params}")
    print(f"\nHead数据效率: 865条 / {head_params/1e4:.0f}万参数 = {865/(head_params/1e4):.1f}条/万参数")
    print(f"(V4: 117万参数, V5: 约{head_params/1e4:.0f}万参数)")
