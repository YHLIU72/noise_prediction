"""
PIMBCN 网络模型（2026-06-12 V4: 单共享Head架构）

=== 相对 V0611_v3 的变更 ===
[方案B] 52独立Head → 1共享Head
  - 去 ModuleDict + for循环路由, 865条数据共同训练一个head
  - Head参数总量: 6100万→117万 (↓98%)
  - 每head训练数据: 17条→865条 (↑50×)
  - 前向从O(52)循环 → O(1)单次批量

[微调] 卷积通道小幅增强 (总增<1000参数)
  - refine: 8→16→8 → 8→24→8
  - broadband: 8→4→1 → 8→6→1
  - tonal_peak: 8→4→1 → 8→6→1
  - freq_proj: 不变 (1024→512→1246)

[保留] V0611_v3所有优化: 嵌入注入+多分辨率损失+IN+EMA+warmup+梯度累积
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
# 共享任务头 (唯一, 卷积通道微调)
# ============================================================================
class SharedTaskHead(nn.Module):
    def __init__(self, freq_bins=1246):
        super().__init__()
        self.freq_bins = freq_bins
        self.refine = nn.Sequential(
            nn.Conv1d(8, 24, kernel_size=3, padding=1),   # 8→24 (原16)
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout1d(0.3),
            nn.Conv1d(24, 8, kernel_size=3, padding=1),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, freq_bins),
        )
        self.broadband_path = nn.Sequential(
            nn.Conv1d(8, 6, kernel_size=7, padding=3), nn.GELU(),  # 4→6
            nn.Conv1d(6, 1, kernel_size=5, padding=2),
        )
        self.tonal_peak_path = nn.Sequential(
            nn.Conv1d(8, 6, kernel_size=1), nn.GELU(),              # 4→6
            nn.Conv1d(6, 1, kernel_size=1),
        )

    def forward(self, x_seq):
        x_seq = x_seq + self.refine(x_seq)
        x_seq = self.freq_proj(x_seq)
        return (self.broadband_path(x_seq) + self.tonal_peak_path(x_seq)).squeeze(1)


# ============================================================================
# 主模型: 1个共享Head (52→1, 865条数据联合训练)
# ============================================================================
class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=1246, embed_dim=16):
        super().__init__()
        # 工况嵌入 (告知解码器为哪个mode/type生成)
        self.mode_embed = nn.Embedding(num_modes, embed_dim)
        self.type_embed = nn.Embedding(num_types, embed_dim)

        self.shared_encoder = FourierFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.shared_decoder_body = SharedDecoderBody(input_dim=256 + 2 * embed_dim)
        self.shared_head = SharedTaskHead(freq_bins=freq_bins)  # 唯一head

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)                              # [B, 256]
        m_emb = self.mode_embed(mode_idx)                            # [B, 16]
        t_emb = self.type_embed(type_idx)                            # [B, 16]
        cond = torch.cat([hidden, m_emb, t_emb], dim=-1)             # [B, 288]
        features = self.shared_decoder_body(cond)                    # [B, 8, 1024]
        return self.shared_head(features)                            # [B, 1246]


# ============================================================================
# 物理损失 V3: 多分辨率MSE + 线性峰值 + Sobolev
# ============================================================================
class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_mse = 5.0
        self.weight_linear = 2.0
        self.weight_grad = 3.0
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
    print(f"(V0611_v3: 17条/head × 52 = 884条/6100万参数 = 0.14条/万参数)")
    print(f"Head数据效率提升: {865/(head_params/1e4) / 0.14:.1f}×")
