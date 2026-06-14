"""
PIMBCN 网络模型（2026-06-12 V6: 折中正则化 — 平衡拟合与泛化）

=== 相对 V5 的变更 ===
[修正] DropPath率回调: [0,0.05,0.1,0.15,0.2,0.2]→[0,0.02,0.05,0.08,0.1,0.1]
  - V5的深层20%丢弃率导致解码器信息瓶颈过强, 模型欠拟合
  - V6折中: 保留DropPath机制但降低强度
[保留] V5所有架构改进: 嵌入32, 通道128, 频率PE, Dropout 0.2, 损失权重 1:300:1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


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


class SharedDecoderBody(nn.Module):
    def __init__(self, input_dim=320, channels_init=128, seq_len_init=16):
        super().__init__()
        self.channels_init = channels_init
        self.seq_len_init = seq_len_init
        self.fc_expand = nn.Sequential(
            nn.Linear(input_dim, channels_init * seq_len_init),
            nn.GELU(), nn.Dropout(0.3),
        )
        # [V6] 折中: V5=[0,0.05,0.1,0.15,0.2,0.2] → V6=[0,0.02,0.05,0.08,0.1,0.1]
        drop_paths = [0.0, 0.02, 0.05, 0.08, 0.1, 0.1]
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(128, 128, drop_path=drop_paths[0]),  # 16→32
            TransposeConvBlock(128, 64,  drop_path=drop_paths[1]),  # 32→64
            TransposeConvBlock(64,  64,  drop_path=drop_paths[2]),  # 64→128
            TransposeConvBlock(64,  32,  drop_path=drop_paths[3]),  # 128→256
            TransposeConvBlock(32,  16,  drop_path=drop_paths[4]),  # 256→512
            TransposeConvBlock(16,  16,  drop_path=drop_paths[5]),  # 512→1024
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


class PhysicsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_mse = 1.0
        self.weight_linear = 300.0
        self.weight_grad = 1.0
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
