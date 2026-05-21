import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionlessFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=2048):
        super(DimensionlessFeatureEmbedding, self).__init__()
        # 保持原有结构，特征提取能力已足够
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.encoder(x)

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_residual=True):
        super(TransposeConvBlock, self).__init__()
        self.use_residual = use_residual
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if self.use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None
            
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            if residual.size(2) != x.size(2):
                residual = F.interpolate(residual, size=x.size(2), mode='linear', align_corners=False)
            x = x + residual
        return x

# ================= 新增：通道注意力机制（SE Block 1D） =================
class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation 模块：显式地建模频域特征通道间的相互依赖关系。
    这对于捕捉不同衰减特性的气动噪声频带极其有效。
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SharedDecoderBody(nn.Module):
    def __init__(self, hidden_dim=2048):
        super(SharedDecoderBody, self).__init__()
        self.seq_len_init = 16 
        self.channels_init = 512 
        
        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(self.channels_init, 512, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            TransposeConvBlock(512, 256, kernel_size=4, stride=2, padding=1),               # 32 -> 64
            TransposeConvBlock(256, 128, kernel_size=4, stride=2, padding=1),               # 64 -> 128
            TransposeConvBlock(128, 64, kernel_size=4, stride=2, padding=1),                # 128 -> 256
            TransposeConvBlock(64, 32, kernel_size=4, stride=2, padding=1),                 # 256 -> 512
            TransposeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),                 # 512 -> 1024
            TransposeConvBlock(16, 8, kernel_size=4, stride=2, padding=1),                  # 1024 -> 2048
        ])
        
        # ================= 新增：在主干末端加入通道注意力 =================
        self.attention = SEBlock1D(channel=8, reduction=2)

    def forward(self, x):
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
            
        x_seq = self.attention(x_seq) # [Batch, 8, 2048]
        return x_seq 

# ================= 修改：增强型残差任务头 =================
class LightweightTaskHead(nn.Module):
    """
    升级版残差任务头：
    引入残差瓶颈结构，增强对特定模式下复杂频谱峰值（Peak Frequencies）的拟合能力，
    同时加入平滑卷积减轻插值伪影。
    """
    def __init__(self, freq_bins=2501):
        super(LightweightTaskHead, self).__init__()
        self.freq_bins = freq_bins
        
        # 残差微调块 (Residual Finetuning Block)
        self.refine_conv1 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.refine_bn1 = nn.BatchNorm1d(16)
        self.refine_conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.refine_bn2 = nn.BatchNorm1d(8)
        
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)
        
        # 平滑卷积 (Anti-Aliasing Convolution)，紧跟在插值之后
        self.smoothing_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_seq):
        # 1. 局部残差细化特征
        identity = x_seq
        out = F.gelu(self.refine_bn1(self.refine_conv1(x_seq)))
        out = self.refine_bn2(self.refine_conv2(out))
        x_seq = F.gelu(out + identity) # 引入残差相加 [Batch, 8, 2048]
        
        # 2. 降维
        x_seq = self.final_conv(x_seq) # [Batch, 1, 2048]
        
        # 3. 插值扩展到目标维度
        x_seq = F.interpolate(x_seq, size=self.freq_bins, mode='linear', align_corners=False) # [Batch, 1, 2501]
        
        # 4. 平滑输出，消除拉伸伪影
        spectrum = self.smoothing_conv(x_seq).squeeze(1) # [Batch, 2501]
        return spectrum

class PI_MBCN(nn.Module):
    # 此类保持不变，核心逻辑依旧是动态掩码路由
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super(PI_MBCN, self).__init__()
        self.num_modes = num_modes
        self.num_types = num_types
        
        self.shared_encoder = DimensionlessFeatureEmbedding(input_dim=3, hidden_dim=2048)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=2048)
        
        self.heads = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                branch_key = f"mode_{m}_type_{t}"
                self.heads[branch_key] = LightweightTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        shared_hidden = self.shared_encoder(x)
        shared_features = self.shared_decoder_body(shared_hidden) 
        
        batch_size = x.size(0)
        spectrum_out = torch.zeros(batch_size, self.heads["mode_0_type_0"].freq_bins, device=x.device, dtype=x.dtype)
        
        combo_idx = mode_idx * self.num_types + type_idx
        unique_combos = torch.unique(combo_idx)
        
        for combo in unique_combos:
            m = (combo // self.num_types).long().item()
            t = (combo % self.num_types).long().item()
            branch_key = f"mode_{m}_type_{t}"
            
            mask = (combo_idx == combo)
            branch_features = shared_features[mask]
            s_pred = self.heads[branch_key](branch_features)
            spectrum_out[mask] = s_pred
            
        return spectrum_out

# ================= 新增：动态物理损失封装器 =================
class PhysicsLossWrapper(nn.Module):
    """
    可学习的多任务损失加权模块 (Based on Homoscedastic Uncertainty)
    它会自动学习 MSE、Cosine 和 OASPL 这三个物理约束之间的最优权重比例，
    避免人工调参导致的收敛次优解。
    """
    def __init__(self):
        super(PhysicsLossWrapper, self).__init__()
        # 初始化三个损失的对数方差为 0 (对应权重为 1)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def safe_oaspl(self, spectrum):
        scaled_spec = spectrum / 10.0
        max_val, _ = torch.max(scaled_spec, dim=1, keepdim=True)
        eps = 1e-10
        sum_exp = torch.sum(torch.pow(10.0, scaled_spec - max_val), dim=1, keepdim=True)
        oaspl = 10.0 * (torch.log10(sum_exp + eps) + max_val)
        return oaspl

    def forward(self, pred_spectrum, target_spectrum):
        # 计算基础损失
        loss_mse = F.mse_loss(pred_spectrum, target_spectrum)
        loss_cosine = 1.0 - F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean()
        
        pred_oaspl = self.safe_oaspl(pred_spectrum)
        target_oaspl = self.safe_oaspl(target_spectrum)
        loss_oaspl = F.mse_loss(pred_oaspl, target_oaspl)
        
        # 动态加权机制: L_i * exp(-log_var_i) + log_var_i
        precision0 = torch.exp(-self.log_vars[0])
        total_loss = precision0 * loss_mse + self.log_vars[0]
        
        precision1 = torch.exp(-self.log_vars[1])
        total_loss += precision1 * loss_cosine + self.log_vars[1]
        
        precision2 = torch.exp(-self.log_vars[2])
        total_loss += precision2 * loss_oaspl + self.log_vars[2]
        
        return total_loss, loss_mse, loss_cosine, loss_oaspl