import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionlessFeatureEmbedding(nn.Module):
    """
    共享特征编码器 (Shared Encoder)
    负责将低维工况特征映射为高维物理隐变量，为所有分支提供统一的特征基座
    """
    def __init__(self, input_dim=3, hidden_dim=2048):
        super(DimensionlessFeatureEmbedding, self).__init__()
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
    """
    转置卷积块：转置卷积 + 批量归一化 + GELU + 残差连接
    用于特征的频域空间渐进式上采样
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_residual=True):
        super(TransposeConvBlock, self).__init__()
        self.use_residual = use_residual
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 处理残差连接中通道数不匹配的问题
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
            # 动态调整残差特征序列长度以对齐主分支
            if residual.size(2) != x.size(2):
                residual = F.interpolate(residual, size=x.size(2), mode='linear', align_corners=False)
            x = x + residual
            
        return x

class SharedDecoderBody(nn.Module):
    """
    共享解码器主干 (Shared Decoder Trunk)
    全局共享：将所有样本的隐变量转化为多通道的通用声学时频特征
    这部分占据了网络的大部分参数，由全量数据共同更新，具备极强的物理规律泛化性
    """
    def __init__(self, hidden_dim=2048):
        super(SharedDecoderBody, self).__init__()
        self.seq_len_init = 16 
        self.channels_init = 512 
        
        # 展平层：隐变量扩展
        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 渐进式上采样主干
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(self.channels_init, 512, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            TransposeConvBlock(512, 256, kernel_size=4, stride=2, padding=1),               # 32 -> 64
            TransposeConvBlock(256, 128, kernel_size=4, stride=2, padding=1),               # 64 -> 128
            TransposeConvBlock(128, 64, kernel_size=4, stride=2, padding=1),                # 128 -> 256
            TransposeConvBlock(64, 32, kernel_size=4, stride=2, padding=1),                 # 256 -> 512
            TransposeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),                 # 512 -> 1024
            TransposeConvBlock(16, 8, kernel_size=4, stride=2, padding=1),                  # 1024 -> 2048
        ])

    def forward(self, x):
        # [Batch, 512, 16]
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
            
        return x_seq  # 返回通用特征图 [Batch, 8, 2048]

class LightweightTaskHead(nn.Module):
    """
    轻量级任务头 (Task-Specific Head)
    针对具体的 mode 和 type 独立实例化。
    参数量极小，主要负责在通用特征上进行微调（Fine-tuning）和插值映射
    """
    def __init__(self, freq_bins=2501):
        super(LightweightTaskHead, self).__init__()
        self.freq_bins = freq_bins
        
        self.refine_conv = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.refine_bn = nn.BatchNorm1d(8)
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)
        
        # 最终的频域平滑与形态矫正
        self.output_refine = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_seq):
        # 通道精炼
        x_seq = F.gelu(self.refine_bn(self.refine_conv(x_seq))) # [Batch, 8, 2048]
        x_seq = self.final_conv(x_seq)                          # [Batch, 1, 2048]
        
        # 物理保持的一维线性插值（保持声学相邻频带的归纳偏置）
        x_seq = F.interpolate(x_seq, size=self.freq_bins, mode='linear', align_corners=False) # [Batch, 1, 2501]
        
        # 输出平滑
        spectrum = self.output_refine(x_seq).squeeze(1) # [Batch, 2501]
        return spectrum

class PI_MBCN(nn.Module):
    """
    物理信息引导的参数共享多分支卷积网络 (Hard Parameter Sharing MTL)
    """
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super(PI_MBCN, self).__init__()
        self.num_modes = num_modes
        self.num_types = num_types
        
        # 1. 实例化全局共享组件
        self.shared_encoder = DimensionlessFeatureEmbedding(input_dim=3, hidden_dim=2048)
        self.shared_decoder_body = SharedDecoderBody(hidden_dim=2048)
        
        # 2. 实例化 52 个轻量级独立分支（字典存储）
        self.heads = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                branch_key = f"mode_{m}_type_{t}"
                self.heads[branch_key] = LightweightTaskHead(freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        # 第一阶段：全量样本并行通过共享主干，提取通用声学特征
        shared_hidden = self.shared_encoder(x)
        shared_features = self.shared_decoder_body(shared_hidden) # 输出: [Batch, 8, 2048]
        
        batch_size = x.size(0)
        spectrum_out = torch.zeros(batch_size, self.heads["mode_0_type_0"].freq_bins, device=x.device, dtype=x.dtype)
        
        # 第二阶段：动态掩码路由 (Dynamic Masked Routing)
        # 将 mode 和 type 映射为唯一索引以进行向量化匹配
        combo_idx = mode_idx * self.num_types + type_idx
        unique_combos = torch.unique(combo_idx)
        
        for combo in unique_combos:
            m = (combo // self.num_types).long().item()
            t = (combo % self.num_types).long().item()
            branch_key = f"mode_{m}_type_{t}"
            
            # 提取属于当前特定子任务的样本掩码
            mask = (combo_idx == combo)
            
            # 并行提取当前分支的特征图，送入专属轻量级 Head
            # 反向传播时，特定样本的梯度将流过其专属 Head，并在 shared_decoder_body 处完美汇合
            branch_features = shared_features[mask]
            s_pred = self.heads[branch_key](branch_features)
            
            # 将预测结果填回输出张量
            spectrum_out[mask] = s_pred
            
        return spectrum_out

def physics_informed_loss_fn(pred_spectrum, target_spectrum):
    """
    声学物理信息引导的数值稳定损失函数
    约束项：频谱均方误差 + 形态余弦相似度 + 总声压级(OASPL)能量守恒约束
    """
    loss_mse_spec = F.mse_loss(pred_spectrum, target_spectrum)
    loss_cosine_spec = 1.0 - F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean()
    
    def safe_oaspl(spectrum):
        """
        利用 Log-Sum-Exp 思想的数值安全声压级计算函数
        避免 10^(L/10) 在网络训练初期因奇异值导致的梯度爆炸 (Gradient Explosion)
        """
        scaled_spec = spectrum / 10.0
        # 提取序列最大值进行放缩保护
        max_val, _ = torch.max(scaled_spec, dim=1, keepdim=True)
        eps = 1e-10
        # 剥离最大项后的稳定求和
        sum_exp = torch.sum(torch.pow(10.0, scaled_spec - max_val), dim=1, keepdim=True)
        # 还原物理量级
        oaspl = 10.0 * (torch.log10(sum_exp + eps) + max_val)
        return oaspl

    pred_oaspl = safe_oaspl(pred_spectrum)
    target_oaspl = safe_oaspl(target_spectrum)
    
    loss_mse_oaspl = F.mse_loss(pred_oaspl, target_oaspl)
    
    # 权重超参数：可根据不同量级损失的梯度动态退火（例如 GradNorm）
    total_loss = loss_mse_spec + 200.0 * loss_cosine_spec + loss_mse_oaspl
    
    return total_loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl