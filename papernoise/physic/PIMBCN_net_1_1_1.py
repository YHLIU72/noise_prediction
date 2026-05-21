import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionlessFeatureEmbedding(nn.Module):
    """
    增强版共享特征编码器
    增加隐藏层维度和层数，提升特征提取能力
    """
    def __init__(self, input_dim=3, hidden_dim=2048):
        super(DimensionlessFeatureEmbedding, self).__init__()
        self.encoder = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            # 第二层
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            # 第三层
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.encoder(x)

class TransposeConvBlock(nn.Module):
    """
    转置卷积块：转置卷积 + 批量归一化 + GELU + 可选残差连接
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_residual=True):
        super(TransposeConvBlock, self).__init__()
        self.use_residual = use_residual
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 残差连接的投影层（当通道数变化时）
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
            # 调整序列长度以匹配
            if residual.size(2) != x.size(2):
                residual = F.interpolate(residual, size=x.size(2), mode='linear', align_corners=False)
            x = x + residual
        
        return x

class SpectrumDecoder1DCNN(nn.Module):
    """
    优化版深度频谱解码器
    采用高效渐进式上采样策略，结合残差连接改善梯度流动
    
    维度变化: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> 2501
    
    设计原则：
    1. 通道数从高到低递减，符合信息压缩到扩展的规律
    2. 使用残差连接缓解梯度消失
    3. 适当减少卷积层数，避免过度参数化
    4. 最后使用线性层微调维度
    """
    def __init__(self, hidden_dim=2048, freq_bins=2501):
        super(SpectrumDecoder1DCNN, self).__init__()
        self.freq_bins = freq_bins
        self.seq_len_init = 16  # 初始序列长度
        self.channels_init = 512  # 增加初始通道数，提升特征表达
        
        # 展平层：将隐向量扩展为多通道序列
        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, self.channels_init * self.seq_len_init),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # ================= 优化后的渐进式转置卷积上采样 =================
        # 通道数策略：512 -> 256 -> 128 -> 64 -> 32 -> 16
        # 使用5层转置卷积，比原来的7层更高效
        self.deconv_layers = nn.ModuleList([
            TransposeConvBlock(self.channels_init, 512, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            TransposeConvBlock(512, 256, kernel_size=4, stride=2, padding=1),               # 32 -> 64
            TransposeConvBlock(256, 128, kernel_size=4, stride=2, padding=1),               # 64 -> 128
            TransposeConvBlock(128, 64, kernel_size=4, stride=2, padding=1),                # 128 -> 256
            TransposeConvBlock(64, 32, kernel_size=4, stride=2, padding=1),                 # 256 -> 512
            TransposeConvBlock(32, 16, kernel_size=4, stride=2, padding=1),                 # 512 -> 1024
            TransposeConvBlock(16, 8, kernel_size=4, stride=2, padding=1),                  # 1024 -> 2048
        ])
        
        # 精炼卷积层：在转置卷积后进行特征精炼
        self.refine_conv = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.refine_bn = nn.BatchNorm1d(8)
        
        # 将通道数降为1
        self.final_conv = nn.Conv1d(8, 1, kernel_size=1)
        
        # 非线性维度调整：从2048到2501
        self.dim_adjust = nn.Sequential(
            nn.Linear(2048, 2274),
            nn.GELU(),
            nn.Linear(2274, 2501),
            nn.GELU()
        )
        
        # 输出精炼层：最后调整频谱曲线
        self.output_refine = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # ================= 渐进式上采样 =================
        # 将隐向量扩展为序列
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)  # [Batch, 512, 16]
        
        # 通过转置卷积层进行上采样
        for deconv in self.deconv_layers:
            x_seq = deconv(x_seq)
        
        # 精炼特征
        x_seq = F.gelu(self.refine_bn(self.refine_conv(x_seq)))  # [Batch, 8, 2048]
        
        # 转换为单通道
        x_seq = self.final_conv(x_seq)  # [Batch, 1, 2048]
        x_seq = x_seq.squeeze(1)        # [Batch, 2048]
        
        # 非线性维度调整
        x_seq = self.dim_adjust(x_seq)  # [Batch, 2501]
        
        # 添加通道维度进行最后的卷积精炼
        x_seq = x_seq.unsqueeze(1)      # [Batch, 1, 2501]
        
        # 输出精炼
        spectrum = self.output_refine(x_seq).squeeze(1)  # [Batch, 2501]
        
        return spectrum

class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super(PI_MBCN, self).__init__()
        self.num_modes = num_modes
        self.num_types = num_types
        # 使用增强版编码器，隐藏层维度为2048
        self.shared_encoder = DimensionlessFeatureEmbedding(input_dim=3, hidden_dim=2048)
        self.branches = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                branch_key = f"mode_{m}_type_{t}"
                # 【修复关键点】：统一 hidden_dim 为 2048
                self.branches[branch_key] = SpectrumDecoder1DCNN(hidden_dim=2048, freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)
        batch_size = x.size(0)
        spectrum_out = torch.zeros(batch_size, self.branches["mode_0_type_0"].freq_bins, device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            m = mode_idx[i].long().item()
            t = type_idx[i].long().item()
            branch_key = f"mode_{m}_type_{t}"
            
            h_i = hidden[i:i+1]
            s_pred = self.branches[branch_key](h_i)
            
            spectrum_out[i] = s_pred[0]
            
        return spectrum_out

def physics_informed_loss_fn(pred_spectrum, target_spectrum):
    """
    简化版损失函数
    保留：
    - 频谱MSE损失
    - 频谱余弦相似度损失
    - 合成OASPL与真实频谱合成OASPL之间的MSE损失
    """
    # 频谱MSE损失
    loss_mse_spec = F.mse_loss(pred_spectrum, target_spectrum)
    
    # 频谱余弦相似度损失（形态匹配）
    loss_cosine_spec = 1.0 - F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean()
    
    # 计算预测频谱的OASPL（从频谱合成）
    # OASPL = 10 * log10(sum(10^(spectrum/10)))
    # 使用数值稳定的计算方式，防止log10(0)或极小值
    eps = 1e-10
    pred_oaspl = 10.0 * torch.log10(torch.sum(torch.pow(10.0, pred_spectrum / 10.0), dim=1, keepdim=True) + eps)
    
    # 计算真实频谱的OASPL（从频谱合成）
    target_oaspl = 10.0 * torch.log10(torch.sum(torch.pow(10.0, target_spectrum / 10.0), dim=1, keepdim=True) + eps)
    
    # OASPL MSE损失
    loss_mse_oaspl = F.mse_loss(pred_oaspl, target_oaspl)
    
    # 总损失
    total_loss = loss_mse_spec + 200.0 * loss_cosine_spec + 20.0 * loss_mse_oaspl
    
    return total_loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl