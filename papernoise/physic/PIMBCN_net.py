import torch
import torch.nn as nn
import torch.nn.functional as F

class DimensionlessFeatureEmbedding(nn.Module):
    """
    增强版共享特征编码器
    增加隐藏层维度和层数，提升特征提取能力
    """
    def __init__(self, input_dim=3, hidden_dim=256):
        super(DimensionlessFeatureEmbedding, self).__init__()
        self.encoder = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            # 第二层
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            # 第三层
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.encoder(x)

class SpectrumDecoder1DCNN(nn.Module):
    """
    深度频谱解码器
    采用渐进式上采样策略，使用线性层确保输出维度正确
    维度变化: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2501
    """
    def __init__(self, hidden_dim=256, freq_bins=2501):
        super(SpectrumDecoder1DCNN, self).__init__()
        self.freq_bins = freq_bins
        self.seq_len_init = 16  # 初始序列长度
        self.channels_init = 256  # 初始通道数
        
        # 展平层：将隐向量扩展为多通道序列
        self.fc_expand = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.channels_init * self.seq_len_init)
        )
        
        # ================= 渐进式转置卷积上采样 =================
        # 第1层: 16 -> 32
        self.deconv1 = nn.ConvTranspose1d(self.channels_init, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        
        # 第2层: 32 -> 64
        self.deconv2 = nn.ConvTranspose1d(256, 192, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(192)
        
        # 第3层: 64 -> 128
        self.deconv3 = nn.ConvTranspose1d(192, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 第4层: 128 -> 256
        self.deconv4 = nn.ConvTranspose1d(128, 96, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(96)
        
        # 第5层: 256 -> 512
        self.deconv5 = nn.ConvTranspose1d(96, 64, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        
        # 第6层: 512 -> 1024
        self.deconv6 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm1d(32)
        
        # 第7层: 1024 -> 2048
        self.deconv7 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm1d(16)
        
        # 使用线性层调整到目标维度
        # 从2048维调整到2501维
        self.final_linear = nn.Conv1d(16, 1, kernel_size=1)
        
        # 输出后处理
        self.output_process = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )
        
        # 增强版OASPL预测头（移除BatchNorm，因为动态路由时batch_size=1）
        self.oaspl_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # 增强版倍频程预测头
        self.octave_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 28)
        )
        
        # 增强版Alpha预测头
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 预测总声压级
        oaspl = self.oaspl_head(x)
        
        # 预测倍频程频谱
        octave_spectrum = self.octave_head(x)
        
        # 动态计算alpha，约束在[0.0, 80.0]
        raw_alpha = self.alpha_head(x)
        dynamic_alpha = 0.0 + 8.0 * torch.sigmoid(raw_alpha)
        
        # ================= 渐进式上采样 =================
        x_seq = self.fc_expand(x).view(-1, self.channels_init, self.seq_len_init)
        
        # 16 -> 32
        x_seq = F.gelu(self.bn1(self.deconv1(x_seq)))
        # 32 -> 64
        x_seq = F.gelu(self.bn2(self.deconv2(x_seq)))
        # 64 -> 128
        x_seq = F.gelu(self.bn3(self.deconv3(x_seq)))
        # 128 -> 256
        x_seq = F.gelu(self.bn4(self.deconv4(x_seq)))
        # 256 -> 512
        x_seq = F.gelu(self.bn5(self.deconv5(x_seq)))
        # 512 -> 1024
        x_seq = F.gelu(self.bn6(self.deconv6(x_seq)))
        # 1024 -> 2048
        x_seq = F.gelu(self.bn7(self.deconv7(x_seq)))
        
        # 转换为单通道
        x_seq = self.final_linear(x_seq)
        
        # 使用插值调整到目标维度2501
        x_seq = F.interpolate(x_seq, size=self.freq_bins, mode='linear', align_corners=False)
        
        # 输出后处理
        spectrum = self.output_process(x_seq).squeeze(1)
        
        return oaspl, octave_spectrum, spectrum, dynamic_alpha

class PI_MBCN(nn.Module):
    def __init__(self, num_modes=4, num_types=13, freq_bins=2501):
        super(PI_MBCN, self).__init__()
        self.num_modes = num_modes
        self.num_types = num_types
        # 使用增强版编码器，隐藏层维度增加到256
        self.shared_encoder = DimensionlessFeatureEmbedding(input_dim=3, hidden_dim=256)
        self.branches = nn.ModuleDict()
        for m in range(self.num_modes):
            for t in range(self.num_types):
                branch_key = f"mode_{m}_type_{t}"
                # 使用增强版解码器
                self.branches[branch_key] = SpectrumDecoder1DCNN(hidden_dim=256, freq_bins=freq_bins)

    def forward(self, x, mode_idx, type_idx):
        hidden = self.shared_encoder(x)
        batch_size = x.size(0)
        oaspl_out = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        octave_spectrum_out = torch.zeros(batch_size, 28, device=x.device, dtype=x.dtype)
        spectrum_out = torch.zeros(batch_size, self.branches["mode_0_type_0"].freq_bins, device=x.device, dtype=x.dtype)
        alpha_out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            m = mode_idx[i].long().item()
            t = type_idx[i].long().item()
            branch_key = f"mode_{m}_type_{t}"
            
            h_i = hidden[i:i+1]
            o_pred, oct_pred, s_pred, dynamic_alpha = self.branches[branch_key](h_i)
            
            oaspl_out[i] = o_pred[0]
            octave_spectrum_out[i] = oct_pred[0]
            spectrum_out[i] = s_pred[0]
            
            # 提取出标量值存入 alpha_out
            alpha_out[i] = dynamic_alpha.squeeze()
            
        return oaspl_out, octave_spectrum_out, spectrum_out, alpha_out

def physics_informed_loss_fn(pred_oaspl, pred_octave, pred_spectrum, learned_alpha, target_oaspl, target_octave, target_spectrum, inputs, rpm_mean, rpm_std, lambda_phy=1.0):
    """
    增强版物理信息损失函数
    调整损失权重，增强频谱预测的约束
    """
    # ================= 数据驱动损失 =================
    # 增加频谱MSE权重，更关注频谱细节
    loss_mse_spec = F.mse_loss(pred_spectrum, target_spectrum)
    
    # 频谱余弦相似度（形态匹配）
    loss_cosine_spec = 1.0 - F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean()
    
    # OASPL损失（增加权重）
    loss_mse_oaspl = F.mse_loss(pred_oaspl, target_oaspl)
    
    # 倍频程损失
    loss_mse_octave = F.mse_loss(pred_octave, target_octave)
    
    rpm_norm = inputs[:, 2] 
    rpm_real = rpm_norm * rpm_std + rpm_mean
    
    grad_outputs = torch.ones_like(pred_oaspl)
    gradients = torch.autograd.grad(
        outputs=pred_oaspl, inputs=inputs, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    grad_rpm_pred_norm = gradients[:, 2]
    grad_rpm_pred_physical = grad_rpm_pred_norm / rpm_std
    
    ln_10 = torch.log(torch.tensor(10.0, device=inputs.device))
    
    # 此时的 learned_alpha 是随着每个样本的工况不同而不同的向量！
    grad_rpm_theory = (10.0 * learned_alpha) / (ln_10 * rpm_real)

    loss_physics = F.mse_loss(grad_rpm_pred_physical, grad_rpm_theory)
    
    # # 由于我们在网络结构里加了硬边界 (Sigmoid 约束 [4,8])，
    # # 这里的软正则化(逼近6.0)权重可以调小甚至去掉，目前保留0.05作为偏好引导
    # loss_alpha_reg = F.mse_loss(learned_alpha, torch.tensor(6.0, device=inputs.device)) * 0.05
    
    total_loss = loss_mse_spec + 10.0 * loss_cosine_spec + 1.0 * loss_mse_oaspl + 20.0 * loss_mse_octave + lambda_phy * loss_physics
    
    return total_loss, loss_mse_spec, loss_cosine_spec, loss_mse_oaspl, loss_mse_octave, loss_physics