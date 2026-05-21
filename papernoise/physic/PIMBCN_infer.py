import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import time
# 导入模型和数据集类
from PIMBCN_net import PI_MBCN
from PIMBCN_data import PIMBCNDataset

def infer_model():
    """
    使用验证集进行推断，输出预测结果和评估指标
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推断设备: {device}")
    
    # 超参数设置（需与训练时一致）
    batch_size = 16
    freq_bins = 2501
    
    # 数据目录
    data_directory = "E:\\lyh\\paddlespeech\\csvdata333"
    
    # 创建数据集（训练集用于获取归一化参数）
    train_dataset = PIMBCNDataset(
        directory_path=data_directory,
        input_cols=[4, 5, 6],
        oaspl_col=11, 
        octave_col=12, 
        spectrum_col=13,
        type_col=3,
        mode_col=2,
        val_split=0.2,
        is_validation=False
    )
    
    # 获取验证集
    val_dataset = train_dataset.get_validation_dataset()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取RPM反归一化参数
    rpm_mean_val, rpm_std_val = train_dataset.get_rpm_norm_params()
    print(f"RPM反归一化参数 => 均值: {rpm_mean_val:.2f}, 标准差: {rpm_std_val:.2f}")
    
    # 初始化模型
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    
    # 加载训练好的模型权重
    model_path = "runs\\pi_mbcn_hvac_20260504_181809_epochs200_bs8_lr0.001_dir_csvdata333\\models\\best_model.pth"
    # 请根据实际训练结果修改上述路径
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {model_path}")
        if 'alpha_avg' in checkpoint:
            print(f"模型保存时的平均alpha: {checkpoint['alpha_avg']:.3f}")
    else:
        print(f"警告：未找到模型文件 {model_path}")
        return
    
    # 模型设置为评估模式
    model.eval()
    
    # 存储预测结果和真实标签
    all_pred_oaspl = []
    all_target_oaspl = []
    all_pred_octave = []
    all_target_octave = []
    all_pred_spectrum = []
    all_target_spectrum = []
    all_alphas = []
    
    # 计算评估指标
    total_oaspl_loss = 0.0
    total_octave_loss = 0.0
    total_spectrum_loss = 0.0
    
    print("\n开始推断...")
    with torch.no_grad():
        for inputs, types, modes, target_oaspl, target_octave, target_spectrum in val_loader:
            inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
            target_oaspl = target_oaspl.to(device)
            target_octave = target_octave.to(device)
            target_spectrum = target_spectrum.to(device)
            
            # 前向传播
            pred_oaspl, pred_octave, pred_spectrum, alpha = model(inputs, modes, types)
            
            # 存储结果
            all_pred_oaspl.append(pred_oaspl.cpu().numpy())
            all_target_oaspl.append(target_oaspl.cpu().numpy())
            all_pred_octave.append(pred_octave.cpu().numpy())
            all_target_octave.append(target_octave.cpu().numpy())
            all_pred_spectrum.append(pred_spectrum.cpu().numpy())
            all_target_spectrum.append(target_spectrum.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())
            
            # 计算损失
            total_oaspl_loss += F.mse_loss(pred_oaspl, target_oaspl).item()
            total_octave_loss += F.mse_loss(pred_octave, target_octave).item()
            total_spectrum_loss += F.mse_loss(pred_spectrum, target_spectrum).item()
    
    # 合并结果
    all_pred_oaspl = np.concatenate(all_pred_oaspl, axis=0)
    all_target_oaspl = np.concatenate(all_target_oaspl, axis=0)
    all_pred_octave = np.concatenate(all_pred_octave, axis=0)
    all_target_octave = np.concatenate(all_target_octave, axis=0)
    all_pred_spectrum = np.concatenate(all_pred_spectrum, axis=0)
    all_target_spectrum = np.concatenate(all_target_spectrum, axis=0)
    all_alphas = np.concatenate(all_alphas, axis=0)
    
    # 计算平均损失
    avg_oaspl_loss = total_oaspl_loss / len(val_loader)
    avg_octave_loss = total_octave_loss / len(val_loader)
    avg_spectrum_loss = total_spectrum_loss / len(val_loader)
    
    # ============ 计算多元评估指标 ============
    
    # ---------------- OASPL 指标 ----------------
    oaspl_mse = avg_oaspl_loss
    oaspl_mae = np.mean(np.abs(all_pred_oaspl - all_target_oaspl))
    oaspl_rmse = np.sqrt(oaspl_mse)
    oaspl_r2 = r2_score(all_target_oaspl.flatten(), all_pred_oaspl.flatten())
    oaspl_pearson, _ = pearsonr(all_target_oaspl.flatten(), all_pred_oaspl.flatten())
    oaspl_mape = np.mean(np.abs((all_pred_oaspl - all_target_oaspl) / (all_target_oaspl + 1e-8))) * 100
    oaspl_max_error = np.max(np.abs(all_pred_oaspl - all_target_oaspl))
    
    # ---------------- 三分之一倍频程频谱指标 ----------------
    octave_mse = avg_octave_loss
    octave_mae = np.mean(np.abs(all_pred_octave - all_target_octave))
    octave_rmse = np.sqrt(octave_mse)
    octave_cosine = np.mean([np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-8) 
                             for p, t in zip(all_pred_octave, all_target_octave)])
    octave_r2 = r2_score(all_target_octave.flatten(), all_pred_octave.flatten())
    octave_pearson, _ = pearsonr(all_target_octave.flatten(), all_pred_octave.flatten())
    
    # ---------------- 声压级曲线指标 ----------------
    spectrum_mse = avg_spectrum_loss
    spectrum_mae = np.mean(np.abs(all_pred_spectrum - all_target_spectrum))
    spectrum_rmse = np.sqrt(spectrum_mse)
    spectrum_cosine = np.mean([np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-8) 
                               for p, t in zip(all_pred_spectrum, all_target_spectrum)])
    spectrum_r2 = r2_score(all_target_spectrum.flatten(), all_pred_spectrum.flatten())
    spectrum_pearson, _ = pearsonr(all_target_spectrum.flatten(), all_pred_spectrum.flatten())
    
    # 频谱特征相似度
    # 峰值频率误差
    pred_peak_freq = np.argmax(all_pred_spectrum, axis=1)
    target_peak_freq = np.argmax(all_target_spectrum, axis=1)
    peak_freq_error = np.mean(np.abs(pred_peak_freq - target_peak_freq))
    
    # 峰值幅度误差
    pred_peak_val = np.max(all_pred_spectrum, axis=1)
    target_peak_val = np.max(all_target_spectrum, axis=1)
    peak_val_mae = np.mean(np.abs(pred_peak_val - target_peak_val))
    
    # 频谱重心误差（对每个样本单独计算）
    freq_bins = np.arange(all_pred_spectrum.shape[1])
    pred_spectrum_center = np.array([np.average(freq_bins, weights=p) for p in all_pred_spectrum])
    target_spectrum_center = np.array([np.average(freq_bins, weights=t) for t in all_target_spectrum])
    spectrum_center_error = np.mean(np.abs(pred_spectrum_center - target_spectrum_center))
    
    # ---------------- Alpha 统计指标 ----------------
    alpha_mean = np.mean(all_alphas)
    alpha_std = np.std(all_alphas)
    alpha_min = np.min(all_alphas)
    alpha_max = np.max(all_alphas)
    alpha_median = np.median(all_alphas)
    
    # ---------------- 综合评分 ----------------
    # 加权综合评分（归一化到[0,100]）
    normalized_scores = [
        (1 - oaspl_rmse / 10) * 25,  # OASPL评分
        (1 - octave_rmse / 5) * 25,   # 倍频程评分
        spectrum_cosine * 25,         # 频谱余弦相似度
        np.clip((6 - abs(alpha_mean - 6)) / 6, 0, 1) * 25  # Alpha合理性评分
    ]
    overall_score = np.sum(normalized_scores)
    
    # 打印评估结果
    print("\n" + "="*80)
    print("验证集多元评估结果")
    print("="*80)
    
    print(f"\n【总声压级 (OASPL) 评估】")
    print(f"  MSE (均方误差): {oaspl_mse:.4f}")
    print(f"  MAE (平均绝对误差): {oaspl_mae:.4f}")
    print(f"  RMSE (均方根误差): {oaspl_rmse:.4f}")
    print(f"  R² (决定系数): {oaspl_r2:.4f}")
    print(f"  Pearson相关系数: {oaspl_pearson:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {oaspl_mape:.2f}%")
    print(f"  Max Error (最大误差): {oaspl_max_error:.4f}")
    
    print(f"\n【三分之一倍频程频谱 (28维) 评估】")
    print(f"  MSE: {octave_mse:.4f}")
    print(f"  MAE: {octave_mae:.4f}")
    print(f"  RMSE: {octave_rmse:.4f}")
    print(f"  余弦相似度: {octave_cosine:.4f}")
    print(f"  R²: {octave_r2:.4f}")
    print(f"  Pearson相关系数: {octave_pearson:.4f}")
    
    print(f"\n【声压级曲线 (2501维) 评估】")
    print(f"  MSE: {spectrum_mse:.4f}")
    print(f"  MAE: {spectrum_mae:.4f}")
    print(f"  RMSE: {spectrum_rmse:.4f}")
    print(f"  余弦相似度: {spectrum_cosine:.4f}")
    print(f"  R²: {spectrum_r2:.4f}")
    print(f"  Pearson相关系数: {spectrum_pearson:.4f}")
    print(f"  峰值频率误差 (bins): {peak_freq_error:.2f}")
    print(f"  峰值幅度误差: {peak_val_mae:.4f}")
    print(f"  频谱重心误差 (bins): {spectrum_center_error:.2f}")
    
    print(f"\n【物理声源指数 (Alpha) 统计】")
    print(f"  均值: {alpha_mean:.3f}")
    print(f"  中位数: {alpha_median:.3f}")
    print(f"  标准差: {alpha_std:.3f}")
    print(f"  范围: [{alpha_min:.3f}, {alpha_max:.3f}]")
    print(f"  物理意义: {'单极子主导' if alpha_mean < 5 else '偶极子主导' if alpha_mean < 7 else '四极子主导'}")
    
    print(f"\n【综合评分】")
    print(f"  加权综合得分: {overall_score:.2f}/100")
    print(f"  OASPL贡献: {normalized_scores[0]:.2f}/25")
    print(f"  倍频程贡献: {normalized_scores[1]:.2f}/25")
    print(f"  频谱相似贡献: {normalized_scores[2]:.2f}/25")
    print(f"  Alpha合理贡献: {normalized_scores[3]:.2f}/25")
    
    print("\n" + "="*80)
    
    # 保存预测结果到文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"infer_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "pred_oaspl.npy"), all_pred_oaspl)
    np.save(os.path.join(output_dir, "target_oaspl.npy"), all_target_oaspl)
    np.save(os.path.join(output_dir, "pred_octave.npy"), all_pred_octave)
    np.save(os.path.join(output_dir, "target_octave.npy"), all_target_octave)
    np.save(os.path.join(output_dir, "pred_spectrum.npy"), all_pred_spectrum)
    np.save(os.path.join(output_dir, "target_spectrum.npy"), all_target_spectrum)
    np.save(os.path.join(output_dir, "alphas.npy"), all_alphas)
    
    # ================= 绘制误差折线图 =================
    plot_error_curves(
        all_pred_spectrum, all_target_spectrum, 
        all_pred_octave, all_target_octave,
        output_dir
    )
    
    print(f"\n预测结果已保存到 {output_dir} 目录")
    print("文件列表:")
    print("  - pred_oaspl.npy: 预测的总声压级")
    print("  - target_oaspl.npy: 真实的总声压级")
    print("  - pred_octave.npy: 预测的三分之一倍频程频谱")
    print("  - target_octave.npy: 真实的三分之一倍频程频谱")
    print("  - pred_spectrum.npy: 预测的声压级曲线")
    print("  - target_spectrum.npy: 真实的声压级曲线")
    print("  - alphas.npy: 预测的物理声源指数")
    print("  - spectrum_error.png: 声压级曲线误差图")
    print("  - octave_error.png: 倍频程频谱误差图")
    
    return {
        # OASPL指标
        'oaspl_mse': oaspl_mse,
        'oaspl_mae': oaspl_mae,
        'oaspl_rmse': oaspl_rmse,
        'oaspl_r2': oaspl_r2,
        'oaspl_pearson': oaspl_pearson,
        'oaspl_mape': oaspl_mape,
        'oaspl_max_error': oaspl_max_error,
        
        # 倍频程频谱指标
        'octave_mse': octave_mse,
        'octave_mae': octave_mae,
        'octave_rmse': octave_rmse,
        'octave_cosine': octave_cosine,
        'octave_r2': octave_r2,
        'octave_pearson': octave_pearson,
        
        # 声压级曲线指标
        'spectrum_mse': spectrum_mse,
        'spectrum_mae': spectrum_mae,
        'spectrum_rmse': spectrum_rmse,
        'spectrum_cosine': spectrum_cosine,
        'spectrum_r2': spectrum_r2,
        'spectrum_pearson': spectrum_pearson,
        'peak_freq_error': peak_freq_error,
        'peak_val_mae': peak_val_mae,
        'spectrum_center_error': spectrum_center_error,
        
        # Alpha指标
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'alpha_median': alpha_median,
        
        # 综合评分
        'overall_score': overall_score
    }

def plot_error_curves(pred_spectrum, target_spectrum, pred_octave, target_octave, output_dir):
    """
    绘制声压级曲线误差图和倍频程频谱误差图
    
    参数:
        pred_spectrum: 预测的声压级曲线 (n_samples, 2501)
        target_spectrum: 真实的声压级曲线 (n_samples, 2501)
        pred_octave: 预测的倍频程频谱 (n_samples, 28)
        target_octave: 真实的倍频程频谱 (n_samples, 28)
        output_dir: 输出目录
    """
    # 设置字体，支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ================= 1. 声压级曲线误差图 (前10000Hz) =================
    # 假设2501个频率点均匀分布在0-10000Hz
    freq_points = 2501  # 前10000Hz对应的点数
    freq_axis = np.linspace(0, 10000, freq_points)  # 频率轴
    
    # 计算每个频率点的平均误差
    spectrum_error = np.abs(pred_spectrum[:, :freq_points] - target_spectrum[:, :freq_points])
    mean_error = np.mean(spectrum_error, axis=0)
    std_error = np.std(spectrum_error, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, mean_error, 'b-', linewidth=2, label='平均绝对误差')
    plt.fill_between(freq_axis, mean_error - std_error, mean_error + std_error, 
                     color='blue', alpha=0.2, label='误差标准差')
    
    plt.xlabel('频率 (Hz)', fontsize=12)
    plt.ylabel('绝对误差 (dB)', fontsize=12)
    plt.title('声压级曲线频率误差分布 (0-10000Hz)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrum_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================= 2. 三分之一倍频程频谱误差图 =================
    # 28个倍频程中心频率 (参考标准三分之一倍频程)
    octave_centers = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000
    ]
    
    # 计算每个倍频程的平均误差
    octave_error = np.abs(pred_octave - target_octave)
    octave_mean_error = np.mean(octave_error, axis=0)
    octave_std_error = np.std(octave_error, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(octave_centers, octave_mean_error, yerr=octave_std_error,
                 fmt='o-', color='red', ecolor='orange', capsize=5,
                 linewidth=2, markersize=8, label='平均绝对误差')
    
    plt.xscale('log')  # 对数坐标轴
    plt.xlabel('中心频率 (Hz)', fontsize=12)
    plt.ylabel('绝对误差 (dB)', fontsize=12)
    plt.title('三分之一倍频程频谱频率误差分布', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'octave_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n误差图已保存到 {output_dir} 目录")


if __name__ == "__main__":
    results = infer_model()