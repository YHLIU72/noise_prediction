"""
PIMBCN 模型推理与评估程序
- 加载训练好的模型进行推理
- 分别对训练集和验证集进行评估
- 计算多种评价指标（MSE, MAE, RMSE, Cosine相似度, OASPL误差等）
- 绘制声压级误差曲线图
- 计算每个频率的误差值和标准差
"""
import os

# 解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

from PIMBCN_net0529 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0529 import PIMBCNDataset


def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=2501).to(device)
    loss_wrapper = PhysicsLossWrapper().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_wrapper.load_state_dict(checkpoint['loss_wrapper_state_dict'])
    
    model.eval()
    loss_wrapper.eval()
    
    print("模型加载完成！")
    return model, loss_wrapper


def compute_metrics(pred_spectrum, target_spectrum, loss_wrapper):
    """计算各种评价指标"""
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(pred_spectrum, target_spectrum).item()
        
        # RMSE
        rmse = torch.sqrt(F.mse_loss(pred_spectrum, target_spectrum)).item()
        
        # MAE
        mae = F.l1_loss(pred_spectrum, target_spectrum).item()
        
        # Cosine相似度
        cosine_sim = F.cosine_similarity(pred_spectrum, target_spectrum, dim=1).mean().item()
        
        # OASPL误差
        pred_oaspl = loss_wrapper._safe_oaspl(pred_spectrum)
        target_oaspl = loss_wrapper._safe_oaspl(target_spectrum)
        oaspl_mse = F.mse_loss(pred_oaspl, target_oaspl).item()
        oaspl_mae = F.l1_loss(pred_oaspl, target_oaspl).item()
        
        # Sobolev梯度损失
        grad_loss = loss_wrapper._sobolev_gradient_loss(pred_spectrum, target_spectrum).item()
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Cosine_Similarity': cosine_sim,
        'OASPL_MSE': oaspl_mse,
        'OASPL_MAE': oaspl_mae,
        'Gradient_Loss': grad_loss
    }


def compute_freq_error_stats(pred_spectrum, target_spectrum):
    """计算每个频率的误差值和标准差"""
    # 计算每个频率的误差
    errors = pred_spectrum - target_spectrum  # [N, freq_bins]
    
    # 每个频率的平均误差（bias）
    freq_mean_error = torch.mean(errors, dim=0).cpu().numpy()
    
    # 每个频率的标准差
    freq_std = torch.std(errors, dim=0).cpu().numpy()
    
    # 每个频率的RMSE
    freq_rmse = torch.sqrt(torch.mean(errors**2, dim=0)).cpu().numpy()
    
    return freq_mean_error, freq_std, freq_rmse


def evaluate_dataset(model, loss_wrapper, dataset, device, dataset_name):
    """评估单个数据集"""
    print(f"\n正在处理 {dataset_name}...")
    
    # 将数据集加载到GPU
    data_X = []
    data_T = []
    data_M = []
    data_Y = []
    
    for i in range(len(dataset)):
        inputs, types, modes, oaspl, octave, spectrum = dataset[i]
        data_X.append(inputs)
        data_T.append(types)
        data_M.append(modes)
        data_Y.append(spectrum)
    
    data_X = torch.stack(data_X).to(device)
    data_T = torch.stack(data_T).to(device)
    data_M = torch.stack(data_M).to(device)
    data_Y = torch.stack(data_Y).to(device)
    
    print(f"{dataset_name} 大小: {len(dataset)} 条")
    
    # 执行推理
    print(f"正在执行 {dataset_name} 推理...")
    with torch.no_grad():
        pred_spectrum = model(data_X, data_M, data_T)
    
    # 计算评价指标
    print(f"计算 {dataset_name} 评价指标...")
    metrics = compute_metrics(pred_spectrum, data_Y, loss_wrapper)
    
    # 计算频率误差统计
    print(f"计算 {dataset_name} 频率误差统计...")
    freq_mean_error, freq_std, freq_rmse = compute_freq_error_stats(pred_spectrum, data_Y)
    
    return pred_spectrum, data_Y, metrics, freq_mean_error, freq_std, freq_rmse


def plot_results(pred_spectrum, target_spectrum, freq_mean_error, freq_std, freq_rmse, save_dir, dataset_name):
    """绘制各种结果图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建频率轴（假设2501个频率点，采样率约为5000Hz，频率范围0-2500Hz）
    freq_axis = np.linspace(0, 2500, 2501)
    
    # 1. 绘制预测谱与真实谱对比图（随机选几个样本）
    plt.figure(figsize=(12, 6))
    sample_indices = [0, 1, 2, 3]  # 选前4个样本展示
    for idx in sample_indices:
        plt.plot(freq_axis, target_spectrum[idx].cpu().numpy(), label=f'Target {idx}', alpha=0.6)
        plt.plot(freq_axis, pred_spectrum[idx].cpu().numpy(), label=f'Pred {idx}', linestyle='--', alpha=0.6)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectrum Value')
    plt.title(f'{dataset_name} - Predicted vs Target Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'spectrum_comparison_{dataset_name.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制频率误差曲线图（平均误差）
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, freq_mean_error, label='Mean Error', color='blue')
    plt.fill_between(freq_axis, freq_mean_error - freq_std, freq_mean_error + freq_std, 
                    alpha=0.3, label='±1 Std', color='blue')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Error')
    plt.title(f'{dataset_name} - Frequency-wise Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'freq_error_dist_{dataset_name.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制频率RMSE图
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, freq_rmse, label='RMSE per Frequency', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RMSE')
    plt.title(f'{dataset_name} - RMSE Distribution Across Frequencies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'freq_rmse_{dataset_name.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 绘制误差直方图
    plt.figure(figsize=(10, 6))
    errors_flat = (pred_spectrum - target_spectrum).cpu().numpy().flatten()
    plt.hist(errors_flat, bins=100, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Error Value')
    plt.ylabel('Count')
    plt.title(f'{dataset_name} - Error Distribution Histogram')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'error_histogram_{dataset_name.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{dataset_name} 图表已保存到: {save_dir}")


def plot_comparison(train_stats, val_stats, save_dir):
    """绘制训练集和验证集的对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    freq_axis = np.linspace(0, 2500, 2501)
    
    # 1. RMSE对比图
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, train_stats['freq_rmse'], label='Training Set', color='blue')
    plt.plot(freq_axis, val_stats['freq_rmse'], label='Validation Set', color='red', linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison - Training vs Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rmse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 平均误差对比图
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, train_stats['freq_mean_error'], label='Training Set', color='blue')
    plt.plot(freq_axis, val_stats['freq_mean_error'], label='Validation Set', color='red', linestyle='--')
    plt.axhline(y=0, color='green', linestyle=':', label='Zero Error')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Error')
    plt.title('Mean Error Comparison - Training vs Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'mean_error_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {save_dir}")


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")
    
    # 加载模型
    model, loss_wrapper = load_model(args.model_path, device)
    
    # 加载数据集
    print("\n正在加载数据集...")
    data_directory = args.data_dir
    train_dataset = PIMBCNDataset(
        directory_path=data_directory, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False
    )
    val_dataset = train_dataset.get_validation_dataset()
    
    # 评估训练集
    train_pred, train_target, train_metrics, train_freq_mean, train_freq_std, train_freq_rmse = \
        evaluate_dataset(model, loss_wrapper, train_dataset, device, "Training Set")
    
    # 评估验证集
    val_pred, val_target, val_metrics, val_freq_mean, val_freq_std, val_freq_rmse = \
        evaluate_dataset(model, loss_wrapper, val_dataset, device, "Validation Set")
    
    # 打印训练集评价指标
    print("\n" + "="*70)
    print("训练集性能评价指标")
    print("="*70)
    for metric_name, value in train_metrics.items():
        print(f"{metric_name:20s}: {value:.6f}")
    print("="*70)
    
    # 打印验证集评价指标
    print("\n" + "="*70)
    print("验证集性能评价指标")
    print("="*70)
    for metric_name, value in val_metrics.items():
        print(f"{metric_name:20s}: {value:.6f}")
    print("="*70)
    
    # 打印对比摘要
    print("\n" + "="*70)
    print("训练集 vs 验证集 指标对比")
    print("="*70)
    for metric_name in train_metrics.keys():
        train_val = train_metrics[metric_name]
        val_val = val_metrics[metric_name]
        diff = val_val - train_val
        print(f"{metric_name:20s}: Train={train_val:.6f} | Val={val_val:.6f} | Diff={diff:+.6f}")
    print("="*70)
    
    # 打印训练集频率统计摘要
    print("\n训练集频率误差统计摘要:")
    print(f"平均频率误差: {np.mean(train_freq_mean):.6f}")
    print(f"频率误差标准差: {np.mean(train_freq_std):.6f}")
    print(f"平均频率RMSE: {np.mean(train_freq_rmse):.6f}")
    print(f"最大频率RMSE: {np.max(train_freq_rmse):.6f}")
    print(f"最小频率RMSE: {np.min(train_freq_rmse):.6f}")
    
    # 打印验证集频率统计摘要
    print("\n验证集频率误差统计摘要:")
    print(f"平均频率误差: {np.mean(val_freq_mean):.6f}")
    print(f"频率误差标准差: {np.mean(val_freq_std):.6f}")
    print(f"平均频率RMSE: {np.mean(val_freq_rmse):.6f}")
    print(f"最大频率RMSE: {np.max(val_freq_rmse):.6f}")
    print(f"最小频率RMSE: {np.min(val_freq_rmse):.6f}")
    
    # 绘制结果图
    print("\n绘制结果图...")
    save_dir = os.path.join(os.path.dirname(args.model_path), 'eval_results')
    
    # 绘制训练集图表
    plot_results(train_pred, train_target, train_freq_mean, train_freq_std, train_freq_rmse, save_dir, "Training Set")
    
    # 绘制验证集图表
    plot_results(val_pred, val_target, val_freq_mean, val_freq_std, val_freq_rmse, save_dir, "Validation Set")
    
    # 绘制对比图表
    train_stats = {
        'freq_rmse': train_freq_rmse,
        'freq_mean_error': train_freq_mean
    }
    val_stats = {
        'freq_rmse': val_freq_rmse,
        'freq_mean_error': val_freq_mean
    }
    plot_comparison(train_stats, val_stats, save_dir)
    
    # 保存数值结果
    np.savez(os.path.join(save_dir, 'frequency_stats_train.npz'),
             freq_mean_error=train_freq_mean,
             freq_std=train_freq_std,
             freq_rmse=train_freq_rmse)
    
    np.savez(os.path.join(save_dir, 'frequency_stats_val.npz'),
             freq_mean_error=val_freq_mean,
             freq_std=val_freq_std,
             freq_rmse=val_freq_rmse)
    
    # 保存评价指标
    import json
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'train': train_metrics,
            'validation': val_metrics
        }, f, indent=4)
    
    print("\n推理与评估完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIMBCN 模型推理与评估')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='训练好的模型文件路径 (best_model.pth)')
    parser.add_argument('--data_dir', type=str, default='F:\\lyh\\paddlespeech\\csvdata333',
                        help='数据目录路径')
    args = parser.parse_args()
    
    main(args)