"""
PIMBCN 推理评估程序（Windows 兼容版）

功能：
1. 加载训练好的模型权重
2. 在训练集和验证集上进行推理
3. 计算并输出完整的评估指标
4. 支持多种格式的结果输出
"""
import os
# 解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



from PIMBCN_net0601_ncjianshao import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0529 import PIMBCNDataset


def load_full_dataset_to_gpu(dataset, device):
    """将数据集一次性完整拉入显存"""
    print(f"正在将数据集载入显存 ({len(dataset)} 条)...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for inputs, types, modes, _, _, target_spectrum in loader:
        return (
            inputs.to(device, non_blocking=True),
            types.to(device, non_blocking=True),
            modes.to(device, non_blocking=True),
            target_spectrum.to(device, non_blocking=True)
        )


def compute_metrics(pred_spectrum, target_spectrum, loss_wrapper):
    """计算所有评估指标"""
    with torch.no_grad():
        total_loss, loss_mse, loss_cosine, loss_oaspl, loss_grad = loss_wrapper(
            pred_spectrum, target_spectrum
        )
        
        # 额外计算一些统计指标
        mae = torch.mean(torch.abs(pred_spectrum - target_spectrum)).item()
        rmse = torch.sqrt(loss_mse).item()
        
        # 计算决定系数 R²
        ss_tot = torch.sum((target_spectrum - target_spectrum.mean()) ** 2).item()
        ss_res = torch.sum((target_spectrum - pred_spectrum) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # 计算相对误差
        rel_error = torch.mean(torch.abs(pred_spectrum - target_spectrum) / (torch.abs(target_spectrum) + 1e-10)).item()
    
    return {
        'total_loss': total_loss.item(),
        'mse': loss_mse.item(),
        'mae': mae,
        'rmse': rmse,
        'cosine_distance': loss_cosine.item(),
        'cosine_similarity': 1 - loss_cosine.item(),
        'oaspl_loss': loss_oaspl.item(),
        'gradient_loss': loss_grad.item(),
        'r2': r2,
        'relative_error': rel_error
    }


def evaluate_model(model, loss_wrapper, data_X, data_T, data_M, data_Y, device, batch_size=32):
    """在数据集上评估模型"""
    model.eval()
    loss_wrapper.eval()
    
    num_samples = data_X.size(0)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="推理进度", leave=False):
            inputs = data_X[i:i + batch_size]
            types = data_T[i:i + batch_size]
            modes = data_M[i:i + batch_size]
            target_spectrum = data_Y[i:i + batch_size]
            
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                pred_spectrum = model(inputs, modes, types)
            
            all_preds.append(pred_spectrum.cpu().numpy())
            all_targets.append(target_spectrum.cpu().numpy())
    
    # 合并所有预测结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 转回 GPU 计算指标
    pred_tensor = torch.from_numpy(all_preds).to(device)
    target_tensor = torch.from_numpy(all_targets).to(device)
    
    metrics = compute_metrics(pred_tensor, target_tensor, loss_wrapper)
    
    return metrics, all_preds, all_targets


def print_metrics(title, metrics):
    """格式化打印评估指标"""
    print(f"\n{'='*60}")
    print(f"【{title}】")
    print(f"{'='*60}")
    print(f"综合损失 (Total Loss):         {metrics['total_loss']:.6f}")
    print(f"谱 MSE (Spectrum MSE):         {metrics['mse']:.6f}")
    print(f"谱 MAE (Spectrum MAE):         {metrics['mae']:.6f}")
    print(f"谱 RMSE (Spectrum RMSE):       {metrics['rmse']:.6f}")
    print(f"Cosine 相似度 (Cosine Sim):    {metrics['cosine_similarity']:.6f}")
    print(f"Cosine 距离 (Cosine Dist):     {metrics['cosine_distance']:.6f}")
    print(f"OASPL MSE (OASPL Loss):        {metrics['oaspl_loss']:.6f}")
    print(f"梯度损失 (Gradient Loss):      {metrics['gradient_loss']:.6f}")
    print(f"决定系数 R² (R-squared):       {metrics['r2']:.6f}")
    print(f"相对误差 (Relative Error):     {metrics['relative_error']:.6f}")
    print(f"{'='*60}")


def plot_spectrum_error(preds, targets, output_dir, dataset_name, sample_indices=[0, 1, 2]):
    """绘制声压级曲线误差图"""
    # 创建频率轴（假设频率范围为 20Hz - 10kHz，共 2501 个点）
    freq_axis = np.linspace(20, 10000, preds.shape[1])
    
    # 选择要绘制的样本
    num_samples_to_plot = min(len(sample_indices), preds.shape[0])
    
    # 绘制单个样本的频谱对比图
    for idx in sample_indices[:num_samples_to_plot]:
        plt.figure(figsize=(12, 6))
        plt.plot(freq_axis, targets[idx], label='目标值 (Target)', color='blue', linewidth=2)
        plt.plot(freq_axis, preds[idx], label='预测值 (Prediction)', color='red', linewidth=2, linestyle='--')
        plt.plot(freq_axis, np.abs(preds[idx] - targets[idx]), label='绝对误差 (Absolute Error)', 
                 color='green', linewidth=1.5, linestyle=':')
        
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('声压级 (dB)', fontsize=12)
        plt.title(f'{dataset_name} - 样本 {idx} 声压级频谱对比', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xscale('log')
        plt.xlim(20, 10000)
        plt.gca().xaxis.set_major_locator(MultipleLocator(base=1000))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectrum_sample_{idx}_{dataset_name.lower()}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # 绘制所有样本的平均误差分布
    mean_error = np.mean(np.abs(preds - targets), axis=0)
    std_error = np.std(np.abs(preds - targets), axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis, mean_error, label='平均绝对误差', color='red', linewidth=2)
    plt.fill_between(freq_axis, mean_error - std_error, mean_error + std_error, 
                     color='red', alpha=0.2, label='误差标准差')
    plt.xlabel('频率 (Hz)', fontsize=12)
    plt.ylabel('平均绝对误差 (dB)', fontsize=12)
    plt.title(f'{dataset_name} - 频率维度误差分布', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.xlim(20, 10000)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_distribution_{dataset_name.lower()}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 绘制预测值与目标值的散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(targets.flatten(), preds.flatten(), alpha=0.3, s=10)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
             'r--', label='理想线')
    plt.xlabel('目标声压级 (dB)', fontsize=12)
    plt.ylabel('预测声压级 (dB)', fontsize=12)
    plt.title(f'{dataset_name} - 预测值 vs 目标值散点图', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_plot_{dataset_name.lower()}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  {dataset_name} 频谱误差图已保存")


def save_results(train_metrics, val_metrics, train_preds, train_targets, val_preds, val_targets, output_dir):
    """保存评估结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标到 CSV（指定UTF-8编码）
    with open(os.path.join(output_dir, 'evaluation_metrics.csv'), 'w', encoding='utf-8') as f:
        f.write("Metric,Train,Validation\n")
        for key in train_metrics.keys():
            f.write(f"{key},{train_metrics[key]},{val_metrics[key]}\n")
    
    # 保存详细报告（指定UTF-8编码）
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PIMBCN 模型评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("【训练集评估结果】\n")
        f.write("-"*40 + "\n")
        for key, value in train_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n【验证集评估结果】\n")
        f.write("-"*40 + "\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n【关键指标对比】\n")
        f.write("-"*40 + "\n")
        f.write(f"MSE 提升: {(train_metrics['mse'] - val_metrics['mse']) / train_metrics['mse'] * 100:.2f}%\n")
        f.write(f"Cosine 相似度下降: {(train_metrics['cosine_similarity'] - val_metrics['cosine_similarity']) * 100:.2f}%\n")
        f.write(f"R² 下降: {(train_metrics['r2'] - val_metrics['r2']) * 100:.2f}%\n")
    
    # 绘制频谱误差图
    print("\n正在绘制频谱误差图...")
    plot_spectrum_error(train_preds, train_targets, output_dir, "训练集")
    plot_spectrum_error(val_preds, val_targets, output_dir, "验证集")
    
    print(f"\n评估结果已保存到: {output_dir}")


def main(model_path, data_dir=None):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")
    
    # 默认数据目录
    if data_dir is None:
        data_dir = "F:\\lyh\\paddlespeech\\csvdata333"
    
    # 加载模型
    print(f"\n正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 初始化模型
    freq_bins = 2501
    model = PI_MBCN(num_modes=4, num_types=13, freq_bins=freq_bins).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_wrapper = PhysicsLossWrapper().to(device)
    
    print(f"模型加载成功！训练停止于 epoch: {checkpoint.get('epoch', '未知')}")
    print(f"最佳验证损失: {checkpoint.get('best_val_loss', '未知')}")
    
    # 加载数据集
    print("\n正在加载数据集...")
    train_dataset = PIMBCNDataset(
        directory_path=data_dir, input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2, val_split=0.2, is_validation=False
    )
    val_dataset = train_dataset.get_validation_dataset()
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")
    
    # 预载入显存
    train_X, train_T, train_M, train_Y = load_full_dataset_to_gpu(train_dataset, device)
    val_X, val_T, val_M, val_Y = load_full_dataset_to_gpu(val_dataset, device)
    
    # 评估训练集
    print("\n正在评估训练集...")
    train_metrics, train_preds, train_targets = evaluate_model(
        model, loss_wrapper, train_X, train_T, train_M, train_Y, device
    )
    
    # 评估验证集
    print("\n正在评估验证集...")
    val_metrics, val_preds, val_targets = evaluate_model(
        model, loss_wrapper, val_X, val_T, val_M, val_Y, device
    )
    
    # 打印结果
    print_metrics("训练集评估结果", train_metrics)
    print_metrics("验证集评估结果", val_metrics)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(model_path), 'evaluation')
    save_results(train_metrics, val_metrics, train_preds, train_targets, val_preds, val_targets, output_dir)
    
    # 输出关键对比
    print("\n【关键指标对比】")
    print(f"{'指标':<20} {'训练集':<15} {'验证集':<15} {'差异':<15}")
    print(f"{'-'*60}")
    for key in ['mse', 'cosine_similarity', 'r2', 'relative_error']:
        train_val = train_metrics[key]
        val_val = val_metrics[key]
        diff = val_val - train_val
        diff_pct = (diff / train_val * 100) if train_val != 0 else 0
        print(f"{key:<20} {train_val:<15.6f} {val_val:<15.6f} {diff_pct:<15.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PIMBCN 推理评估程序')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--data', type=str, default=None, help='数据目录路径')
    args = parser.parse_args()
    
    main(args.model, args.data)