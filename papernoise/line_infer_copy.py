import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import DataLoader
from noisedata_copy import LineOctavedata
from soundmodel_copy import LineOctaveModel

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei",  "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置参数
class InferConfig:
    def __init__(self):
        self.data_dir = "../csvdata333"  # 数据目录（与line_train_copy.py一致）
        self.model_path = "runs/line/exp_nc16000_epochs100_batchsize16_lr1e-05_wd1e-0520260203_151101/checkpoint/line_best_model_epoch.pth"  # 模型路径（与line_train_copy.py一致）
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = 42
        self.val_split = 0.2
        self.nc = 16000  # 与line_train_copy.py一致
        
        # 输入参数名称
        self.input_param_names = ['Qv 体积流量 (m3/h)', 'DP HVAC inlet (Pa)', 'N 鼓风机转速 (rpm)']
        
        # 图表和结果保存路径
        self.save_dir = "line_infer_copy_results"
        os.makedirs(self.save_dir, exist_ok=True)

# 加载模型
def load_model(config):
    # 初始化模型 - 与line_train_copy.py一致
    model = LineOctaveModel(nc=config.nc).to(config.device)
    
    # 加载模型权重
    checkpoint = torch.load(config.model_path, map_location=config.device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"模型加载完成，加载路径: {config.model_path}")
    return model

# 加载数据
def load_data(config):
    # 加载数据集 - 与line_train_copy.py一致
    train_dataset = LineOctavedata(
        directory_path=config.data_dir,
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    val_dataset = train_dataset.get_validation_dataset()
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    print(f"数据加载完成，验证集样本数: {len(val_dataset)}")
    return val_dataset, val_loader

# 执行推理 - 结合line_train_copy.py和line_infer.py的推理逻辑
def infer(model, val_loader, config):
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 数据移至设备
            inputs = inputs.to(config.device)
            # 与line_train_copy.py一致，只取前2501个点
            targets = targets[:,:2501].to(config.device)
            
            # 前向传播 - 与line_train_copy.py一致，直接使用模型输出
            outputs = model(inputs)
            
            # 保存结果
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
    
    # 合并所有结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)
    
    return all_predictions, all_targets, all_inputs

# 计算每个频率点的误差指标
def calculate_frequency_errors(all_predictions, all_targets):
    # 生成频率轴（0-10kHz，2501个点）
    freqs = np.linspace(0, 10000, num=2501)
    
    # 计算绝对误差
    abs_errors = np.abs(all_predictions - all_targets)
    
    # 计算每个频率点的平均绝对误差
    avg_abs_errors = np.mean(abs_errors, axis=0)
    
    # 计算每个频率点的均方误差
    mse_errors = np.mean((all_predictions - all_targets) ** 2, axis=0)
    
    # 计算每个频率点的均方根误差
    rmse_errors = np.sqrt(mse_errors)
    
    # 计算每个频率点的相对误差（%）
    # 避免除以零，使用np.where
    relative_errors = np.where(all_targets != 0, (abs_errors / np.abs(all_targets)) * 100, 0)
    avg_relative_errors = np.mean(relative_errors, axis=0)
    
    return avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs

# 绘制验证集误差折线图
def plot_error_linechart(all_predictions, all_targets, config):
    # 计算每个频率点的误差指标
    avg_abs_errors, _, _, _, freqs = calculate_frequency_errors(all_predictions, all_targets)
    
    # 创建折线图
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, avg_abs_errors, color='red', linewidth=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Absolute Error (dB)')
    plt.title('Validation Set Line Spectrum Error')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    save_path = os.path.join(config.save_dir, 'line_spectrum_error.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"误差折线图已保存至: {save_path}")
    plt.close()

# 将误差结果保存为Excel文件
def save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs, config):
    # 创建DataFrame
    error_data = {
        'Frequency (Hz)': freqs,
        'Mean Absolute Error (dB)': avg_abs_errors,
        'Mean Relative Error (%)': avg_relative_errors,
        'Mean Squared Error (dB²)': mse_errors,
        'Root Mean Squared Error (dB)': rmse_errors
    }
    
    df = pd.DataFrame(error_data)
    
    # 保存为Excel文件
    excel_path = os.path.join(config.save_dir, 'line_spectrum_errors.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"误差结果已保存为 {excel_path}")
    print("Excel文件前10行内容：")
    print(df.head(10))
    
    return excel_path

# 绘制单个实例的真实与预测频谱对比图
def plot_prediction_comparison(predictions, targets, inputs, norm_params, idx, config):
    # 反归一化输入参数
    input_mean = norm_params['input_mean']
    input_std = norm_params['input_std']
    # original_inputs = inputs[idx] * input_std + input_mean
    
    # 生成频率轴
    freqs = np.linspace(0, 10000, num=2501)
    
    # 创建对比图
    plt.figure(figsize=(12, 8))
    plt.plot(freqs, targets[idx], color='blue', linewidth=1, label='Measured Spectrum')
    plt.plot(freqs, predictions[idx], color='red', linewidth=1, label='Predicted Spectrum')
    
    # 设置图表属性
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Sound Pressure Level (dB)')
    plt.title('Single Instance Line Spectrum Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.save_dir, f'line_spectrum_comparison_{idx}.png')
    plt.savefig(save_path, dpi=300)
    print(f"频谱对比图已保存至: {save_path}")
    plt.close()

# 主函数
def main():
    # 初始化配置
    config = InferConfig()
    print(f"使用设备: {config.device}")
    
    # 加载模型
    model = load_model(config)
    
    # 加载数据
    val_dataset, val_loader = load_data(config)
    
    # 执行推理
    print("开始推理...")
    all_predictions, all_targets, all_inputs = infer(model, val_loader, config)
    
    # 计算每个频率点的误差指标
    print("计算各频率点误差指标...")
    avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs = calculate_frequency_errors(all_predictions, all_targets)
    
    # 绘制验证集误差折线图
    print("绘制验证集误差折线图...")
    plot_error_linechart(all_predictions, all_targets, config)
    
    # 将误差结果保存为Excel文件
    print("保存误差结果到Excel文件...")
    save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs, config)
    
    # 随机选择一个实例绘制对比图
    print("绘制单个实例频谱对比图...")
    np.random.seed(config.random_seed)
    random_idx = np.random.randint(0, len(all_predictions))
    
    # 获取归一化参数
    norm_params = {
        'input_mean': val_dataset.input_mean,
        'input_std': val_dataset.input_std
    }
    
    # 绘制对比图
    plot_prediction_comparison(all_predictions, all_targets, all_inputs, norm_params, random_idx, config)
    
    print("推理完成!")

if __name__ == "__main__":
    main()