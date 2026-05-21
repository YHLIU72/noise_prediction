import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import ast
from torch.utils.data import DataLoader
from noisedata import LineOctavedata
from soundmodel import LineOctaveModel

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置参数
class InferConfig:
    def __init__(self):
        self.data_dir = "../csvdata333"  # 数据目录
        self.model_path = "runs/line/exp_nc16000_epochs100_batchsize16_lr1e-05_wd1e-0520260201_210442/checkpoint/line_best_model_epoch.pth"
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = 42
        self.val_split = 0.2
        
        # 输入参数名称（参考1_3_octave_infer.py中的参数名称）
        self.input_param_names = ['Qv 体积流量 (m3/h)', 'DP HVAC inlet (Pa)', 'N 鼓风机转速 (rpm)']

# 加载模型
def load_model(config):
    # 初始化模型
    model = LineOctaveModel(nc=16000).to(config.device)
    
    # 加载模型权重
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载完成，加载路径: {config.model_path}")
    return model

# 加载数据
def load_data(config):
    # 加载数据集
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

# 执行推理
def infer(model, val_loader, config):
    all_predictions = []
    all_targets = []
    all_inputs = []
    all_mode_types = []
    
    with torch.no_grad():
        for inputs, targets, type_ids, mode_ids in val_loader:
            # 数据移至设备
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            type_ids = type_ids.to(config.device)
            mode_ids = mode_ids.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            batch_size = torch.arange(inputs.size(0), device=config.device)
            outputs = outputs[batch_size, mode_ids, type_ids, :].squeeze()
            
            # 保存结果
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
            all_mode_types.append(mode_ids.cpu().numpy())
    
    # 合并所有结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_mode_types = np.concatenate(all_mode_types, axis=0)
    
    return all_predictions, all_targets, all_inputs, all_mode_types

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

def plot_error_linechart(all_predictions, all_targets):
    # 计算每个频率点的误差指标
    avg_abs_errors, _, _, _, freqs = calculate_frequency_errors(all_predictions, all_targets)
    
    # 创建折线图
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, avg_abs_errors, color='red', linewidth=1)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('平均绝对误差 (dB)')
    plt.title('验证集声压级线谱频谱误差')
    plt.grid(True)
    
    # 标注部分误差值（每隔100个点标注一个）
    for i in range(0, len(avg_abs_errors), 100):
        plt.text(freqs[i], avg_abs_errors[i], f'{avg_abs_errors[i]:.2f}', 
                 ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig('line_spectrum_error.png', dpi=300)
    print("验证集误差折线图已保存为 line_spectrum_error.png")

# 将误差结果保存为Excel文件

def save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs, config):
    # 创建DataFrame
    error_data = {
        '频率 (Hz)': freqs,
        '平均绝对误差 (dB)': avg_abs_errors,
        '平均相对误差 (%)': avg_relative_errors,
        '均方误差 (dB²)': mse_errors,
        '均方根误差 (dB)': rmse_errors
    }
    
    df = pd.DataFrame(error_data)
    
    # 保存为Excel文件
    excel_path = 'line_spectrum_errors.xlsx'
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
    original_inputs = inputs[idx] * input_std + input_mean
    
    # 生成频率轴
    freqs = np.linspace(0, 10000, num=2501)  # 假设频率范围是0-10kHz
    
    # 创建对比图
    plt.figure(figsize=(12, 8))
    plt.plot(freqs, targets[idx], color='blue', linewidth=1, label='真实频谱')
    plt.plot(freqs, predictions[idx], color='red', linewidth=1, label='预测频谱')
    
    # 构建输入参数文本，格式化为紧凑形式（参考1_3_octave_infer.py的做法）
    param_text_parts = []
    for name, value in zip(config.input_param_names, original_inputs):
        # 简化参数名称，只保留关键部分
        if '体积流量' in name:
            param_text_parts.append(f'Qv: {value:.1f}')
        elif 'HVAC' in name:
            param_text_parts.append(f'DP: {value:.1f}')
        elif '转速' in name:
            param_text_parts.append(f'N: {value:.0f}')
        elif 'Diameter' in name:
            param_text_parts.append(f'D: {value:.1f}')
        elif '流速v1' in name:
            param_text_parts.append(f'v1: {value:.3f}')
        elif '流速v2' in name:
            param_text_parts.append(f'v2: {value:.3f}')
        elif '流速v3' in name:
            param_text_parts.append(f'v3: {value:.3f}')
    
    # 合并参数文本，用分号分隔
    param_text = '; '.join(param_text_parts)
    
    # 将参数文本添加到横坐标标题的括号中
    plt.xlabel(f'频率 (Hz) ({param_text})')
    plt.ylabel('声压级 (dB)')
    plt.title('单个实例的真实频谱与预测频谱对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'line_spectrum_comparison_{idx}.png', dpi=300)
    print(f"实例 {idx} 的频谱对比图已保存为 line_spectrum_comparison_{idx}.png")


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
    all_predictions, all_targets, all_inputs, all_mode_ids = infer(model, val_loader, config)
    
    # 计算每个频率点的误差指标
    print("计算各频率点误差指标...")
    avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, freqs = calculate_frequency_errors(all_predictions, all_targets)
    
    # 绘制验证集误差折线图
    print("绘制验证集误差折线图...")
    plot_error_linechart(all_predictions, all_targets)
    
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
    
    # 传递config参数给plot_prediction_comparison函数
    plot_prediction_comparison(all_predictions, all_targets, all_inputs, norm_params, random_idx, config)
    
    print("推理完成!")

if __name__ == "__main__":
    main()