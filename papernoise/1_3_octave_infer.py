import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from noisedata import Octave_1_3_data
from soundmodel import Octave_1_3_Model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 配置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei","Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# 配置参数
class InferenceConfig:
    def __init__(self):
        self.data_dir = "../csvdata333"
        self.model_path = "runs/Octave_1_3/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260201_205606/checkpoint/octave_1_3_best_model_epoch_98.pth"
        self.batch_size = 32
        self.nc = 800  # 与训练时的nc参数一致
        self.val_split = 0.2
        self.random_seed = 42
        
        # 三分之一倍频程中心频率
        self.octave_frequencies = [20 ,25 ,31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                                 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
        
        # 输入参数名称
        self.input_param_names = ['Qv 体积流量 (m3/h)', 'DP HVAC inlet (Pa)', 'N 鼓风机转速 (rpm)' ]

# 加载数据集
def load_data(config):
    print("加载数据集...")
    train_dataset = Octave_1_3_data(
        directory_path=config.data_dir,
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    val_dataset = train_dataset.get_validation_dataset()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return val_dataset, val_loader

# 加载模型
def load_model(config):
    print("加载模型...")
    model = Octave_1_3_Model(nc=config.nc).to(config.device)
    
    # 加载模型权重
    checkpoint = torch.load(config.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# 推理函数
def inference(model, val_loader, config):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets,type_ids,mode_ids in val_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            type_ids = type_ids.to(config.device)
            mode_ids = mode_ids.to(config.device)
            
            # 前向传播
            outputs = model(inputs)
            batch_size = torch.arange(inputs.size(0), device=device)
            outputs = outputs[batch_size,mode_ids,type_ids, :].squeeze()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_preds, all_targets

# 计算每个中心频率的平均误差

def calculate_errors(preds, targets, frequencies):
    # 计算绝对误差
    abs_errors = np.abs(preds - targets)
    
    # 计算每个频率的平均绝对误差
    avg_abs_errors = np.mean(abs_errors, axis=0)
    
    # 计算每个频率的均方误差
    mse_errors = np.mean((preds - targets) ** 2, axis=0)
    
    # 计算每个频率的均方根误差
    rmse_errors = np.sqrt(mse_errors)
    
    # 计算每个频率的相对误差（%）
    # 避免除以零，使用np.where
    relative_errors = np.where(targets != 0, (abs_errors / np.abs(targets)) * 100, 0)
    avg_relative_errors = np.mean(relative_errors, axis=0)
    
    # 如果频率列表长度与误差列表长度不匹配，截断或扩展频率列表
    if len(avg_abs_errors) != len(frequencies):
        frequencies = frequencies[:len(avg_abs_errors)]
    
    return avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies

# 绘制误差柱状图

def plot_error_bar_chart(avg_errors, frequencies, config):
    plt.figure(figsize=(15, 8))
    
    # 创建柱状图
    bars = plt.bar(range(len(frequencies)), avg_errors, color='skyblue')
    
    # 在每个柱子上添加误差值
    for i, (bar, error) in enumerate(zip(bars, avg_errors)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{error:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 设置图表属性
    plt.xlabel('中心频率 (Hz)', fontsize=12)
    plt.ylabel('平均绝对误差 (dBA)', fontsize=12)
    plt.title('验证集三分之一倍频程频谱平均误差', fontsize=14)
    plt.xticks(range(len(frequencies)), frequencies, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('octave_error_bar_chart.png', dpi=300)
    print("误差柱状图已保存为 octave_error_bar_chart.png")
    
    plt.show()

# 将误差结果保存为Excel文件

def save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies, config):
    # 创建DataFrame
    error_data = {
        '中心频率 (Hz)': frequencies,
        '平均绝对误差 (dBA)': avg_abs_errors,
        '平均相对误差 (%)': avg_relative_errors,
        '均方误差 (dBA²)': mse_errors,
        '均方根误差 (dBA)': rmse_errors
    }
    
    df = pd.DataFrame(error_data)
    
    # 保存为Excel文件
    excel_path = 'octave_spectrum_errors.xlsx'
    df.to_excel(excel_path, index=False)
    
    print(f"误差结果已保存为 {excel_path}")
    print("Excel文件内容：")
    print(df)
    
    return excel_path

# 绘制单个实例的真实频谱和预测频谱对比图
def plot_spectrum_comparison(pred, target, frequencies, input_params, param_names, config):
    plt.figure(figsize=(15, 10))
    
    # 创建柱状图
    width = 0.35
    x = np.arange(len(frequencies))
    
    plt.bar(x - width/2, target, width, label='真实值', color='skyblue')
    # plt.bar(x + width/2, pred, width, label='预测值', color='salmon')
    
    # # 构建输入参数文本，格式化为紧凑形式
    # param_text_parts = []
    # for name, value in zip(param_names, input_params):
    #     # 简化参数名称，只保留关键部分
    #     if '体积流量' in name:
    #         param_text_parts.append(f'Qv: {value:.1f}')
    #     elif 'HVAC' in name:
    #         param_text_parts.append(f'DP: {value:.1f}')
    #     elif '转速' in name:
    #         param_text_parts.append(f'N: {value:.0f}')
    #     elif 'Diameter' in name:
    #         param_text_parts.append(f'D: {value:.1f}')
    #     elif '流速v1' in name:
    #         param_text_parts.append(f'v1: {value:.3f}')
    #     elif '流速v2' in name:
    #         param_text_parts.append(f'v2: {value:.3f}')
    #     elif '流速v3' in name:
    #         param_text_parts.append(f'v3: {value:.3f}')
    
    # # 合并参数文本，用分号分隔
    # param_text = '; '.join(param_text_parts)
    
    # # 设置图表属性，将输入参数整合到横坐标标题中
    # plt.xlabel(f'中心频率 (Hz) ({param_text})', fontsize=10)
    # plt.ylabel('声压级 (dBA)', fontsize=12)
    # plt.title('三分之一倍频程频谱真实值与预测值对比', fontsize=14)
    # plt.xticks(x, frequencies, rotation=45)
    # plt.legend(fontsize=12)
    # plt.grid(axis='y', alpha=0.3)
    
    # # 调整布局，确保横坐标标题有足够空间显示
    # plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # # 保存图表
    # plt.savefig('spectrum_comparison.png', dpi=300, bbox_inches='tight')
    # print("频谱对比图已保存为 spectrum_comparison.png")
    
    plt.show()

# 获取单个实例的原始输入参数
def get_original_input_params(dataset, index):
    # 获取原始数据索引
    data_idx = dataset.indices[index]
    
    # 获取原始输入特征
    input_sample = dataset.data.iloc[data_idx, dataset.input_cols].values.astype(float)
    
    return input_sample

# 主函数

def main(config):
    # 加载数据
    val_dataset, val_loader = load_data(config)
    
    # 加载模型
    model = load_model(config)
    
    # 推理
    print("开始推理...")
    all_preds, all_targets = inference(model, val_loader, config)
    
    # 计算误差
    avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies = calculate_errors(
        all_preds, all_targets, config.octave_frequencies
    )
    
    # 绘制误差柱状图
    plot_error_bar_chart(avg_abs_errors, frequencies, config)
    
    # 保存误差结果到Excel文件
    save_errors_to_excel(
        avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies, config
    )
    
    # 随机选择一个实例绘制对比图
    import random
    random_index = random.randint(0, len(all_preds) - 1)
    
    # 获取原始输入参数
    original_input = get_original_input_params(val_dataset, random_index)
    
    # 绘制对比图
    plot_spectrum_comparison(
        all_preds[random_index], 
        all_targets[random_index], 
        frequencies, 
        original_input, 
        config.input_param_names, 
        config
    )
    
    print("推理完成!")

if __name__ == "__main__":
    config = InferenceConfig()
    config.device = device  # 添加device属性
    main(config)