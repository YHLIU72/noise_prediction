import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from noisedata_copy import Octave_1_3_data
from soundmodel_copy import Octave_1_3_Model

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置参数
class InferConfig:
    def __init__(self):
        self.data_dir = "../csvdata333"  # 数据目录（与Octave_1_3_train_copy.py一致）
        self.model_path = "runs/Octave_1_3/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260203_150806/checkpoint/octave_1_3_best_model_epoch_92.pth"  # 模型路径（与Octave_1_3_train_copy.py一致）
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = 42
        self.val_split = 0.2
        self.nc = 800  # 与Octave_1_3_train_copy.py一致
        
        # 三分之一倍频程中心频率（标准的28个频率点）
        self.octave_frequencies = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                                 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
        
        # 输入参数名称
        self.input_param_names = ['Qv 体积流量 (m3/h)', 'DP HVAC inlet (Pa)', 'N 鼓风机转速 (rpm)']
        
        # 图表和结果保存路径
        self.save_dir = "1_3_octave_infer_copy_results"
        os.makedirs(self.save_dir, exist_ok=True)

# 加载模型
def load_model(config):
    # 初始化模型 - 与Octave_1_3_train_copy.py一致
    model = Octave_1_3_Model(nc=config.nc).to(config.device)
    
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
    # 加载数据集 - 与Octave_1_3_train_copy.py一致
    train_dataset = Octave_1_3_data(
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

# 执行推理 - 结合Octave_1_3_train_copy.py和1_3_octave_infer.py的推理逻辑
def infer(model, val_loader, config):
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # 数据移至设备
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # 前向传播 - 与Octave_1_3_train_copy.py一致，直接使用模型输出
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

# 计算每个三分之一倍频程中心频率的误差指标
def calculate_frequency_errors(all_predictions, all_targets, frequencies):
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
    
    # 确保频率列表长度与误差列表长度一致
    if len(avg_abs_errors) != len(frequencies):
        frequencies = frequencies[:len(avg_abs_errors)]
    
    return avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies

# 绘制误差柱状图
def plot_error_bar_chart(avg_errors, frequencies, config):
    plt.figure(figsize=(15, 8))
    
    # 创建柱状图
    bars = plt.bar(range(len(frequencies)), avg_errors, color='skyblue', edgecolor='black', linewidth=1.5)
    
    # 在每个柱子上添加误差值
    for i, (bar, error) in enumerate(zip(bars, avg_errors)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{error:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 设置图表属性
    plt.xlabel('Center Frequency (Hz)', fontsize=16, fontweight='bold')
    plt.ylabel('Mean Absolute Error (dBA)', fontsize=16, fontweight='bold')
    plt.title('1/3 Octave Band Spectrum Average Error on Validation Set', fontsize=18, fontweight='bold')
    plt.xticks(range(len(frequencies)), frequencies, rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # 设置图框加粗描黑
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.save_dir, 'octave_error_bar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"误差柱状图已保存至: {save_path}")
    plt.close()

# 将误差结果保存为Excel文件
def save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies, config):
    # 创建DataFrame
    error_data = {
        'Center Frequency (Hz)': frequencies,
        'Mean Absolute Error (dBA)': avg_abs_errors,
        'Mean Relative Error (%)': avg_relative_errors,
        'Mean Squared Error (dBA²)': mse_errors,
        'Root Mean Squared Error (dBA)': rmse_errors
    }
    
    df = pd.DataFrame(error_data)
    
    # 保存为Excel文件
    excel_path = os.path.join(config.save_dir, 'octave_spectrum_errors.xlsx')
    df.to_excel(excel_path, index=False)
    
    print(f"误差结果已保存为 {excel_path}")
    print("Excel文件内容：")
    print(df)
    
    return excel_path

# 绘制单个实例的真实与预测频谱对比图
def plot_spectrum_comparison(pred, target, frequencies, input_params, param_names, config):
    plt.figure(figsize=(15, 10))
    
    # 创建柱状图
    width = 0.35
    x = np.arange(len(frequencies))
    
    plt.bar(x - width/2, target, width, label='Measured Spectrum', color='skyblue', edgecolor='black', linewidth=1)
    plt.bar(x + width/2, pred, width, label='Predicted Spectrum', color='salmon', edgecolor='black', linewidth=1)
    
    # 设置图表属性
    plt.xlabel('Center Frequency (Hz)', fontsize=16, fontweight='bold')
    plt.ylabel('Sound Pressure Level (dBA)', fontsize=16, fontweight='bold')
    plt.title('Single Instance 1/3 Octave Band Spectrum Comparison', fontsize=18, fontweight='bold')
    plt.xticks(x, frequencies, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(config.save_dir, 'octave_spectrum_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies = calculate_frequency_errors(
        all_predictions, all_targets, config.octave_frequencies
    )
    
    # 绘制误差柱状图
    print("绘制误差柱状图...")
    plot_error_bar_chart(avg_abs_errors, frequencies, config)
    
    # 将误差结果保存为Excel文件
    print("保存误差结果到Excel文件...")
    save_errors_to_excel(avg_abs_errors, avg_relative_errors, mse_errors, rmse_errors, frequencies, config)
    
    # 随机选择一个实例绘制对比图
    print("绘制单个实例频谱对比图...")
    np.random.seed(config.random_seed)
    random_idx = np.random.randint(0, len(all_predictions))
    
    # 绘制对比图
    plot_spectrum_comparison(
        all_predictions[random_idx], 
        all_targets[random_idx], 
        frequencies, 
        all_inputs[random_idx], 
        config.input_param_names, 
        config
    )
    
    print("推理完成!")

if __name__ == "__main__":
    main()