import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import os

# 设置字体支持
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 配置参数
class ComparisonConfig:
    def __init__(self):
        # 模型路径
        self.original_model_path = "runs/spl/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260201_203236/checkpoint/best_model_epoch_95.pth"  # 原始模型路径
        self.copy_model_path = "runs/spl/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260203_150442/checkpoint/best_model_epoch_100.pth"  # 对比模型路径

        
        # 数据集配置
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = 42
        self.val_split = 0.2
        
        # 工作模式标签（使用英文）
        self.mode_labels = ["CVAF", "CVAR", "HFF", "HDF"]
        
        # 图表保存路径
        self.save_dir = "comparison_results"
        os.makedirs(self.save_dir, exist_ok=True)

# 加载模型和数据的通用函数
def load_model_and_data():
    config = ComparisonConfig()
    
    # 加载原始模型和数据集
    print("加载原始模型和数据集...")
    from noisedata import SPLdata as OriginalSPLdata
    from soundmodel import SPLModel as OriginalSPLModel
    
    original_train_dataset = OriginalSPLdata(
        directory_path="../csvdata333",
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    original_val_dataset = original_train_dataset.get_validation_dataset()
    
    original_val_loader = DataLoader(
        original_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    original_model = OriginalSPLModel(nc=800).to(config.device)
    checkpoint = torch.load(config.original_model_path, map_location=config.device)
    original_model.load_state_dict(checkpoint["model_state_dict"])
    original_model.eval()
    
    # 加载对比模型和数据集
    print("加载对比模型和数据集...")
    from noisedata_copy import SPLdata as CopySPLdata
    from soundmodel_copy import SPLModel as CopySPLModel
    
    copy_train_dataset = CopySPLdata(
        directory_path="../csvdata333",
        val_split=config.val_split,
        random_seed=config.random_seed
    )
    copy_val_dataset = copy_train_dataset.get_validation_dataset()
    
    copy_val_loader = DataLoader(
        copy_val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False
    )
    
    copy_model = CopySPLModel(in_nc=5, nc=800).to(config.device)
    checkpoint = torch.load(config.copy_model_path, map_location=config.device)
    copy_model.load_state_dict(checkpoint["model_state_dict"])
    copy_model.eval()
    
    return config, original_model, original_val_loader, copy_model, copy_val_loader

# 执行原始模型的推理
def infer_original_model(model, val_loader, device):
    all_predictions = []
    all_targets = []
    all_mode_ids = []
    
    with torch.no_grad():
        for inputs, targets, type_ids, mode_ids in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            type_ids = type_ids.to(device)
            mode_ids = mode_ids.to(device)
            
            # 前向传播：多分枝输出
            outputs = model(inputs)
            batch_size = torch.arange(inputs.size(0), device=device)
            outputs = outputs[batch_size, mode_ids, type_ids, :].squeeze()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_mode_ids.append(mode_ids.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_mode_ids = np.concatenate(all_mode_ids, axis=0)
    
    return all_predictions, all_targets, all_mode_ids

# 执行对比模型的推理
def infer_copy_model(model, val_loader, device):
    all_predictions = []
    all_targets = []
    all_mode_ids = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 压缩输出维度，从(batch, 1)变为(batch)
            outputs = outputs.squeeze()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # 对比模型的数据集可能没有mode_ids，需要根据实际情况调整
            all_mode_ids.append(np.zeros_like(targets.cpu().numpy()))  # 临时填充，需要根据实际情况修改
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_mode_ids = np.concatenate(all_mode_ids, axis=0)
    
    return all_predictions, all_targets, all_mode_ids

# 计算MAE
def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

# 按工作模式计算MAE
def calculate_mae_by_mode(predictions, targets, mode_ids, num_modes=4):
    mae_by_mode = []
    for mode in range(num_modes):
        mode_mask = (mode_ids == mode)
        if np.sum(mode_mask) > 0:
            mode_mae = calculate_mae(predictions[mode_mask], targets[mode_mask])
            mae_by_mode.append(mode_mae)
        else:
            mae_by_mode.append(0.0)
    return mae_by_mode

# 绘制总声压级预测值与实测值对比散点图（对角线形式）
def plot_spl_scatter(predictions, targets, model_name, config):
    plt.figure(figsize=(8, 8))
    
    # 绘制散点图
    plt.scatter(targets, predictions, alpha=0.5, s=80, color='blue', label='Predictions')
    
    # 绘制对角线
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Ideal')
    
    # 设置图表属性
    plt.xlabel('Measured SPL (dBA)', fontsize=40, fontweight='bold')
    plt.ylabel('Predicted SPL (dBA)', fontsize=40, fontweight='bold')
    # plt.title(f'Total Sound Pressure Level Prediction vs Measurement\n{model_name}', fontsize=20, fontweight='bold')
    plt.legend(fontsize=32, loc='upper left')
    plt.grid(True, alpha=0.3, linewidth=1)
    
    # 加大坐标轴数字刻度
    plt.tick_params(axis='both', which='major', labelsize=40, width=2.5, length=12)
    
    # 设置图框加粗描黑
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # 保存图表
    save_path = os.path.join(config.save_dir, f'spl_scatter_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SPL对比散点图已保存至: {save_path}")
    
    plt.close()

# 绘制残差分布图
def plot_residual_distribution(predictions, targets, model_name, config):
    residuals = predictions - targets
    
    plt.figure(figsize=(8, 8))
    
    # 绘制直方图
    plt.hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=2)
    
    # 绘制零线
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Residual')
    
    # 设置图表属性
    plt.xlabel('Residual(Predicted-Measured)', fontsize=40, fontweight='bold')
    plt.ylabel('Frequency', fontsize=40, fontweight='bold')
    # plt.title(f'Residual Distribution\n{model_name}', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.3, linewidth=1)
    
    # 加大坐标轴数字刻度
    plt.tick_params(axis='both', which='major', labelsize=32, width=2.5, length=12)
    
    # 设置图框加粗描黑
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # 添加统计信息 - 调整位置到左上角，避免与图例重叠
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.02, 0.95, f'Mean: {mean_residual:.2f} dBA\nStd: {std_residual:.2f} dBA', 
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round', alpha=0.8, facecolor='white', edgecolor='black', linewidth=1.5))
    
    # 调整图例位置到右上角，避免与统计信息重叠
    plt.legend(fontsize=18, loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.8)
    
    # 保存图表
    save_path = os.path.join(config.save_dir, f'residual_distribution_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"残差分布图已保存至: {save_path}")
    
    plt.close()

# 绘制不同工作模式下原始模型的MAE柱状图
def plot_mae_by_mode(mae_by_mode, config):
    plt.figure(figsize=(12, 7))  # 稍微增加图框高度
    
    # 设置柱状图参数
    x = np.arange(len(config.mode_labels))
    width = 0.4  # 调细柱子宽度
    
    # 绘制柱状图
    bars = plt.bar(x, mae_by_mode, width, label='Original Model', color='green', edgecolor='black', linewidth=2)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                    f'{height:.2f}', ha='center', va='bottom', fontsize=32, fontweight='bold')
    
    add_labels(bars)
    
    # 设置图表属性
    plt.xlabel('Working Mode', fontsize=40, fontweight='bold')
    plt.ylabel('MAE(dBA)', fontsize=40, fontweight='bold')
    # plt.title('MAE of Original Model Across Different Working Modes', fontsize=20, fontweight='bold')
    plt.xticks(x, config.mode_labels, fontsize=32, fontweight='bold', rotation=0)
    plt.yticks(fontsize=32, fontweight='bold')
    # plt.grid(True, axis='y', alpha=0.3, linewidth=1)
    
    # 调大y轴数值范围
    max_mae = max(mae_by_mode)
    plt.ylim(0, max_mae * 1.3)  # 设置y轴上限为最大值的1.3倍，调大数值范围
    

    # 设置图框加粗描黑
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    
    # 保存图表
    save_path = os.path.join(config.save_dir, 'mae_original_by_mode.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MAE柱状图已保存至: {save_path}")
    
    plt.close()

# 主函数
def main():
    # 加载模型和数据
    config, original_model, original_val_loader, copy_model, copy_val_loader = load_model_and_data()
    
    # 执行推理
    print("执行原始模型推理...")
    original_predictions, original_targets, original_mode_ids = infer_original_model(original_model, original_val_loader, config.device)
    
    print("执行对比模型推理...")
    copy_predictions, copy_targets, copy_mode_ids = infer_copy_model(copy_model, copy_val_loader, config.device)
    
    # 计算整体MAE
    original_mae = calculate_mae(original_predictions, original_targets)
    copy_mae = calculate_mae(copy_predictions, copy_targets)
    
    print(f"原始模型整体MAE: {original_mae:.4f} dB")
    print(f"对比模型整体MAE: {copy_mae:.4f} dB")
    
    # 按工作模式计算MAE（仅原始模型）
    original_mae_by_mode = calculate_mae_by_mode(original_predictions, original_targets, original_mode_ids)
    
    # 绘制对比散点图
    plot_spl_scatter(original_predictions, original_targets, "Original Model", config)
    plot_spl_scatter(copy_predictions, copy_targets, "Comparison Model", config)
    
    # 绘制残差分布图
    plot_residual_distribution(original_predictions, original_targets, "Original Model", config)
    plot_residual_distribution(copy_predictions, copy_targets, "Comparison Model", config)
    
    # 绘制原始模型的MAE柱状图
    plot_mae_by_mode(original_mae_by_mode, config)
    
    print("\n所有图表已生成并保存至文件夹:", config.save_dir)

if __name__ == "__main__":
    main()