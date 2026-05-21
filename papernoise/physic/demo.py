import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIMBCN_data import PIMBCNDataset

# 设置全局字体为Times New Roman（用于论文插图）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2

def plot_oaspl_vs_rpm(dataset, mode_idx, type_idx, save_path=None):
    """
    绘制同一工况下总声压级随转速变化的曲线（使用未归一化的转速）
    
    参数:
        dataset: PIMBCNDataset实例
        mode_idx: 模式索引
        type_idx: 类型索引
        save_path: 保存路径，为None时不保存
    """
    # 获取同一工况下的所有数据（使用原始数据而非归一化数据）
    rpm_values = []
    oaspl_values = []
    
    # 遍历数据集原始数据
    for i in range(len(dataset.data)):
        data_type = dataset.data.iloc[i, dataset.type_col]
        data_mode = dataset.data.iloc[i, dataset.mode_col]
        
        if data_type == type_idx and data_mode == mode_idx:
            # 获取未归一化的原始输入特征
            input_sample = dataset.data.iloc[i, dataset.input_cols].values.astype(float)
            rpm_raw = input_sample[2]  # RPM是第3个输入特征（索引为2）
            rpm_values.append(rpm_raw)
            
            # 获取OASPL
            oaspl = dataset.data.iloc[i, dataset.oaspl_col]
            oaspl_values.append(oaspl)
    
    # 按转速排序
    sorted_indices = np.argsort(rpm_values)
    rpm_values = np.array(rpm_values)[sorted_indices]
    oaspl_values = np.array(oaspl_values)[sorted_indices]
    
    # 创建图（宽度14厘米，高度8厘米）
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))  # 转换为英寸
    
    # 绘制曲线（增加线条宽度和标记大小）
    ax.plot(rpm_values, oaspl_values, 'b-', linewidth=2.5, marker='o', markersize=6, markeredgewidth=1.5, markeredgecolor='blue', label='OASPL')
    
    # 设置标题和标签
    # ax.set_title(f'OASPL vs RPM (Mode: {mode_idx}, Type: {type_idx})', fontsize=14)
    ax.set_xlabel('RPM', fontsize=12)
    ax.set_ylabel('OASPL (dBA)', fontsize=12)
    
    # 添加网格和图例
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 设置边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 设置刻度线宽度
    ax.tick_params(width=1.5, length=6)
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图片（高分辨率用于论文）
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        print(f"OASPL图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_octave_spectrum(octave_data, save_path=None):
    """
    绘制三分之一倍频程频谱图（中心频率从20到10000 Hz）
    
    参数:
        octave_data: 28维倍频程数据（张量或数组）
        save_path: 保存路径，为None时不保存
    """
    # 三分之一倍频程中心频率（从20Hz到10000Hz，共28个）
    octave_freqs = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
        6300, 8000, 10000
    ]
    
    # 转换为numpy数组
    if isinstance(octave_data, torch.Tensor):
        octave_data = octave_data.numpy()
    
    # 创建图
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))
    
    # 绘制柱状图（增加边框宽度）
    ax.bar(np.arange(len(octave_freqs)), octave_data, width=0.75, color='darkorange', edgecolor='black', linewidth=1.2)
    
    # 设置x轴标签
    ax.set_xticks(np.arange(len(octave_freqs)))
    ax.set_xticklabels([str(f) for f in octave_freqs], rotation=45, ha='right', fontsize=10)
    
    # 设置标题和标签
    # ax.set_title('1/3 Octave Band Spectrum', fontsize=14)
    ax.set_xlabel('Center Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Sound Pressure Level (dBA)', fontsize=12)
    
    # 添加网格
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 设置边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 设置刻度线宽度
    ax.tick_params(width=1.5, length=6)
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图片（高分辨率用于论文）
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        print(f"倍频程频谱图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_narrowband_spectrum(spectrum_data, save_path=None):
    """
    绘制窄带声压级曲线图（频率从0到10000 Hz，间隔4 Hz）
    
    参数:
        spectrum_data: 2501维频谱数据（张量或数组）
        save_path: 保存路径，为None时不保存
    """
    # 频率范围（0-10000 Hz，共2501个点，间隔4 Hz）
    freq_range = np.arange(0, 10001, 4)  # 0, 4, 8, ..., 10000
    
    # 确保频率点数与数据点数匹配
    if len(freq_range) != len(spectrum_data):
        freq_range = np.linspace(0, 10000, len(spectrum_data))
    
    # 转换为numpy数组
    if isinstance(spectrum_data, torch.Tensor):
        spectrum_data = spectrum_data.numpy()
    
    # 创建图
    fig, ax = plt.subplots(figsize=(14/2.54, 8/2.54))
    
    # 绘制曲线（增加线条宽度）
    ax.plot(freq_range, spectrum_data, 'g-', linewidth=2.0)
    
    # 设置标题和标签
    # ax.set_title('Narrowband Sound Pressure Level Spectrum', fontsize=14)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Sound Pressure Level (dBA)', fontsize=12)
    
    # 添加网格
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴范围
    ax.set_xlim(0, 10000)
    
    # 调整布局
    plt.tight_layout()
    
    # 设置边框宽度
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 设置刻度线宽度
    ax.tick_params(width=1.5, length=6)
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图片（高分辨率用于论文）
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        print(f"窄带频谱图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_for_working_condition(dataset, mode_idx, type_idx, output_dir='./plots'):
    """
    为指定工况绘制所有三张图
    
    参数:
        dataset: PIMBCNDataset实例
        mode_idx: 模式索引
        type_idx: 类型索引
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 绘制OASPL随转速变化图（使用未归一化的转速）
    oaspl_path = os.path.join(output_dir, f'oaspl_rpm_mode{mode_idx}_type{type_idx}.png')
    plot_oaspl_vs_rpm(dataset, mode_idx, type_idx, oaspl_path)
    
    # 2. 获取第一个该工况的数据点绘制倍频程和窄带频谱
    for i in range(len(dataset.data)):
        data_type = dataset.data.iloc[i, dataset.type_col]
        data_mode = dataset.data.iloc[i, dataset.mode_col]
        
        if data_type == type_idx and data_mode == mode_idx:
            # 获取数据（注意：dataset[i]返回的是归一化后的数据，但我们需要原始数据）
            # 直接从原始数据获取
            oaspl = dataset.data.iloc[i, dataset.oaspl_col]
            
            octave_sample = dataset.data.iloc[i, dataset.octave_col]
            import ast
            octave_list = ast.literal_eval(octave_sample) if isinstance(octave_sample, str) else octave_sample
            octave = torch.tensor(octave_list, dtype=torch.float32)
            
            spectrum_sample = dataset.data.iloc[i, dataset.spectrum_col]
            spectrum_list = ast.literal_eval(spectrum_sample) if isinstance(spectrum_sample, str) else spectrum_sample
            # 补齐或截断到2501点
            if len(spectrum_list) > 2501:
                spectrum_list = spectrum_list[:2501]
            elif len(spectrum_list) < 2501:
                spectrum_list = spectrum_list + [0.0] * (2501 - len(spectrum_list))
            spectrum = torch.tensor(spectrum_list, dtype=torch.float32)
            
            # 绘制倍频程频谱
            octave_path = os.path.join(output_dir, f'octave_mode{mode_idx}_type{type_idx}.png')
            plot_octave_spectrum(octave, octave_path)
            
            # 绘制窄带频谱
            spectrum_path = os.path.join(output_dir, f'spectrum_mode{mode_idx}_type{type_idx}.png')
            plot_narrowband_spectrum(spectrum, spectrum_path)
            
            print(f"工况 (Mode: {mode_idx}, Type: {type_idx}) 的三张图已全部保存")
            break

if __name__ == "__main__":
    # 示例用法
    data_directory = "E:\\lyh\\paddlespeech\\csvdata333"
    
    # 创建数据集
    dataset = PIMBCNDataset(
        directory_path=data_directory,
        input_cols=[4, 5, 6],
        oaspl_col=11, octave_col=12, spectrum_col=13,
        type_col=3, mode_col=2,
        val_split=0.0,  # 使用全部数据
        is_validation=False
    )
    
    # 绘制指定工况的三张图 (示例: mode=0, type=0)
    plot_all_for_working_condition(dataset, mode_idx=0, type_idx=0, output_dir='./plots')
    
    print("所有图片绘制完成！")