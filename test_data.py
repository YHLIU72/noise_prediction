# import pandas as pd
# import os

# # 读取原始 Excel（不将任一行作为列名，保留原始行索引）
# file_path = r"E:\lyh\paddlespeech\data_init\BYD HTH-FFT.xlsx"
# sheet_name = pd.ExcelFile(file_path).sheet_names[0]
# df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

# # 检查是否至少有三行
# if df.shape[0] < 3:
# 	raise RuntimeError(f"文件行数不足，无法检查第三行: {file_path}")

# # 第三行索引为2，保留第三行内容等于 'FFT信号' 的列（去除首尾空白）
# third_row = df.iloc[3].astype(str).str.strip()
# mask = third_row == 'FFT信号'

# kept_cols = df.columns[mask.values]
# df_filtered = df.loc[:, kept_cols]

# # 输出文件路径
# base, ext = os.path.splitext(file_path)
# output_path = f"{base}-FFT-filtered{ext}"

# # 将过滤后的表写回Excel，不保留行索引与列名（保持原始结构）
# df_filtered.to_excel(output_path, index=False, header=False)

# print(f"已保存过滤后的文件: {output_path}")
# print(f"保留列数: {len(kept_cols)} / 原始列数: {df.shape[1]}")

# import pandas as pd

# data=pd.read_csv('NU2.csv')
# print(len(data.iloc[10,13]))
# # print(data.iloc[9,13])


import pandas as pd
import re
import os

def process_excel_file(input_file, output_file=None):
    """
    处理Excel文件中第1行的特定格式内容
    
    参数:
    input_file: 输入的Excel文件路径
    output_file: 输出的Excel文件路径（可选）
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)
    
    # 获取列名（第一行）
    columns = df.columns.tolist()
    
    # 只处理第1行（索引0对应第1行数据）
    i = 0  # 第1行的索引
    
    # 遍历第1行的每一列
    for j, column in enumerate(columns):
        cell_value = str(df.iloc[i, j])
        
        # 检查是否为需要处理的格式（HDF、HFF、CVAF开头）
        if any(prefix in cell_value for prefix in ['HDF', 'HFF', 'CVAF']):
            # 使用正则表达式匹配模式：前缀-数字-数字_数字 From 日期 To 日期
            pattern = r'([A-Z]+)-(\d+)-(\d+)'
            match = re.match(pattern, cell_value)
            
            if match:
                prefix = match.group(1)  # 获取前缀（HDF、HFF、CVAF）
                middle_num = match.group(2)  # 中间数字
                last_num_str = match.group(3)  # 最后一部分数字（字符串形式）
                
                # 处理最后一部分数字
                if last_num_str.startswith('0'):
                    # 以0开头：0代表负号，提取后面的数字并转为负数
                    actual_num = -int(last_num_str[1:])
                    # 减1操作
                    new_num = actual_num - 1
                    # 格式化为3位数字，负数用0开头表示
                    if new_num < 0:
                        new_last_num = '0' + str(abs(new_num)).zfill(2)
                    else:
                        new_last_num = str(new_num).zfill(3)
                else:
                    # 不以0开头：直接转为正数并减1
                    actual_num = int(last_num_str)
                    new_num = actual_num - 1
                    new_last_num = str(new_num).zfill(3)
                
                # 重建单元格内容（保持原始时间戳不变）
                time_pattern = r'From (.*) To (.*)'
                time_match = re.search(time_pattern, cell_value)
                
                if time_match:
                    from_time = time_match.group(1)
                    to_time = time_match.group(2)
                    
                    # 重建内容
                    new_cell_value = f'{prefix}-{middle_num}-{new_last_num}_1 From {from_time} To {to_time}'
                    df.iloc[i, j] = new_cell_value
                    
                    print(f'处理完成: {cell_value} -> {new_cell_value}')
    
    # 设置输出文件路径
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f'{filename}_processed{ext}'
    
    # 保存处理后的文件
    df.to_excel(output_file, index=False)
    print(f'文件已保存: {output_file}')
    
    return df

# def main():
#     """主函数"""
#     # 输入文件路径
#     input_file = input('请输入Excel文件路径（或直接拖拽文件到此处）: ').strip().strip('"')
    
#     # 检查文件是否存在
#     if not os.path.exists(input_file):
#         print('文件不存在，请检查路径是否正确')
#         return
    
#     # 处理文件
#     try:
#         result_df = process_excel_file(input_file)
#         print('\n处理完成！')
#         print('前5行处理结果:')
#         print(result_df.head())
        
#     except Exception as e:
#         print(f'处理过程中出现错误: {e}')

# # 如果需要直接运行，取消注释下面的代码
# if __name__ == "__main__":
#     main()

# 也可以直接调用函数处理特定文件
# process_excel_file('你的文件路径.xlsx')

# 也可以直接调用函数处理特定文件
process_excel_file('E:\lyh\paddlespeech\data\T2X RHD-FFT.xlsx','E:\lyh\paddlespeech\data\T2X RHD-FFT_processed.xlsx')


# import numpy as np
# from scipy.interpolate import interp1d

# def spl_to_third_octave_20_10000(a_weighted_spl, ref_pressure=2e-5):
#     """
#     将A计权声压级频谱转换为20Hz-10000Hz范围的1/3倍频程声压级频谱
    
#     参数:
#     a_weighted_spl : array_like
#         A计权声压级值(dB)，长度应为6401（对应0-25600Hz，间隔4Hz）
#     ref_pressure : float, 可选
#         参考声压，默认为20μPa (2e-5 Pa)
    
#     返回:
#     third_octave_centers : ndarray
#         1/3倍频程中心频率(Hz)，范围20Hz-10000Hz
#     third_octave_spl : ndarray
#         对应的1/3倍频程声压级(dB)
#     """
    
#     # 验证输入数据长度（0-25600Hz，间隔4Hz，共6401个点）
#     expected_length = 6401  # (25600/4) + 1
#     if len(a_weighted_spl) != expected_length:
#         raise ValueError(f"输入声压级数组长度应为{expected_length}（0-25600Hz，间隔4Hz）")
    
#     # 创建对应的频率向量（0, 4, 8, ..., 25600 Hz）
#     freq_vector = np.linspace(0, 25600, expected_length)
    
#     # 转换为线性声压值[2](@ref)
#     pressure_linear = ref_pressure * 10**(a_weighted_spl / 20)
    
#     # 完整的1/3倍频程中心频率(Hz)，只保留20Hz-10000Hz范围[1](@ref)
#     all_centers = np.array([
#         1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00,
#         10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
#         100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
#         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
#         10000, 12500, 16000, 20000, 25000
#     ])
    
#     # 筛选20Hz-10000Hz范围内的中心频率[3](@ref)
#     valid_centers = all_centers[(all_centers >= 20) & (all_centers <= 10000)]
    
#     # 预处理数据：去除0Hz点（避免对数计算问题）
#     valid_freq_mask = (freq_vector > 0) & (freq_vector <= freq_vector[-1])
#     valid_freq = freq_vector[valid_freq_mask]
#     valid_pressure = pressure_linear[valid_freq_mask]
    
#     if len(valid_freq) == 0:
#         raise ValueError("没有有效的频率数据可用于处理")
    
#     # 使用对数线性插值提高精度[1](@ref)
#     log_freq = np.log10(valid_freq)
#     log_pressure = np.log10(valid_pressure)  # 避免log(0)
    
#     # 创建插值函数
#     pressure_interp = interp1d(log_freq, log_pressure, kind='linear', 
#                               bounds_error=False, fill_value='extrapolate')
    
#     third_octave_spl = []
    
#     # 计算每个1/3倍频程带的声压级[1,3](@ref)
#     for fc in valid_centers:
#         # 计算频带边界[1](@ref)
#         lower_bound = fc / (2**(1/6))  # 下边界
#         upper_bound = fc * (2**(1/6))  # 上边界
        
#         # 找到当前频带内的频率点
#         band_mask = (valid_freq >= lower_bound) & (valid_freq <= upper_bound)
#         band_freqs = valid_freq[band_mask]
#         band_pressures = valid_pressure[band_mask]
        
#         if len(band_freqs) > 5:  # 确保有足够的数据点进行精确计算
#             # 计算频带内的声压均方值（能量平均）[1](@ref)
#             mean_square_pressure = np.mean(band_pressures**2)
            
#             # 转换为声压级[2](@ref)
#             if mean_square_pressure > 0:
#                 spl = 10 * np.log10(mean_square_pressure / ref_pressure**2)
#             else:
#                 spl = 0  # 无声压时设为0dB
#         else:
#             # 如果数据点不足，使用更精确的插值方法
#             try:
#                 # 在频带内均匀采样多个点进行插值
#                 test_freqs = np.logspace(np.log10(lower_bound), np.log10(upper_bound), 20)
#                 test_log_freqs = np.log10(test_freqs)
#                 test_log_pressures = pressure_interp(test_log_freqs)
#                 test_pressures = 10**test_log_pressures
                
#                 mean_square_pressure = np.mean(test_pressures**2)
                
#                 if mean_square_pressure > 0:
#                     spl = 10 * np.log10(mean_square_pressure / ref_pressure**2)
#                 else:
#                     spl = 0
#             except:
#                 spl = 0
        
#         third_octave_spl.append(spl)
    
#     return third_octave_spl

# # 辅助函数：验证和可视化结果
# def validate_and_plot_third_octave_20_10000(a_weighted_spl):
#     """
#     验证20Hz-10000Hz范围的1/3倍频程转换结果并绘制对比图
#     """
#     import matplotlib.pyplot as plt
    
#     # 计算1/3倍频程结果
#     spl_values = spl_to_third_octave_20_10000(a_weighted_spl)
    
#     # 创建原始频率向量用于绘图
#     freq_vector = np.linspace(0, 25600, len(a_weighted_spl))
    
#     # 绘制结果
#     plt.figure(figsize=(12, 6))
    
#     # 绘制原始A计权频谱（对数频率轴）
#     mask_plot = (freq_vector >= 10) & (freq_vector <= 20000)
#     plt.semilogx(freq_vector[mask_plot], a_weighted_spl[mask_plot], 
#                 'b-', alpha=0.5, label='原始A计权频谱', linewidth=0.8)
    
#     # 绘制1/3倍频程结果
#     plt.semilogx(centers, spl_values, 'ro-', 
#                 label='1/3倍频程声压级 (20Hz-10kHz)', markersize=6, linewidth=2)
    
#     # 标记每个中心频率点
#     for fc, spl in zip(centers, spl_values):
#         plt.annotate(f'{fc:.0f}Hz', (fc, spl), textcoords="offset points", 
#                     xytext=(0,10), ha='center', fontsize=8, alpha=0.7)
    
#     plt.xlabel('频率 (Hz)')
#     plt.ylabel('声压级 (dB)')
#     plt.title('A计权频谱与1/3倍频程声压级对比 (20Hz-10kHz范围)')
#     plt.grid(True, which="both", ls="-", alpha=0.3)
#     plt.legend()
#     plt.xlim(10, 20000)
    
#     # 自动调整Y轴范围
#     valid_spl = spl_values[spl_values > -np.inf]
#     if len(valid_spl) > 0:
#         max_spl = np.max(valid_spl)
#         plt.ylim(max(0, max_spl - 50), max_spl + 5)
    
#     plt.tight_layout()
#     plt.show()
    
#     return centers, spl_values

# # 使用示例
# if __name__ == "__main__":
#     # 示例：创建测试数据（6401个点，对应0-25600Hz，间隔4Hz）
#     a_weighted_spl = np.zeros(6401)
    
#     # 在20Hz-10000Hz范围内设置峰值（模拟实际测量数据）
#     peak_frequencies = [31.5, 100, 250, 500, 1000, 2000, 4000, 8000]
#     peak_values = [55, 60, 58, 65, 70, 68, 65, 60]  # dB
    
#     for freq, value in zip(peak_frequencies, peak_values):
#         # 找到最接近的频率索引
#         idx = int(freq / 4)
#         if idx < len(a_weighted_spl):
#             a_weighted_spl[idx] = value
    
#     # 添加平滑的背景噪声（主要在中低频）
#     for i in range(1, len(a_weighted_spl)-1):
#         if a_weighted_spl[i] == 0:  # 只在没有峰值的地方添加噪声
#             # 频率相关的噪声水平（低频噪声较强）
#             freq_current = i * 4
#             if freq_current <= 1000:
#                 noise_level = 25 + 10 * np.random.random()
#             else:
#                 noise_level = 20 + 5 * np.random.random()
            
#             a_weighted_spl[i] = noise_level
    
#     # 确保最小值合理
#     a_weighted_spl = np.maximum(0, a_weighted_spl)
    
#     # 应用函数
#     centers, spl_values = spl_to_third_octave_20_10000(a_weighted_spl)
    
#     # 打印结果摘要
#     print("1/3倍频程声压级计算结果摘要 (20Hz-10kHz):")
#     print(f"处理频带数量: {len(centers)}")
#     print(f"频率范围: {centers[0]:.1f} - {centers[-1]:.1f} Hz")
#     print(f"声压级范围: {np.min(spl_values):.1f} - {np.max(spl_values):.1f} dB")
    
#     print("\n详细结果:")
#     print("中心频率(Hz)\t下边界(Hz)\t上边界(Hz)\t声压级(dB)")
#     for i, fc in enumerate(centers):
#         lower = fc / (2**(1/6))
#         upper = fc * (2**(1/6))
#         print(f"{fc:8.1f}\t{lower:8.1f}\t{upper:8.1f}\t{spl_values[i]:8.2f}")
    
#     # 验证和绘图
#     validate_and_plot_third_octave_20_10000(a_weighted_spl)