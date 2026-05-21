import pandas as pd
import os
import glob
import wave
import re
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.signal import hilbert,welch
from DP_data import process_excel_polynomial_fit,predict_pressure

def read_excel_file(file_path, sheet_name=None):
    """
    读取Excel文件并返回DataFrame和对应的工作表名称
    
    参数:
        file_path (str): Excel文件路径
        sheet_name (str, optional): 要读取的工作表名称，默认为None（读取第一个工作表）
    
    返回:
        tuple: (DataFrame, str) - 包含数据的DataFrame和对应的工作表名称
    """
    # 创建ExcelFile对象以获取工作表信息
    xls = pd.ExcelFile(file_path)
    
    # 如果未指定工作表名称，则使用第一个工作表
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    
    # 读取指定工作表数据
    df = pd.read_excel(xls, sheet_name=sheet_name,skiprows=24)
    
    # 返回DataFrame和工作表名称
    return df, sheet_name

def get_all_sheet_names(file_path):
    """获取Excel文件中所有工作表的名称"""
    xls = pd.ExcelFile(file_path)
    return xls.sheet_names

def clean_and_save_to_excel(df, column_name, output_path='cleaned_data.xlsx'):
    """
    清理DataFrame并保存为Excel文件，删除指定列中包含NaN或0的行
    
    参数:
        df (pd.DataFrame): 输入的DataFrame
        column_name (str): 需要检查的列名
        output_path (str, optional): 输出Excel文件路径，默认为'cleaned_data.xlsx'
    """
    # 过滤掉指定列中为NaN或0的行
    df_clean = df[(df[column_name].notna()) & (df[column_name] != 0)]
    # 保存清理后的数据到Excel文件
    df_clean.to_excel(output_path, index=False)

def extract_scientific_data(match_info, excel_file_path):
    """
    从Excel文件中提取符合特定条件的列数据
    
    参数:
    match_info: 需要匹配的特定字符串（如"CVAF-83-17"）
    excel_file_path: Excel文件路径
    
    返回:
    list: 包含第11行开始所有科学计数法数值的列表，保持完整精度
    """
    try:
        # 读取Excel文件的第一个工作表，第一行作为列名
        df = pd.read_excel(excel_file_path, sheet_name=0, header=0)
        
        target_column = None
        
        # 遍历所有列名，查找与match_info完全匹配的列
        for col_name in df.columns:
            pattern = r'(\w+-\d+-\d+)'
            match = re.search(pattern, col_name)
            # 仅当正则匹配到目标格式时才继续检查，避免未定义变量
            if match:
                extracted_text = match.group(1)
                # 将列名转换为字符串并检查是否与match_info完全一致
                if extracted_text == match_info:
                    # 检查该列第一行(索引0)的值是否为'mc1'
                    if str(df.iloc[0][col_name]).strip().lower() == 'mc1':
                        target_column = col_name
                        break
        
        if target_column is None:
            raise ValueError(f"未找到列名与'{match_info}'完全匹配且第一行值为'mc1'的列")
        
        # 提取 Excel 中第11行（即第一个数据行为 header 时的第11行）开始的所有值，包含第11行
        # 说明：pd.read_excel(header=0) 会把第一行作为列名，DataFrame 的第0行对应 Excel 的第2行
        # 因此 Excel 的第11行对应 DataFrame 的第10行（索引10）
        series = df.iloc[10:][target_column]
        # 将科学计数法字符串转换为数值（不可解析的项变为 NaN），保持顺序
        values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)

        # 确保返回长度为 3201 (0..25600 @ 8Hz => 25600/8 + 1)
        expected_len = int(25600 // 8) + 1  # 3201
        if len(values) < expected_len:
            # 用 0.0 填充到预期长度
            pad = np.zeros(expected_len - len(values), dtype=np.float64)
            values = np.concatenate([values, pad])
        elif len(values) > expected_len:
            # 截断多余点，保证上游始终收到固定长度
            values = values[:expected_len]

        return values.tolist()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件: {excel_file_path}")
    except Exception as e:
        raise RuntimeError(f"数据处理错误: {str(e)}")
    
def extract_spl_data(match_field: str, excel_path: str):
    """
    从Excel文件中提取声压级频谱数据
    
    参数:
        match_field (str): 要匹配的字段名
        excel_path (str): Excel文件路径
        
    返回:
        Union[List[float], str]: 匹配列的6401个声压级数据列表，或错误信息字符串
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, header=0)
        
        # 检查数据行数是否足够
        if len(df) < 1:
            raise ValueError("错误：Excel文件中没有数据行")
        
        # 正则表达式模式
        pattern = r'(\w+-\d+-\d+)'
        
        # 遍历所有列名进行匹配
        matched_column = None
        for col_name in df.columns:
            if pd.isna(col_name):
                continue
                
            col_name_str = str(col_name)
            match_result = re.match(pattern, col_name_str)
            
            if match_result:
                extracted_field = match_result.group(1)
                if extracted_field == match_field:
                    matched_column = col_name
                    break
        
        if matched_column is None:
            raise ValueError(f"错误：未找到与字段 '{match_field}' 匹配的列")
        
        # 关键修复：正确提取整列数据
        spl_data = df[matched_column].tolist()
        
        # 安全的数据类型转换和处理
        # spl_data = []
        

        # 情况2：多行数据，每行一个值
        # spl_data = pd.to_numeric(data_series, errors='coerce').tolist()
        
        # # 过滤掉NaN值
        # spl_data = [x for x in spl_data if not np.isnan(x)]
        
        # 验证数据长度
        expected_length = 6401
        
        if len(spl_data) < expected_length:
            raise ValueError(f"错误：数据不足，期望{expected_length}个值，实际得到{len(spl_data)}个有效值")
        elif len(spl_data) > expected_length:
            # 如果数据多于期望值，取前6401个
            spl_data = spl_data[:expected_length]
        
        # 最终验证
        if len(spl_data) != expected_length:
            raise ValueError(f"错误：有效数据长度不符，期望{expected_length}个值，实际得到{len(spl_data)}个值")
        
        return spl_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：未找到指定的Excel文件 - {excel_path}")
    except Exception as e:
        raise RuntimeError(f"错误：处理过程中发生异常 - {str(e)}")
# def calculate_a_weighted_spectra_high_precision(power_spectrum_8hz, fs=25600, max_freq=10000):
#     """
#     高精度计算A计权声压级频谱和三分之一倍频程谱
#     专门针对微小声压值优化，避免精度损失
    
#     参数:
#     power_spectrum_8hz: 自功率谱数据 (Pa²), 长度3201, 0-25600Hz @8Hz间隔
#     fs: 采样频率 (默认25600Hz)
#     max_freq: 最大分析频率 (默认10000Hz)
    
#     返回:
#     tuple: (linear_spectrum_df, third_octave_df)
#     """
#     # 使用 float64 实现并补全 A 计权 (IEC 61672)
#     # 将输入转换为数值，非数值项转换为 NaN，以保证后续计算稳健
#     power_spectrum = pd.to_numeric(pd.Series(power_spectrum_8hz), errors='coerce').to_numpy(dtype=np.float64)
#     n_points = len(power_spectrum)

#     # 期望的点数：8 Hz 分辨率 => n = fs/8 + 1 = 3201
#     expected_n_points = int(fs // 8) + 1

#     # 不允许在此函数中执行插值或合并，强制上游保证传入长度为 3201
#     if n_points != expected_n_points:
#         raise RuntimeError(f"输入频谱点数不为 {expected_n_points} (8Hz 分辨率)，当前为 {n_points}，请确保 extract_scientific_data 返回长度为 {expected_n_points}。")

#     # 频率轴：从0到fs，包含 n_points 个点
#     freqs = np.linspace(0, fs, n_points, dtype=np.float64)

#     # 限制频率范围
#     idx_10k = np.where(freqs <= max_freq)[0]
#     freqs_10k = freqs[idx_10k]
#     power_spectrum_10k = power_spectrum[idx_10k]

#     # A计权函数（向量化实现，返回 dB 修正值，1kHz 归一化为 0 dB）
#     def a_weighting_db(f):
#         f = np.asarray(f, dtype=np.float64)
#         # 避免除零
#         f_safe = np.where(f == 0, 1e-20, f)
#         f2 = f_safe ** 2
#         # IEC 61672 A-weighting
#         num = (12194.0 ** 2) * (f2 ** 2)
#         den = (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)
#         ra = num / den
#         a = 20.0 * np.log10(ra) + 2.0
#         # 归一化，使 1 kHz 对应 0 dB
#         a -= (20.0 * np.log10(((12194.0 ** 2) * (1000.0 ** 4)) / ((1000.0 ** 2 + 20.6 ** 2) * np.sqrt((1000.0 ** 2 + 107.7 ** 2) * (1000.0 ** 2 + 737.9 ** 2)) * (1000.0 ** 2 + 12194.0 ** 2))) + 2.0)
#         return a

#     # 线性频谱（每个频点的 A 计权 SPL）
#     # 注意：输入的 power_spectrum_8hz 单位为 Pa^2（每个频点的能量），非 PSD
#     reference_pressure = 20e-6
#     a_weights = a_weighting_db(freqs_10k)
#     spl_linear = []

#     for p, aw in zip(power_spectrum_10k, a_weights):
#         if np.isnan(p) or p <= 0:
#             spl = -120.0
#         else:
#             # 输入为每点能量 (Pa^2)，直接开方得到有效声压
#             p_rms = np.sqrt(p)
#             spl = 20.0 * np.log10(p_rms / reference_pressure)
#         spl_linear.append(spl + aw)

#     # 计算三分之一倍频程（中心频率生成规则与原函数保持一致）
#     def calculate_third_octave():
#         center_freqs = []
#         n = -16
#         while True:
#             f_center = 1000.0 * (2.0 ** (n / 3.0))
#             if f_center > max_freq * 1.1:
#                 break
#             if f_center >= 20.0:
#                 center_freqs.append(f_center)
#             n += 1

#         spl_third = []
#         valid_centers = []
#         for f_center in center_freqs:
#             lower = f_center / (2.0 ** (1.0 / 6.0))
#             upper = f_center * (2.0 ** (1.0 / 6.0))
#             idx_band = np.where((freqs_10k >= lower) & (freqs_10k <= upper))[0]
#             if len(idx_band) == 0:
#                 continue
#             # 输入为每点能量 (Pa^2)，将频带内每点能量求和得到频带总能量
#             band_power = np.sum(power_spectrum_10k[idx_band])
#             if band_power <= 0 or np.isnan(band_power):
#                 band_spl = -120.0
#             else:
#                 p_rms = np.sqrt(band_power)
#                 band_spl = 20.0 * np.log10(p_rms / reference_pressure)
#             a_w = a_weighting_db([f_center])[0]
#             spl_third.append(band_spl + a_w)
#             valid_centers.append(f_center)

#         return valid_centers, spl_third

#     center_freqs, spl_third_octave = calculate_third_octave()

#     return spl_linear, spl_third_octave

from scipy.interpolate import interp1d
def spl_to_third_octave_20_10000(a_weighted_spl, ref_pressure=2e-5):
    """
    将A计权声压级频谱转换为20Hz-10000Hz范围的1/3倍频程声压级频谱
    
    参数:
    a_weighted_spl : array_like
        A计权声压级值(dB)，长度应为6401（对应0-25600Hz，间隔4Hz）
    ref_pressure : float, 可选
        参考声压，默认为20μPa (2e-5 Pa)
    
    返回:
    third_octave_centers : ndarray
        1/3倍频程中心频率(Hz)，范围20Hz-10000Hz
    third_octave_spl : ndarray
        对应的1/3倍频程声压级(dB)
    """
    
    # 验证输入数据长度（0-25600Hz，间隔4Hz，共6401个点）并转换为 numpy 数组
    expected_length = 6401  # (25600/4) + 1
    a_weighted_spl = np.asarray(a_weighted_spl, dtype=np.float64)
    if a_weighted_spl.size != expected_length:
        raise ValueError(f"输入声压级数组长度应为{expected_length}（0-25600Hz，间隔4Hz），当前为 {a_weighted_spl.size}")

    # 创建对应的频率向量（0, 4, 8, ..., 25600 Hz）
    freq_vector = np.linspace(0, 25600, expected_length)

    # 转换为线性声压值
    # 避免对非数值或 -inf 引入 NaN，先替换无效值
    with np.errstate(invalid='ignore'):
        pressure_linear = ref_pressure * 10.0 ** (a_weighted_spl / 20.0)
    # 将不可用值替换为 0
    pressure_linear = np.nan_to_num(pressure_linear, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 完整的1/3倍频程中心频率(Hz)，只保留20Hz-10000Hz范围[1](@ref)
    all_centers = np.array([
        1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00,
        10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0,
        100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
        1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500, 16000, 20000, 25000
    ])
    
    # 筛选20Hz-10000Hz范围内的中心频率[3](@ref)
    valid_centers = all_centers[(all_centers >= 20) & (all_centers <= 10000)]
    
    # 预处理数据：去除0Hz点（避免对数计算问题）
    valid_freq_mask = (freq_vector > 0) & (freq_vector <= freq_vector[-1])
    valid_freq = freq_vector[valid_freq_mask]
    valid_pressure = pressure_linear[valid_freq_mask]
    
    if len(valid_freq) == 0:
        raise ValueError("没有有效的频率数据可用于处理")
    
    # 使用对数线性插值提高精度[1](@ref)
    log_freq = np.log10(valid_freq)
    log_pressure = np.log10(valid_pressure)  # 避免log(0)
    
    # 创建插值函数
    pressure_interp = interp1d(log_freq, log_pressure, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
    
    third_octave_spl = []
    
    # 计算每个1/3倍频程带的声压级[1,3](@ref)
    for fc in valid_centers:
        # 计算频带边界[1](@ref)
        lower_bound = fc / (2**(1/6))  # 下边界
        upper_bound = fc * (2**(1/6))  # 上边界
        
        # 找到当前频带内的频率点
        band_mask = (valid_freq >= lower_bound) & (valid_freq <= upper_bound)
        band_freqs = valid_freq[band_mask]
        band_pressures = valid_pressure[band_mask]
        
        if len(band_freqs) > 5:  # 确保有足够的数据点进行精确计算
            # 计算频带内的声压均方值（能量平均）[1](@ref)
            mean_square_pressure = np.mean(band_pressures**2)
            
            # 转换为声压级[2](@ref)
            if mean_square_pressure > 0:
                spl = 10 * np.log10(mean_square_pressure / ref_pressure**2)
            else:
                spl = 0  # 无声压时设为0dB
        else:
            # 如果数据点不足，使用更精确的插值方法
            try:
                # 在频带内均匀采样多个点进行插值
                test_freqs = np.logspace(np.log10(lower_bound), np.log10(upper_bound), 20)
                test_log_freqs = np.log10(test_freqs)
                test_log_pressures = pressure_interp(test_log_freqs)
                test_pressures = 10**test_log_pressures
                
                mean_square_pressure = np.mean(test_pressures**2)
                
                if mean_square_pressure > 0:
                    spl = 10 * np.log10(mean_square_pressure / ref_pressure**2)
                else:
                    spl = 0
            except:
                spl = 0
        
        third_octave_spl.append(spl)
    
    return third_octave_spl

def find_specific_columns_by_regex(match_info, excel_file_path, target_columns):
    """
    在Excel文件的第二列中通过正则表达式匹配行，并返回该行指定列的值
    
    参数:
    match_info: 匹配信息（用于正则表达式匹配）
    excel_file_path: Excel文件路径
    target_columns: 需要返回的指定列，可以是列名列表或列索引列表
    
    返回:
    dict: 包含指定列名和对应值的字典，保持原始精度
    """
    try:
        # 读取Excel文件的第一个工作表
        df = pd.read_excel(excel_file_path, sheet_name=0, header=0)
        
        # 检查数据框是否为空
        if df.empty:
            raise ValueError("Excel文件为空或无法读取数据")
        
        # 确保有第二列（索引为1的列）
        if df.shape[1] < 2:
            raise ValueError("Excel文件列数不足，至少需要2列")
        
        # 定义正则表达式模式
        pattern = r'(\w+-\d+-\d+)'
        target_row_index = None
        
        # 遍历第二列（索引为1）的每一行进行正则匹配
        for index, value in enumerate(df.iloc[:, 1]):  # 第二列索引为1
            if pd.notna(value):  # 确保值不为空
                str_value = str(value)
                match = re.match(pattern, str_value)
                if match and match.group(1) == match_info:
                    target_row_index = index
                    break
        
        if target_row_index is None:
            raise ValueError(f"未找到与'{match_info}'匹配的行")
        
        # 处理目标列参数
        if isinstance(target_columns, list):
            # 如果输入的是列名列表，检查这些列是否存在
            missing_columns = [col for col in target_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"以下列名不存在于数据中: {missing_columns}")
            
            # 提取指定列的数据，保持完整精度
            result = {}
            for col in target_columns:
                cell_value = df.iloc[target_row_index][col]
                # 使用str()确保数值完整转换，避免科学计数法截断
                result[col] = str(cell_value)
            
            return result
        
        elif isinstance(target_columns, int):
            # 如果输入的是单个列索引
            if target_columns >= df.shape[1] or target_columns < 0:
                raise ValueError(f"列索引{target_columns}超出范围")
            
            col_name = df.columns[target_columns]
            cell_value = df.iloc[target_row_index][col_name]
            return {col_name: str(cell_value)}
        
        else:
            raise ValueError("target_columns参数应为列名列表或列索引")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件: {excel_file_path}")
    except Exception as e:
        raise RuntimeError(f"数据处理错误: {str(e)}")

def create_wav_info_csv(input_folder, file_type, excel_path, output_csv_path,sheet_name=None, fit_results=None, hvac_parameter=None):
    """
    根据WAV文件和Excel数据创建包含匹配信息的CSV文件
    
    参数:
        input_folder (str): 包含WAV文件的文件夹路径
        file_type (str): 文件类型过滤（此处应为'wav'）
        excel_path (str): Excel数据文件路径
        output_csv_path (str): 输出CSV文件路径
    """
    # 1. 读取Excel所有工作表数据并合并（确保能搜索全部数据）
    xls = pd.ExcelFile(excel_path)
    excel_df = pd.concat(
        [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names],
        ignore_index=True
    )
    
    # 2. 收集所有WAV文件路径和名称
    wav_files = glob.glob(os.path.join(input_folder, f'*.{file_type}'))
    csv_data = []  # 存储CSV行数据
    
    for wav_path in wav_files:
        wav_name = os.path.basename(wav_path)
        
        # 3. 解析WAV文件名（按'-'分割并取前3部分）
        parts = wav_name.split('-')[:3]
        if len(parts) < 3:
            print(f"跳过无效文件名: {wav_name}（分割后不足3部分）")
            continue
        
        model, qv_str, dp_str = parts
        if model=="CVAF"or model=="CAVF":
            model="CVAF"
        if model=="CVAR"or model=="CAVR":
            model="CVAR"
        try:
            # 4. 处理体积流量（第二部分）
            qv_wav = float(qv_str)
            rounded_qv = round(qv_wav)
            
            # 5. 处理实测压力（第三部分，0开头转为负值）
            if dp_str.startswith('0') and len(dp_str) > 1:
                dp_bench = -int(dp_str[1:])  # 020 → -20
            else:
                dp_bench = float(dp_str)
            rounded_dp = round(dp_bench)
            
        except (ValueError, IndexError):
            print(f"跳过解析失败的文件: {wav_name}")
            continue
        
        # 6. 在Excel中匹配对应行（精确匹配型号，四舍五入匹配数值）
        mask = (
            (excel_df['Mode'] == model) &
            (excel_df['Qv 体积流量\n(m3/h)'].round() == rounded_qv) &
            (excel_df['DP 实测压力\nBench\n(Pa)[input]'].round() == rounded_dp)
        )
        matched_rows = excel_df[mask]
        # 计算音频谱
        # 模式解释：\w+ 匹配一个或多个字母数字字符（包括CVAF），
        # 后面跟着 -数字-数字 的模式
        pattern = r'(\w+-\d+-\d+)'
        extracted_text = None
        match = re.search(pattern, wav_name)
        if match:
            extracted_text = match.group(1)

        # 如果未能从文件名解析出目标标识，则跳过该文件，避免后续未定义变量使用
        if extracted_text is None:
            print(f"跳过：无法从文件名解析标识，文件名={wav_name}")
            continue
        fft_name=sheet_name+"-FFT.xlsx"
        sound_qualityname=sheet_name+"-MIC1.xlsx"
        fft_dir = os.path.join("E:\lyh\paddlespeech\data",fft_name)
        sound_quality_dir = os.path.join("E:\lyh\paddlespeech\data_init",sound_qualityname)
                # 尝试从 FFT Excel 提取自功率谱并计算 A 计权谱，若出错则跳过该 WAV 文件
        try:
            spectrum_4hz = extract_spl_data(extracted_text, fft_dir)
        except Exception as e:
            print(f"跳过 {wav_name}：无法从 {fft_dir} 提取频谱数据，原因: {e}")
            continue

        if not spectrum_4hz:
            print(f"跳过 {wav_name}：提取到的频谱数据为空")
            continue

        try:
            spl_third_octave = spl_to_third_octave_20_10000(spectrum_4hz)
        except Exception as e:
            print(f"跳过 {wav_name}：频谱计算失败，原因: {e}")
            continue

        # 尝试读取声品质表中的指定列，出错时使用空值占位但不终止处理（记录警告）
        try:
            result_soundquality = find_specific_columns_by_regex(
                extracted_text, sound_quality_dir,
                ['SQ Metric (Loudness Free) [sone]', 
                 'SQ Metric (Sharpness [Zwicker, DIN 45631 A1 2010 Free]) [acum]', 
                 'SQ Metric (Roughness [DIN 45631 A1 2010 Free]) [asper]', 
                 'SQ Metric (Articulation Index [Closed]) [%]']
            )
        except Exception as e:
            print(f"警告：{wav_name} 未能从 {sound_quality_dir} 提取声品质数据，原因: {e}")
            continue
        # 8. 提取匹配结果（取首行，无匹配则用NaN）
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
            mode_val = row['Mode']
            type_val = sheet_name
            qv_val = row['Qv 体积流量\n(m3/h)']
            # dp_hvac_val = row['DP\nHVAC inlet\n(Pa)']
            pe_val = row['Pe\n(W)']
            rpm_val = row['N 鼓风机转速\n(rpm)']
           
              # 换算压力
            try:
                dp_pred = predict_pressure(
                    polynomial_results=fit_results,
                    model=sheet_name,  # 替换为实际工作表名称（型号）
                    real_speed=rpm_val,  # 替换为实际转速值
                    real_power=pe_val   # 替换为实际功率值
                )
            except ValueError as e:
                print(f"预测压力时出错: {e}")
                continue
            
            m1_val = row['Lp M1 麦克风总计值\n (dBA)']
            row = (hvac_parameter['hvac'] == type_val)
            sheet_row=hvac_parameter[row]
            if not sheet_row.empty and not sheet_row['diameter'].empty:
                if model=="CVAF"or model=="CVAR":
                    diameter=sheet_row['diameter'].values[0]
                    area1=sheet_row['cold exchange'].values[0]
                    area2=sheet_row['cold path'].values[0]
                    area3=sheet_row['vent'].values[0]
                    v1=qv_val/3600/area1*1000000
                    v2=qv_val/3600/area2*1000000
                    v3=qv_val/3600/area3*1000000
                elif model=='HFF':
                    diameter=sheet_row['diameter'].values[0]
                    area1=sheet_row['heat exchange'].values[0]
                    area2=sheet_row['heat path'].values[0]
                    area3=sheet_row['foot'].values[0]
                    v1=qv_val/3600/area1*1000000
                    v2=qv_val/3600/area2*1000000
                    v3=qv_val/3600/area3*1000000
                elif model=='HDF':
                    diameter=sheet_row['diameter'].values[0]
                    area1=sheet_row['heat exchange'].values[0]
                    area2=sheet_row['heat path'].values[0]
                    area3=sheet_row['defrost'].values[0]
                    v1=qv_val/3600/area1*1000000
                    v2=qv_val/3600/area2*1000000
                    v3=qv_val/3600/area3*1000000
                else:
                    print(f"未找到匹配数据，已跳过文件: {wav_name}")
                    continue

                # 8. 添加到CSV数据列表
                csv_data.append({
                    'WAV文件路径': wav_path,
                    'WAV文件名': wav_name,
                    'Mode': mode_val,
                    'Type': type_val,
                    'Qv 体积流量\n(m3/h)': qv_val,
                    'DP\nHVAC inlet\n(Pa)': dp_pred,
                    'N 鼓风机转速\n(rpm)': rpm_val,
                    'Diameter': diameter,
                    '流速v1': v1,
                    '流速v2': v2,
                    '流速v3': v3,
                    'Lp M1 麦克风总计值\n (dBA)': m1_val,
                    '1-3 octave band SPL': spl_third_octave,
                    'Magnitudes': spectrum_4hz,
                    'Loudness': result_soundquality['SQ Metric (Loudness Free) [sone]'],
                    'Sharpness': result_soundquality['SQ Metric (Sharpness [Zwicker, DIN 45631 A1 2010 Free]) [acum]'],
                    'Roughness': result_soundquality['SQ Metric (Roughness [DIN 45631 A1 2010 Free]) [asper]'],
                    'qingxidu': result_soundquality['SQ Metric (Articulation Index [Closed]) [%]']
                    })
            else:
                print(f"警告：未找到 {type_val} 的 diameter 数据，已跳过文件: {wav_name}")     
        else:
            print(f"未找到匹配数据，已跳过文件: {wav_name}")
        
        
        
    # 9. 保存为CSV文件
    pd.DataFrame(csv_data).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV文件已生成: {output_csv_path}")  

def process_sheet(file_path, sheet_name, fit_results,hvac_parameter):
    """
    处理指定的Excel工作表，清理数据并生成包含WAV文件信息的CSV
    """
    # 读取工作表
    df, sheet_name = read_excel_file(file_path, sheet_name=sheet_name)
    # 清理并保存数据
    clean_and_save_to_excel(df, 'N 鼓风机转速\n(rpm)', f'{sheet_name}.xlsx')
    # 生成csv文件
    base_dir = os.path.dirname(file_path)
    wav_dir = os.path.join(base_dir, sheet_name)
    file_type = "wav"
    excel_path = f"{sheet_name}.xlsx"
    csv_path = f"{sheet_name}.csv"
    create_wav_info_csv(wav_dir, file_type, excel_path, csv_path, sheet_name, fit_results,hvac_parameter)      

if __name__ == "__main__":
    file_path = "E:\lyh\paddlespeech\快速预测数据采集表 .xlsm"
    # sheet_names = get_all_sheet_names(file_path)
    # print("所有工作表名称:", sheet_names)
    """" ['SA5H', 'H97D', 'MAR2 2Z', 'MAR2 EVA2', 'BYD HTH', 
    'E0Y-3Z3M', 'NU2', 'X03', 'MBQ', 'SRH', 'MEB', 'T2X RHD',
      'M1E', 'KKL', 'GEM', 'T1X', 'H37', 'AD02']"

    工作表的所有列名
        ['Mode', 'Qv 体积流量\n(m3/h)', 'Qm\n(kg/h)', 'DP 实测压力\nBench\n(Pa)[input]', 'DP\nHVAC inlet\n(Pa)', 'DP排水口压 
    力\n(Pa)', 'DP空调压降\n(Pa)', 'U  鼓风机电压\n(V)', 'I  鼓风机电流\n(A)', 'Pe\n(W)', 'N 鼓风机转速\n(rpm)', 'Lp M1  
    麦克风总计值\n (dBA)', 'Lp M2\n (dBA)', 'RPM  12阶频率', '鼓风机\n直径', '鼓风机\n结构参数', '风道\n截面积', '风道\n 
    长度', '测试结果\n频域\n（超链接：新建数据表）', '测试结果\n时域\n（超链接）', '测试原文件\n（超链接）', 'NOTE', 'Aeraulic Conditions', 'Outlets', 'Remarks']
"""
    """
    写一个创建csv文件的函数，
    传入参数是输入文件夹的路径、读取的文件类型控制参数、一个excel文件路径，输出csv文件路径。
    将输入文件夹的以.wav结尾的文件的路径和名称全部读取出来，分别作为csv文件的第一列和第二列。
    wav文件的名称以-分隔，将其拆分，得到的前三个部分，第一部分是空调型号，类型是字符串。
    第二部分是体积流量，类型是数值。第三部分是实测压力，类型是数值，如果是负值会命名为0后面跟上数字，以0作为负号，
    例如HFF-83-020-.wav，分成三部分后为HFF、83、020,020应转换成-20。
    这三部分分别对应excel文件里的'Mode', 'Qv 体积流量\n(m3/h)'，'DP 实测压力\nBench\n(Pa)[input]'。
    在excel文件里对应列查找第一部分相同、第二部分四舍五入后相等、第三部分（0开头的负值先转化为真实负值）四舍五入后向等的行，按顺序返回该行对应的'Mode', 'Qv 体积流量\n(m3/h)'，'DP\nHVAC inlet\n(Pa)'，'N 鼓风机转速\n(rpm)'的字符串或值，分别作为csv文件的第三第四第五列​
    
    """
    # sheet_names = ['M1E', 'KKL', 'GEM', 'T1X', 'H37', 'AD02']
    # sheet_names = ['AD02']
    sheet_names = ['M1E','KKL','AD02']
    # analyzer = ISO5321PsychoacousticAnalyzer(65536)
    fit_results = process_excel_polynomial_fit('blower1.xlsx')
    hvac_parameter=pd.read_excel('hvac_parameter.xlsx')
    # print(hvac_parameter[hvac_parameter['hvac']=='M1E'])
    for sheet_name in sheet_names:
        process_sheet(file_path, sheet_name, fit_results,hvac_parameter)


    # # 读取工作表
    # df, sheet_name = read_excel_file(file_path,sheet_name="MAR2 EVA2")
    # # print(df.columns.tolist())  # 转换为列表形式打印，更易读
    # # 清理并保存数据
    # clean_and_save_to_excel(df, 'N 鼓风机转速\n(rpm)', f'{sheet_name}.xlsx')
    # # 生成csv文件
    # wav_dir = r"E:\lyh\paddlespeech\MAR2 EVA2"
    # file_type = "wav"
    # excel_path = f"{sheet_name}.xlsx"
    # csv_path = f"{sheet_name}.csv"
    # create_wav_info_csv(wav_dir, file_type, excel_path, csv_path,sheet_name)

# import pandas as pd
# file_path = "E:\lyh\paddlespeech\data_init\AD02-MIC1.xlsx"
# xls= pd.ExcelFile(file_path)
# df=xls.parse(sheet_name=xls.sheet_names[0])
# print(df.columns.tolist())
# # print(df[:12])

