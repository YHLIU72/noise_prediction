import pandas as pd
import os
import glob
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.signal import hilbert,welch
from DP_data import process_excel_polynomial_fit,predict_pressure
from sound_quality import ISO5321PsychoacousticAnalyzer
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

def calculate_audio_quality(file_path):
    """
    计算音频文件的五个声品质指标:响度、尖锐度、粗糙度、波动度和音调度
    
    参数:
        file_path (str): WAV音频文件路径
        
    返回:
        dict: 包含五个声品质指标的字典
    """
    # 读取音频文件
    y, sr = librosa.load(file_path, sr=None)
    
    # 1. 响度计算 (使用A计权)
    # ... 现有响度计算代码 ...
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    a_weight = librosa.A_weighting(librosa.mel_frequencies(n_mels=80, fmin=20, fmax=sr/2))
    a_weight = a_weight[:, np.newaxis]  # 形状变为 (80, 1)

    loudness = np.mean(S_db + a_weight)
    
    # 2. 尖锐度计算 (基于频谱重心)
    # ... 现有尖锐度计算代码 ...
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sharpness = np.mean(spectral_centroid) / sr  # 归一化到0-1范围
    
    # 3. 粗糙度计算 (基于振幅调制)
    # ... 现有粗糙度计算代码 ...
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    mod_amplitude = np.max(amplitude_envelope) - np.min(amplitude_envelope)
    roughness = mod_amplitude / np.mean(amplitude_envelope)
    
    # 4. 波动度计算 (基于频率调制)
    # ... 现有波动度计算代码 ...
    zero_crossings = librosa.zero_crossings(y, pad=False)
    fluctuation = np.sum(zero_crossings) / len(y)
    
    # 5. 音调度计算 (基于基频变化)
    # ... 现有音调度计算代码 ...
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)]
    tonality = np.std(f0_clean) / np.mean(f0_clean) if len(f0_clean) > 0 else 0
    
    return {
        '响度': loudness,
        '尖锐度': sharpness,
        '粗糙度': roughness,
        '波动度': fluctuation,
        '音调度': tonality
    }

def calculate_spectrum(audio_path):
    # 读取音频文件，不改变原始采样率
    y, sr = librosa.load(audio_path, sr=None)
    
    # 应用汉明窗减少频谱泄漏
    window = np.hamming(len(y))
    y_windowed = y * window
    
    # 计算FFT
    n_fft = len(y_windowed)
    fft_result = np.fft.fft(y_windowed)
    
    # 计算频率轴
    frequencies = np.fft.fftfreq(n_fft, 1/sr)
    
    # 只保留正频率部分
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    magnitudes = np.abs(fft_result[positive_mask])
    
    # 限制频率范围在0到20000Hz
    freq_mask = frequencies <= 20000
    frequencies = frequencies[freq_mask]
    magnitudes = magnitudes[freq_mask]
    
    # 归一化幅度值到0-1范围
    magnitudes = magnitudes / np.max(magnitudes)
    
    return frequencies, magnitudes

def calculate_one_third_octave_spectrum(y, sr, reference_pressure=20e-6):
    """
    计算1/3倍频程频谱（修正版本）
    
    参数:
        y: 音频信号
        sr: 采样率
        reference_pressure: 参考声压 (20µPa)
        
    返回:
        center_freqs: 中心频率列表
        spl_values: 对应的声压级(dB)列表
    """
    # 标准1/3倍频程中心频率 (ISO标准) 25个中心频率
    center_freqs = np.array([
        40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
        630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
        10000
    ])
    
    # 只保留小于奈奎斯特频率的中心频率
    nyquist = sr / 2
    valid_mask = center_freqs < nyquist
    center_freqs = center_freqs[valid_mask]
    
    # 计算每个1/3倍频程的上下限频率
    octave_ratios = 2 ** (1/6)  # 1/3倍频程
    lower_freqs = center_freqs / octave_ratios
    upper_freqs = center_freqs * octave_ratios
    
    # 使用Welch方法计算功率谱密度
    nperseg = min(16384, len(y))  # 段长
    if nperseg < 256:
        nperseg = 256
    
    # 计算功率谱
    freqs, psd = signal.welch(
        y, 
        fs=sr, 
        nperseg=nperseg,
        scaling='density',
        average='mean'
    )
    
    # 计算每个1/3倍频程带的能量
    spl_values = []
    valid_center_freqs = []
    
    for i, fc in enumerate(center_freqs):
        lower = lower_freqs[i]
        upper = upper_freqs[i]
        
        # 找到频带内的频率索引
        idx = np.where((freqs >= lower) & (freqs <= upper))[0]
        
        if len(idx) > 0:
            # 使用梯形法积分计算频带能量
            band_energy = np.trapz(psd[idx], freqs[idx])
            
            # 计算声压有效值
            p_rms = np.sqrt(band_energy)
            
            # 计算声压级
            if p_rms > 0:
                spl = 20 * np.log10(p_rms / reference_pressure)
            else:
                spl = -np.inf
        else:
            # 没有频率点落在该频带内
            continue
        
        valid_center_freqs.append(fc)
        spl_values.append(spl)
    
    return spl_values
# def compute_bark_band_loudness(audio_path):
#     """
#     基于Zwicker听觉模型计算音频文件在24个Bark频带下的频带响度（特定响度，sone/Bark）
    
#     参数:
#         audio_path (str): 音频文件路径
#     返回:
#         np.array: 24个Bark频带的特定响度值（sone/Bark）
#     """
#     # 24个Bark频带边界(Hz)
#     bark_boundaries = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 
#                        1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 
#                        6400, 7700, 9500, 12000, 15500]
#     num_bands = len(bark_boundaries) - 1  # 共24个频带
    
#     # 加载音频文件（保持原始采样率）
#     y, sr = librosa.load(audio_path, sr=None)
    
#     # 使用Welch方法计算功率谱密度(PSD)，提高频率分辨率以适应高频Bark带
#     f, Pxx = welch(y, sr, nperseg=8192)  # 增加nperseg提升高频分辨率
#     delta_f = f[1] - f[0]  # 频率分辨率(Hz)
    
#     # -------------------------- Zwicker模型核心计算 --------------------------
#     # 1. 功率谱转换为声压级(SPL, dB re 20µPa)
#     p0 = 20e-6  # 参考声压(20µPa)
#     Z = 400     # 空气特性阻抗(Ω)
#     power_ref = (p0 ** 2) / Z  # 参考功率密度(W/Hz)
#     spl = 10 * np.log10(np.maximum(Pxx * delta_f / power_ref, 1e-20))  # 避免log(0)
    
#     # 2. ISO 226:2003等响度曲线（安静环境下的听阈，dB SPL）
#     def iso226_threshold(freq):
#         if freq <= 0:
#             return np.inf  # 0Hz处听阈无穷大
#         f_khz = freq / 1000
#         return 3.64 * (f_khz)**-0.8 - 6.5 * np.exp(-0.6 * (f_khz - 3.3)**2) + 1e-3 * (f_khz)**4
    
#     thresholds = np.array([iso226_threshold(freq) for freq in f])
    
#     # 3. 计算激励水平（线性刻度）：高于听阈的声压级转换为线性激励
#     excitation_linear = np.zeros_like(spl)
#     audible_mask = spl > thresholds  # 仅保留可听频率分量
#     excitation_linear[audible_mask] = 10 ** ((spl[audible_mask] - thresholds[audible_mask]) / 10)
    
#     # 4. 计算每个Bark频带的特定响度
#     bark_loudness = []
#     for i in range(num_bands):
#         band_low, band_high = bark_boundaries[i], bark_boundaries[i+1]
#         freq_mask = (f >= band_low) & (f <= band_high)  # 当前频带的频率掩码
        
#         if not np.any(freq_mask):
#             bark_loudness.append(0.0)
#             continue
        
#         # 积分频带内的线性激励
#         band_excitation = np.sum(excitation_linear[freq_mask] * delta_f)
        
#         if band_excitation <= 0:
#             bark_loudness.append(0.0)
#             continue
        
#         # Zwicker模型：激励水平(dB)转特定响度(sone/Bark)
#         E_band = 10 * np.log10(band_excitation)  # 频带激励水平(dB)
#         specific_loudness = 0.08 * (E_band ** 0.23) if E_band > 0 else 0.0
        
#         bark_loudness.append(specific_loudness)
    
#     return np.array(bark_loudness)

def create_wav_info_csv(input_folder, file_type, excel_path, output_csv_path,sheet_name=None,fit_results=None, analyzer=None,hvac_parameter=None):
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
        if model=="CVAR":
            mask = (
            (excel_df['Mode'] == model) &
            (excel_df['Qv 体积流量\n(m3/h)'].round() == rounded_qv) &
            (excel_df['DP排水口压力\n(Pa)'].round() == rounded_dp)
        )
            matched_rows = excel_df[mask]
        else:
            mask = (
                (excel_df['Mode'] == model) &
                (excel_df['Qv 体积流量\n(m3/h)'].round() == rounded_qv) &
                (excel_df['DP 实测压力\nBench\n(Pa)[input]'].round() == rounded_dp)
            )
            matched_rows = excel_df[mask]

        # 7. 计算频谱图
        frequencies, magnitudes = calculate_spectrum(wav_path)
        
        # 8. 计算1-3 octave 频段声压级
        y, sr = librosa.load(wav_path, sr=None)
        octave_result = calculate_one_third_octave_spectrum(y, sr)
        
        # 9. 计算Bark频带响度
        y, sr = librosa.load(wav_path, sr=None)
        sound_quality = analyzer.analyze_audio_signal(y)
        # print(bark_band_loudness.shape)
        # exit(0)
        
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
                    'Diameter': diameter,
                    '流速v1': v1,
                    '流速v2': v2,
                    '流速v3': v3,
                    'DP\nHVAC inlet\n(Pa)': dp_pred,
                    'N 鼓风机转速\n(rpm)': rpm_val,
                    'Lp M1 麦克风总计值\n (dBA)': m1_val,
                    '1-3 octave band SPL': octave_result,
                    'Magnitudes': magnitudes.tolist(),
                    'Bark Band Loudness': sound_quality['specific_loudness'].tolist(),
                    'Loudness': sound_quality['loudness'],
                    'Sharpness': sound_quality['sharpness'],
                    'Roughness': sound_quality['roughness'],
                    'Fluctuation': sound_quality['fluctuation']
                    })
            else:
                print(f"警告：未找到 {type_val} 的 diameter 数据，已跳过文件: {wav_name}")   
        else:
            print(f"未找到匹配数据，已跳过文件: {wav_name}")
        
        
        
    # 9. 保存为CSV文件
    pd.DataFrame(csv_data).to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV文件已生成: {output_csv_path}")  

def process_sheet(file_path, sheet_name,fit_results, analyzer,hvac_parameter):
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
    create_wav_info_csv(wav_dir, file_type, excel_path, csv_path, sheet_name,fit_results, analyzer,hvac_parameter)

def process_csv_files():
    # 读取Excel文件 (需要安装openpyxl库支持xlsx格式)
    wheel_df = pd.read_excel(r'E:\lyh\paddlespeech\noise\wheel_list1.xlsx', engine='openpyxl')
    
    # 获取csvdata文件夹下所有CSV文件
    csv_files = glob.glob(r'E:/lyh/paddlespeech/csvdata/*.csv')
    
    for csv_path in csv_files:
        # 获取CSV文件名（不含扩展名）作为匹配键
        csv_filename = os.path.splitext(os.path.basename(csv_path))[0]
        
        # 在Excel中查找匹配的行（第一列的值与CSV文件名匹配）
        # 注意：Excel第一列的索引为0
        matching_rows = wheel_df[wheel_df.iloc[:, 0] == csv_filename]
        
        if len(matching_rows) == 0:
            print(f"警告: 未找到 {csv_filename} 对应的Excel数据，跳过该文件")
            continue
        if len(matching_rows) > 1:
            print(f"警告: {csv_filename} 在Excel中有多个匹配行，使用第一行数据")
        
        # 获取匹配行的第三、四、五、六列数据（Excel列索引为2,3,4,5）
        wheel_row = matching_rows.iloc[0]
        insert_values = wheel_row.iloc[2:6].values  # 提取C-F列数据
        
        # 读取CSV文件
        csv_df = pd.read_csv(csv_path)
        
        # 将Excel数据插入CSV的第十至第十三列（0-based索引9-12）
        for col_idx, value in zip(range(9, 13), insert_values):
            # 如果CSV列数不足，添加新列
            if col_idx >= len(csv_df.columns):
                csv_df.insert(col_idx, f'column_{col_idx}', value)
            else:
                # 如果列已存在，更新值
                csv_df.iloc[:, col_idx] = value
        
        # 保存修改后的CSV文件
        csv_df.to_csv(csv_path, index=False)
        print(f"已处理: {csv_path}")
        # print(csv_df.iloc[:2, 9:13])      

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
    # sheet_names = ['SA5H', 'H97D', 'MAR2 2Z', 'MAR2 EVA2', 'BYD HTH', 
    # 'E0Y 3Z3M', 'NU2', 'X03', 'MBQ', 'SRH', 'MEB', 'T2X RHD']
    sheet_names = ['SA5H', 'H97D', 'MRA2 EVA2', 'BYD HTH', 
    'E0Y 3Z3M', 'NU2', 'X03', 'MBQ', 'SRH', 'MEB', 'T2X RHD']
    analyzer = ISO5321PsychoacousticAnalyzer(65536)
    fit_results = process_excel_polynomial_fit('blower1.xlsx')
    hvac_parameter=pd.read_excel('hvac_parameter.xlsx')
    for sheet_name in sheet_names:
        process_sheet(file_path, sheet_name,fit_results, analyzer,hvac_parameter)
    # process_csv_files()


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

