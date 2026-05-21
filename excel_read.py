import pandas as pd
import os
import glob
import wave
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.signal import hilbert,welch
from DP_data import process_excel_polynomial_fit,predict_pressure
from sound_quality1 import ISO5321PsychoacousticAnalyzer

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
def calculate_one_third_octave_spectrum_from_wav(wav_path, sensitivity_v_per_pa=0.04978, 
                                               full_scale_voltage=10.0, reference_pressure=20e-6,
                                               a_weighting=True):
    """
    直接从WAV文件路径计算物理精确的1/3倍频程声压级频谱，支持A计权
    
    参数:
        wav_path: WAV文件路径
        sensitivity_v_per_pa: 麦克风灵敏度 (V/Pa)
        full_scale_voltage: 满量程电压 (V)
        reference_pressure: 参考声压 (Pa)
        a_weighting: 是否应用A计权
    
    返回:
        center_freqs: 中心频率列表 (Hz)
        spl_values: 对应的声压级列表 (dB SPL)
        spl_values_a: A计权声压级列表 (dB(A)) [如启用A计权]
    """
    
    # 读取WAV文件[1](@ref)
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            # 获取音频参数
            n_channels = wav_file.getnchannels()
            samp_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            print(f"音频信息: {n_channels}声道, 采样宽度{samp_width}字节, 采样率{framerate}Hz, {n_frames}样本")
            
            # 读取音频数据
            frames = wav_file.readframes(n_frames)
            
        # 根据采样宽度解析数据
        if samp_width == 2:  # 16-bit
            dtype = np.int16
            max_value = 32768.0
        elif samp_width == 3:  # 24-bit  
            # 24-bit需要特殊处理
            dtype = np.int32
            max_value = 8388608.0
            # 将3字节数据转换为32位整数
            frames = np.frombuffer(frames, dtype=np.uint8)
            frames = frames.reshape(-1, 3)
            frames = np.pad(frames, ((0, 0), (1, 0)), mode='constant')  # 填充为4字节
            frames = frames.view(np.int32).flatten()
            frames = (frames >> 8).astype(np.int32)  # 右移8位得到24位有符号整数
        elif samp_width == 4:  # 32-bit
            dtype = np.int32
            max_value = 2147483648.0
        else:
            raise ValueError(f"不支持的采样宽度: {samp_width}字节")
        
        if samp_width != 3:  # 24-bit已经特殊处理过
            # 将字节数据转换为numpy数组
            audio_data = np.frombuffer(frames, dtype=dtype)
            print(f"音频数据形状: {audio_data[:10]}")  # 打印前10个样本
        
        # 处理多声道：转换为单声道
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)
            audio_data = np.mean(audio_data, axis=1)  # 各声道平均
            print(f"已将{n_channels}声道转换为单声道")
        
        # 将整数样本转换为归一化的浮点数 (-1到1)
        y = audio_data.astype(np.float64) / max_value
        
    except Exception as e:
        raise ValueError(f"读取WAV文件失败: {e}")
    
    # 将归一化的样本值转换为实际电压，再转换为实际声压 (Pa)
    actual_voltage = y * full_scale_voltage
    pressure_signal_pa = actual_voltage / sensitivity_v_per_pa
    
    # 标准1/3倍频程中心频率 (ISO标准)[3](@ref)
    all_center_freqs = np.array([
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 
        4000, 5000, 6300, 8000, 10000
    ])
    
    # 只保留小于奈奎斯特频率的中心频率
    nyquist = framerate / 2
    valid_mask = all_center_freqs < nyquist
    center_freqs = all_center_freqs[valid_mask]
    
    if len(center_freqs) == 0:
        raise ValueError(f"采样率{framerate}Hz过低，无法分析任何1/3倍频程频带")
    
    print(f"分析频率范围: {center_freqs[0]:.1f}-{center_freqs[-1]:.1f}Hz")
    
    # 计算每个1/3倍频程的上下限频率
    octave_ratio = 2 ** (1/6)  # 1/3倍频程比率
    lower_freqs = center_freqs / octave_ratio
    upper_freqs = center_freqs * octave_ratio
    
    # 使用Welch方法计算功率谱密度
    nperseg = min(16384, len(pressure_signal_pa))
    freqs, psd = signal.welch(
        pressure_signal_pa, 
        fs=framerate, 
        nperseg=nperseg,
        scaling='density',
        average='mean'
    )
    
    # 计算A计权修正值[6,7](@ref)
    if a_weighting:
        a_weighting_factors = calculate_a_weighting_factors(center_freqs)
        print("已启用A计权滤波")
    
    spl_values = []
    spl_values_a = []
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
            if p_rms > 1e-10:  # 避免对极小声压取对数
                spl = 20 * np.log10(p_rms / reference_pressure)
            else:
                spl = -120.0  # 极安静环境
        else:
            # 没有频率点落在该频带内
            continue
        
        valid_center_freqs.append(fc)
        spl_values.append(spl)
        
        # 应用A计权修正[6](@ref)
        if a_weighting:
            spl_a = spl + a_weighting_factors[i]
            spl_values_a.append(spl_a)
    
    if a_weighting:
        return valid_center_freqs, spl_values, spl_values_a
    else:
        return valid_center_freqs, spl_values

def calculate_a_weighting_factors(frequencies):
    """
    计算A计权网络的频率修正值[6,7](@ref)
    
    参数:
        frequencies: 频率数组 (Hz)
    
    返回:
        weighting_factors: A计权修正值数组 (dB)
    """
    # A计权公式基于IEC 61672-1标准[7](@ref)
    f = np.array(frequencies)
    f2 = f ** 2
    
    # A计权公式[8](@ref)
    ra = (12194**2 * f2**2) / (
        (f2 + 20.6**2) * 
        np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
        (f2 + 12194**2)
    )
    
    # 计算分贝值并归一化到1kHz为0dB[6](@ref)
    a_weighting = 20 * np.log10(ra) - 20 * np.log10(1.2588966)
    
    # 对于极低频率进行特殊处理
    a_weighting[f < 0.1] = -100  # 低于0.1Hz几乎完全衰减
    
    return a_weighting

def apply_a_weighting_filter_design(pressure_signal, fs):
    """
    使用IIR滤波器设计实现A计权(时域滤波方法)[8](@ref)
    
    参数:
        pressure_signal: 声压信号 (Pa)
        fs: 采样率 (Hz)
    
    返回:
        filtered_signal: A计权滤波后的信号
    """
    # A计权滤波器的极点频率[8](@ref)
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    
    # 计算模拟滤波器系数[8](@ref)
    NUMs = [
        (2 * np.pi * f4)**2 * (10**(1.9997/20)), 
        0, 0, 0, 0
    ]
    
    DENs = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1)**2]
    )
    DENs = np.polymul(DENs, [1, 2 * np.pi * f3])
    DENs = np.polymul(DENs, [1, 2 * np.pi * f2])
    
    # 使用双线性变换设计数字滤波器[8](@ref)
    b, a = signal.bilinear(NUMs, DENs, fs=fs)
    
    # 应用滤波器
    filtered_signal = signal.lfilter(b, a, pressure_signal)
    
    return filtered_signal
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

def calibrate_audio_signal(y, sensitivity_v_per_pa=0.04978, full_scale_voltage=10.0):
    """
    将音频信号从归一化的电压值转换为具有物理单位（Pa）的声压信号。

    参数:
        y: 输入的音频信号（归一化到[-1, 1]的电压样本）。
        sensitivity_v_per_pa: 麦克风灵敏度 (V/Pa)。例如，49.78 mV/Pa 应输入为 0.04978。
        full_scale_voltage: 数据采集设备的满量程电压 (V)。

    返回:
        calibrated_signal: 校准后的声压信号 (单位: Pa)。
    """
    # 1. 将归一化的样本值还原为实际电压
    actual_voltage = y * full_scale_voltage  # 单位: V

    # 2. 根据灵敏度将电压转换为声压
    # 灵敏度公式: 灵敏度 = 输出电压 / 输入声压 => 声压 = 电压 / 灵敏度
    pressure_signal_pa = actual_voltage / sensitivity_v_per_pa  # 单位: Pa

    return pressure_signal_pa

def create_wav_info_csv(input_folder, file_type, excel_path, output_csv_path,sheet_name=None, fit_results=None, analyzer=None,hvac_parameter=None):
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
        frequencies, magnitudes = calculate_spectrum(wav_path)
        # print(len(magnitudes))
        # exit()
        # 7. 计算1-3 octave 频段声压级
        # y, sr = librosa.load(wav_path, sr=None, mono=True)
        fre,result_no_a,octave_result = calculate_one_third_octave_spectrum_from_wav(wav_path)
        # 9. 计算Bark频带响度
        # bark_band_loudness = compute_bark_band_loudness(wav_path)
        # y, sr = librosa.load(wav_path, sr=None)
        # 1. 读取WAV文件，得到归一化的样本数据 y 和采样率 fs
        y, fs = librosa.load(wav_path, sr=None, mono=True)

        # 2. 进行物理校准，将y转换为声压信号
        calibrated_y = calibrate_audio_signal(y, sensitivity_v_per_pa=0.04978, full_scale_voltage=10.0)
        sound_quality = analyzer.analyze_audio_signal(calibrated_y)

      
        
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

def process_sheet(file_path, sheet_name, fit_results, analyzer,hvac_parameter):
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
    create_wav_info_csv(wav_dir, file_type, excel_path, csv_path, sheet_name, fit_results, analyzer,hvac_parameter)      

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
    sheet_names = ['KKL', 'GEM', 'H37', 'AD02']
    analyzer = ISO5321PsychoacousticAnalyzer(65536)
    fit_results = process_excel_polynomial_fit('blower1.xlsx')
    hvac_parameter=pd.read_excel('hvac_parameter.xlsx')
    # print(hvac_parameter[hvac_parameter['hvac']=='M1E'])
    for sheet_name in sheet_names:
        process_sheet(file_path, sheet_name, fit_results, analyzer,hvac_parameter)


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

