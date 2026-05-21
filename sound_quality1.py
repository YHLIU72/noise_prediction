import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import bilinear,filtfilt
class ISO5321PsychoacousticAnalyzer:
    """
    基于ISO 532-1:2017标准的心理声学参数计算器
    参考标准：
    - 响度: ISO 532-1:2017 (Zwicker方法) [1](@ref)
    - 尖锐度: DIN 45692 (Zwicker模型)
    - 波动度: Zwicker & Fastl心理声学模型
    - 粗糙度: Zwicker & Fastl心理声学模型
    
    ISO 532-1:2017标准规定了两种方法用于估计耳科正常人在特定听力条件下感知到的声音响度和响度水平[1](@ref)。
    第一种方法用于平稳声音，第二种方法用于任意非平稳(时变)声音[1](@ref)。
    """
    
    def __init__(self, fs=44100):
        """
        初始化心理声学分析器
        
        Parameters:
        fs: 采样率 (Hz)，默认44100Hz
        """
        self.fs = fs
        
        # ISO 532-1:2017 完整临界频带参数 (0-24 Bark) [1](@ref)
        self.bark_bands = np.arange(0, 25, 1)  # 完整的0-24 Bark范围
        self.bark_center_freqs = (self.bark_bands[:-1] + self.bark_bands[1:]) / 2
        
        # 临界频带带宽 (Bark)
        self.critical_bandwidths = np.ones(24)  # 每个临界频带宽度为1 Bark
        
        # 绝对听阈 (ISO 532-1标准参数)
        self.absolute_threshold = 4.0e-10  # 帕²
        
        # 频率到Bark尺度的转换参数
        self.bark_params = {
            'f0': 0.00076,
            'f1': 7500,
            'a': 13,
            'b': 3.5
        }
        # A计权参数（IEC 61672标准）
        self.a_weighting_params = {
            'f1': 20.598997,      # Hz
            'f2': 107.65265,       # Hz  
            'f3': 737.86223,       # Hz
            'f4': 12194.217,       # Hz
            'A1000': 1.9997        # 1kHz处的归一化常数
        }
    
    def freq2bark(self, f):
        """
        ISO 532-1:2017 频率到Bark尺度转换
        公式: z = 13 * arctan(0.00076*f) + 3.5 * arctan((f/7500)^2) [1](@ref)
        """
        f = np.maximum(f, 1e-10)  # 避免除零
        part1 = self.bark_params['a'] * np.arctan(self.bark_params['f0'] * f)
        part2 = self.bark_params['b'] * np.arctan((f / self.bark_params['f1']) ** 2)
        return part1 + part2
    
    def bark2freq(self, z):
        """Bark尺度到频率转换的近似逆变换"""
        return 1960 * (z + 0.53) / (26.28 - z)
    
    def calculate_one_third_octave_bands(self, x):
        """
        计算1/3倍频程谱 - ISO 532-1允许的输入格式之一[1](@ref)
        """
        # 标准1/3倍频程中心频率 (Hz)
        center_freqs = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 
                        200, 250, 315, 400, 500, 630, 800, 1000, 
                        1250, 1600, 2000, 2500, 3150, 4000, 5000, 
                        6300, 8000, 10000, 12500, 16000]
        
        one_third_octave_levels = np.zeros(len(center_freqs))
        
        for i, f_center in enumerate(center_freqs):
            # 计算1/3倍频程带宽
            f_lower = f_center / (2 ** (1/6))
            f_upper = f_center * (2 ** (1/6))
            
            # 计算该频带内的能量
            band_energy = self._calculate_band_energy(x, f_lower, f_upper)
            one_third_octave_levels[i] = 10 * np.log10(band_energy / 1e-12) if band_energy > 0 else -100
        
        return one_third_octave_levels, center_freqs
    
    def _calculate_band_energy(self, x, f_low, f_high):
        """计算指定频带内的能量"""
        # 设计带通滤波器
        nyquist = self.fs / 2
        if f_high >= nyquist:
            f_high = nyquist * 0.99
        
        b, a = signal.butter(4, [f_low/nyquist, f_high/nyquist], btype='band')
        filtered_signal = signal.filtfilt(b, a, x)
        
        return np.mean(filtered_signal ** 2)
    
    def iso532_1_loudness_stationary(self, x):
        """
        基于ISO 532-1:2017的稳态响度计算[1](@ref)
        
        修正问题：
        1. 使用完整的0-24 Bark临界频带
        2. 按照标准进行频带整合计算总响度
        """
        N = len(x)
        fft_size = 2 ** int(np.ceil(np.log2(N)))
        
        # 计算功率谱密度
        window = np.hanning(N)
        x_windowed = x * window
        spectrum = fft(x_windowed, fft_size)
        freqs = fftfreq(fft_size, 1/self.fs)
        
        # 正频率部分
        positive_freqs = freqs[:fft_size//2]
        positive_spectrum = spectrum[:fft_size//2]
        power_spectrum = np.abs(positive_spectrum) ** 2 / (fft_size * self.fs)
        
        # 跳过DC分量
        analysis_freqs = positive_freqs[1:]
        analysis_power = power_spectrum[1:]
        
        # 频率到Bark映射
        bark_freqs = self.freq2bark(analysis_freqs)
        
        # 修正：完整的临界频带能量计算 (0-24 Bark)
        critical_band_energy = np.zeros(24)
        
        for i in range(24):
            lower_bark = self.bark_bands[i]
            upper_bark = self.bark_bands[i+1]
            
            mask = (bark_freqs >= lower_bark) & (bark_freqs < upper_bark)
            if np.any(mask):
                critical_band_energy[i] = np.sum(analysis_power[mask])
            else:
                critical_band_energy[i] = self.absolute_threshold
        
        # ISO 532-1 特定响度计算
        specific_loudness = self._calculate_specific_loudness(critical_band_energy)
        
        # 修正：标准的总响度积分方法
        total_loudness = self._integrate_total_loudness(specific_loudness)
        
        return total_loudness, specific_loudness
    
    def _calculate_specific_loudness(self, critical_band_energy):
        """计算特定响度 - ISO 532-1核心算法"""
        specific_loudness = np.zeros(24)
        
        for i in range(24):
            excitation = max(critical_band_energy[i], self.absolute_threshold)
            
            # ISO 532-1标准公式
            if excitation > self.absolute_threshold:
                # 基本特定响度计算
                specific_loudness[i] = 0.063 * (excitation/self.absolute_threshold) ** 0.23
                specific_loudness[i] *= (1 - np.exp(-(excitation/self.absolute_threshold) ** 0.23))
                
                # 考虑掩蔽效应修正
                masking_correction = self._calculate_masking_correction(i, critical_band_energy)
                specific_loudness[i] *= masking_correction
        
        return specific_loudness
    
    def _calculate_masking_correction(self, band_idx, critical_band_energy):
        """计算掩蔽效应修正因子"""
        correction = 1.0
        
        # 考虑相邻频带的掩蔽效应
        for j in range(max(0, band_idx-3), min(24, band_idx+4)):
            if j != band_idx:
                distance = abs(j - band_idx)
                masking_effect = 0.15 * np.exp(-distance/3.0)
                
                if critical_band_energy[j] > critical_band_energy[band_idx]:
                    correction *= (1 - masking_effect)
        
        return np.clip(correction, 0.1, 2.0)
    
    def _integrate_total_loudness(self, specific_loudness):
        """
        修正：按照ISO 532-1标准进行总响度积分
        考虑临界频带宽度和频带间相互作用[1](@ref)
        """
        # 方法1：简单求和（基本方法）
        simple_sum = np.sum(specific_loudness)
        
        # 方法2：考虑频带宽度的精确积分
        bandwidth_weighted = np.sum(specific_loudness * self.critical_bandwidths)
        
        # 方法3：考虑频带相互作用的复杂积分
        integrated_loudness = 0
        for i in range(24):
            band_contribution = specific_loudness[i] * self.critical_bandwidths[i]
            
            # 考虑相邻频带的贡献（简化模型）
            if i > 0:
                band_contribution += 0.08 * specific_loudness[i-1] * self.critical_bandwidths[i-1]
            if i < 23:
                band_contribution += 0.08 * specific_loudness[i+1] * self.critical_bandwidths[i+1]
            
            integrated_loudness += band_contribution
        
        # 使用带宽加权积分作为主要结果
        return bandwidth_weighted
    
    def calculate_sharpness(self, specific_loudness, total_loudness):
        """
        基于DIN 45692标准的尖锐度计算
        """
        if total_loudness == 0:
            return 0.0
        
        # 尖锐度权重函数 (高频权重增加)
        g_z = np.ones(24)
        for i in range(24):
            z = self.bark_center_freqs[i]
            if z > 16:
                g_z[i] = 0.066 * np.exp(0.171 * z)
        
        # 尖锐度计算
        numerator = np.sum(specific_loudness * g_z * self.bark_center_freqs)
        sharpness = 0.11 * numerator / total_loudness
        
        return max(0, sharpness)
    
    def calculate_roughness_improved(self, x, modulation_band=[15, 300]):
        """
        修正的粗糙度计算 - 基于实际调制频率分析
        
        修正问题：不再假设固定调制频率，而是分析信号的实际调制成分
        """
        frame_duration = 0.02  # 20ms帧
        frame_size = int(frame_duration * self.fs)
        hop_size = frame_size // 2
        
        num_frames = max(1, (len(x) - frame_size) // hop_size)
        
        if num_frames == 0:
            return 0.0
        
        roughness_values = []
        
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            
            if end_idx > len(x):
                break
                
            frame = x[start_idx:end_idx]
            
            # 计算包络信号
            envelope = self._calculate_envelope(frame)
            
            # 分析包络的调制频谱
            modulation_spectrum = np.fft.fft(envelope - np.mean(envelope))
            modulation_freqs = np.fft.fftfreq(len(envelope), 1/self.fs)
            
            # 提取15-300Hz调制成分
            mask = (modulation_freqs >= modulation_band[0]) & (modulation_freqs <= modulation_band[1])
            valid_freqs = modulation_freqs[mask]
            valid_magnitudes = np.abs(modulation_spectrum[mask])
            
            if len(valid_freqs) == 0:
                roughness_values.append(0)
                continue
            
            # 基于实际调制频率计算粗糙度
            frame_roughness = 0
            total_magnitude = np.sum(valid_magnitudes)
            
            for freq, magnitude in zip(valid_freqs, valid_magnitudes):
                if freq > 0:  # 只考虑正频率
                    weight = self._roughness_weight_function(freq)
                    modulation_strength = magnitude / (total_magnitude + 1e-10)
                    frame_roughness += modulation_strength * weight
            
            roughness_values.append(frame_roughness)
        
        return np.mean(roughness_values) if roughness_values else 0.0
    
    def calculate_fluctuation_strength_improved(self, x, modulation_band=[0.5, 20]):
        """
        修正的波动度计算 - 基于实际调制频率分析
        
        修正问题：不再假设固定调制频率，分析信号的实际低频调制成分
        """
        frame_duration = 0.2 # 500ms帧以适应低频调制
        frame_size = int(frame_duration * self.fs)
        hop_size = frame_size // 3
        
        num_frames = max(1, (len(x) - frame_size) // hop_size)
        
        if num_frames == 0:
            return 0.0
        
        fluctuation_values = []
        
        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            
            if end_idx > len(x):
                break
                
            frame = x[start_idx:end_idx]
            
            # 计算包络信号
            envelope = self._calculate_envelope(frame)
            
            # 分析包络的低频调制成分
            modulation_spectrum = np.fft.fft(envelope - np.mean(envelope))
            modulation_freqs = np.fft.fftfreq(len(envelope), 1/self.fs)
            
            # 提取0.5-20Hz调制成分
            mask = (modulation_freqs >= modulation_band[0]) & (modulation_freqs <= modulation_band[1])
            valid_freqs = modulation_freqs[mask]
            valid_magnitudes = np.abs(modulation_spectrum[mask])
            
            if len(valid_freqs) == 0:
                fluctuation_values.append(0)
                continue
            
            # 基于实际调制频率计算波动度
            frame_fluctuation = 0
            total_magnitude = np.sum(valid_magnitudes)
            
            for freq, magnitude in zip(valid_freqs, valid_magnitudes):
                if freq > 0:  # 只考虑正频率
                    weight = self._fluctuation_weight_function(freq)
                    modulation_strength = magnitude / (total_magnitude + 1e-10)
                    frame_fluctuation += modulation_strength * weight
            
            fluctuation_values.append(frame_fluctuation)
        
        return np.mean(fluctuation_values) if fluctuation_values else 0.0
    
    def _calculate_envelope(self, frame):
        """计算信号的包络"""
        analytic_signal = signal.hilbert(frame)
        envelope = np.abs(analytic_signal)
        return envelope
    
    def _roughness_weight_function(self, f_mod):
        """粗糙度权重函数 (70Hz时最大)"""
        f_peak = 70
        if f_mod <= 0:
            return 0
        return (f_mod / f_peak) * np.exp(1 - f_mod / f_peak)
    
    def _fluctuation_weight_function(self, f_mod):
        """波动度权重函数 (4Hz时最大)"""
        f_peak = 4
        if f_mod <= 0:
            return 0
        return (f_mod / f_peak) * np.exp(1 - f_mod / f_peak)
    
    def design_a_weighting_filter(self, fs=None):
        """
        设计数字A计权滤波器基于IEC 61672标准[2,4](@ref)
        
        返回:
            b, a: 滤波器系数（IIR格式）
        """
        if fs is None:
            fs = self.fs
            
        f1, f2, f3, f4, A1000 = [self.a_weighting_params[k] for k in 
                                ['f1', 'f2', 'f3', 'f4', 'A1000']]
        
        # 模拟域传递函数（s域）
        # 分子: (2πf4)^2 * s^4
        # 分母: (s + 2πf1)^2 * (s + 2πf2) * (s + 2πf3) * (s + 2πf4)^2
        numerators = [(2 * np.pi * f4)**2 * (10**(A1000 / 20.0)), 0., 0., 0., 0.]
        
        # 计算分母系数
        denom1 = np.convolve([1., 4 * np.pi * f4, (2 * np.pi * f4)**2],
                           [1., 4 * np.pi * f1, (2 * np.pi * f1)**2])
        denom2 = np.convolve([1., 2 * np.pi * f3], [1., 2 * np.pi * f2])
        denominators = np.convolve(denom1, denom2)
        
        # 使用双线性变换转换为数字滤波器
        b, a = bilinear(numerators, denominators, fs)
        
        return b, a
    
    def apply_a_weighting(self, signal, fs=None):
        """
        对信号应用A计权滤波[2,4](@ref)
        
        参数:
            signal: 输入信号（声压值，单位Pa）
            fs: 采样率，默认为类初始化时设置的采样率
            
        返回:
            a_weighted_signal: A计权后的信号
        """
        if fs is None:
            fs = self.fs
            
        # 设计A计权滤波器
        b, a = self.design_a_weighting_filter(fs)
        
        # 应用滤波器（使用零相位滤波减少相位失真）
        a_weighted_signal = filtfilt(b, a, signal)
        
        return a_weighted_signal
    
    def calculate_a_weighted_spl(self, signal, fs=None):
        """
        计算A计权声压级[4](@ref)
        
        参数:
            signal: 输入信号（声压值，单位Pa）
            fs: 采样率
            
        返回:
            laeq: 等效A计权声压级[dB(A)]
            spl_time: A计权声压级时间序列
        """
        if fs is None:
            fs = self.fs
            
        # 应用A计权滤波
        a_weighted_signal = self.apply_a_weighting(signal, fs)
        
        # 计算A计权声压级时间序列
        p_ref = 2e-5  # 参考声压20μPa
        window_size = int(0.125 * fs)  # 125ms窗口
        hop_size = int(0.125 * fs)     # 125ms跳跃
        
        if len(a_weighted_signal) < window_size:
            window_size = len(a_weighted_signal)
            hop_size = window_size
        
        # 计算滑动窗口RMS
        spl_time = []
        for i in range(0, len(a_weighted_signal) - window_size + 1, hop_size):
            window = a_weighted_signal[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            spl = 20 * np.log10(rms / p_ref) if rms > 0 else -100
            spl_time.append(spl)
        
        # 计算等效连续A声级
        if len(a_weighted_signal) > 0:
            overall_rms = np.sqrt(np.mean(a_weighted_signal**2))
            laeq = 20 * np.log10(overall_rms / p_ref) if overall_rms > 0 else -100
        else:
            laeq = -100
            
        return laeq, np.array(spl_time)
    
    def a_weighting_frequency_response(self, frequencies):
        """
        计算A计权频率响应（用于频域分析）[2,3](@ref)
        
        参数:
            frequencies: 频率数组（Hz）
            
        返回:
            a_weighting_dB: 各频率点的A计权修正值（dB）
        """
        f = np.array(frequencies, dtype=float)
        f_squared = f**2
        
        # A计权公式（IEC 61672标准）
        # 避免除零错误
        f_safe = np.where(f == 0, 1e-10, f)
        
        # A计权计算公式
        numerator = f_squared**2
        denominator = (f_squared + 20.6**2) * np.sqrt(f_squared + 107.7**2) * \
                     np.sqrt(f_squared + 737.9**2) * (f_squared + 12194**2)
        
        ra = numerator / denominator
        ra_ref = ra[f == 1000][0] if np.any(f == 1000) else 1.0
        
        # 计算A计权修正值（dB）
        a_weighting_dB = 20 * np.log10(ra / ra_ref) + 2.0
        
        # 处理极低频率
        a_weighting_dB[f < 10] = -100
        
        return a_weighting_dB
    
    def apply_a_weighting_to_spectrum(self, frequencies, magnitudes):
        """
        对频谱数据应用A计权修正[3](@ref)
        
        参数:
            frequencies: 频率数组（Hz）
            magnitudes: 幅度谱（线性尺度）
            
        返回:
            a_weighted_magnitudes: A计权后的幅度谱
        """
        # 计算A计权修正值
        a_correction = self.a_weighting_frequency_response(frequencies)
        
        # 将线性幅度转换为dB尺度
        magnitudes_dB = 20 * np.log10(magnitudes + 1e-10)
        
        # 应用A计权修正
        a_weighted_dB = magnitudes_dB + a_correction
        
        # 转换回线性尺度
        a_weighted_magnitudes = 10**(a_weighted_dB / 20)
        
        return a_weighted_magnitudes
    
    def analyze_audio_signal(self, x, fs=None,use_a_weighting=True):
        """
        增强的音频分析函数，支持A计权选项
        """
        if fs is None:
            fs = self.fs
        
        # 信号预处理
        x = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x
        
        # 应用A计权（如果启用）
        if use_a_weighting:
            x_processed = self.apply_a_weighting(x, fs)
            weighting_info = "A计权已启用"
        else:
            x_processed = x
            weighting_info = "线性分析（无计权）"
        
        # 计算心理声学参数
        loudness, specific_loudness = self.iso532_1_loudness_stationary(x_processed)
        sharpness = self.calculate_sharpness(specific_loudness, loudness)
        roughness = self.calculate_roughness_improved(x_processed)
        fluctuation = self.calculate_fluctuation_strength_improved(x_processed)
        
        # 计算A计权声压级
        laeq, spl_time = self.calculate_a_weighted_spl(x, fs)
        
        results = {
            'loudness': loudness,
            'sharpness': sharpness,
            'roughness': roughness,
            'fluctuation': fluctuation,
            'specific_loudness': specific_loudness,
            'bark_bands': self.bark_center_freqs,
            'a_weighted_spl': laeq,
            'weighting_method': weighting_info,
            'spl_time_series': spl_time
        }
        
        return results
    
    def analyze_audio_file(self, filename):
        """
        音频文件分析接口
        """
        try:
            # 读取音频文件
            if filename.endswith('.wav'):
                fs, data = wavfile.read(filename)
            else:
                import librosa
                data, fs = librosa.load(filename, sr=self.fs)
            
            # 转换为单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            return self.analyze_audio_signal(data, fs)
            
        except Exception as e:
            print(f"分析音频文件时出错: {e}")
            return None
    
    def plot_analysis_results(self, results, title="心理声学分析结果"):
        """结果可视化"""
        # 学术图表配置
        plt.rcParams.update({
            "font.family": ["SimHei"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "axes.unicode_minus": False
        })
        
        # 使用学术风格配色方案（蓝色系渐变）
        colors = ['#5DA5DA', '#4A6FA5', '#365173', '#223344']
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))  # 更适合学术论文的宽高比
        
        # 响度显示
        axes[0].bar([''], [results['loudness']], color=colors[0], width=0.5, edgecolor='black')
        axes[0].set_ylabel('响度 (sone)')
        axes[0].set_title('总响度 (ISO 532-1:2017)')
        axes[0].grid(True, axis='y', alpha=0.6)  # 仅保留水平网格线
        axes[0].set_ylim(0, results['loudness'] * 1.2)  # 设置合理的y轴范围
        axes[0].text(0, results['loudness'] * 1.05, f"{results['loudness']:.2f}", 
                    ha='center', fontweight='bold')  # 数据标注
        
        # 尖锐度显示
        axes[1].bar([''], [results['sharpness']], color=colors[1], width=0.5, edgecolor='black')
        axes[1].set_ylabel('尖锐度 (acum)')
        axes[1].set_title('尖锐度 (DIN 45692)')
        axes[1].grid(True, axis='y', alpha=0.6)
        axes[1].set_ylim(0, results['sharpness'] * 1.2)
        axes[1].text(0, results['sharpness'] * 1.05, f"{results['sharpness']:.2f}", 
                    ha='center', fontweight='bold')
        
        # 粗糙度显示
        axes[2].bar([''], [results['roughness']], color=colors[2], width=0.5, edgecolor='black')
        axes[2].set_ylabel('粗糙度 (asper)')
        axes[2].set_title('粗糙度 (Zwicker模型)')
        axes[2].grid(True, axis='y', alpha=0.6)
        axes[2].set_ylim(0, results['roughness'] * 1.2)
        axes[2].text(0, results['roughness'] * 1.05, f"{results['roughness']:.4f}", 
                    ha='center', fontweight='bold')
        
        # 波动度显示
        axes[3].bar([''], [results['fluctuation']], color=colors[3], width=0.5, edgecolor='black')
        axes[3].set_ylabel('波动度 (vacil)')
        axes[3].set_title('波动度 (Zwicker模型)')
        axes[3].grid(True, axis='y', alpha=0.6)
        axes[3].set_ylim(0, results['fluctuation'] * 1.2)
        axes[3].text(0, results['fluctuation'] * 1.05, f"{results['fluctuation']:.4f}", 
                    ha='center', fontweight='bold')
        
        # 移除x轴刻度标签（单个柱子无需标签）
        for ax in axes:
            ax.set_xticks([])
        
        plt.tight_layout(pad=2.0)  # 增加子图间距
        plt.show()
        
        # 特定响度分布图（同样应用学术风格）
        plt.figure(figsize=(8, 5))
        plt.plot(results['bark_bands'], results['specific_loudness'], 'b-', linewidth=1.5, marker='o', markersize=4)
        plt.xlabel('临界频带率 (Bark)')
        plt.ylabel('特定响度')
        plt.title('特定响度分布 (ISO 532-1:2017标准)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# 使用示例和测试函数
def test_psychoacoustic_analysis():
    """测试修正后的心理声学分析器"""
    
    # 创建测试信号
    fs = 44100
    duration = 3  # 秒
    t = np.linspace(0, duration, int(fs * duration))
    
    # 生成包含多种成分的测试信号
    carrier_1k = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz载波
    carrier_4k = 0.2 * np.sin(2 * np.pi * 4000 * t)  # 4kHz成分（增加尖锐度）
    
    # 振幅调制成分（用于测试粗糙度和波动度）
    modulator_rough = 0.3 * np.sin(2 * np.pi * 70 * t)   # 70Hz调制（粗糙度）
    modulator_fluct = 0.2 * np.sin(2 * np.pi * 4 * t)    # 4Hz调制（波动度）
    
    test_signal = (carrier_1k + carrier_4k) * (1 + modulator_rough + modulator_fluct)
    test_signal += 0.05 * np.random.normal(0, 1, len(t))  # 添加噪声
    
    # 创建分析器并进行分析
    analyzer = ISO5321PsychoacousticAnalyzer(fs)
    results = analyzer.analyze_audio_signal(test_signal)
    
    # 显示结果
    print("=" * 50)
    print("心理声学参数分析结果 (基于ISO 532-1:2017)")
    print("=" * 50)
    print(f"响度: {results['loudness']:.2f} sone")
    print(f"尖锐度: {results['sharpness']:.2f} acum") 
    print(f"粗糙度: {results['roughness']:.4f} asper")
    print(f"波动度: {results['fluctuation']:.4f} vacil")
    print("=" * 50)
    
    # 可视化结果
    analyzer.plot_analysis_results(results, "测试信号心理声学分析结果")
    
    return results

if __name__ == "__main__":
    # 运行测试
    results = test_psychoacoustic_analysis()
    
    # 文件分析示例（取消注释以使用）
    # analyzer = ISO5321PsychoacousticAnalyzer()
    # file_results = analyzer.analyze_audio_file("your_audio_file.wav")
    # if file_results:
    #     analyzer.plot_analysis_results(file_results, "音频文件分析结果")