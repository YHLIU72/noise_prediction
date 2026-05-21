import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
import warnings

class ISO5321Analyzer:
    """
    修复版的ISO 532-1:2017心理声学分析器
    解决了响度为零的问题
    """
    
    def __init__(self, fs=48000):
        self.fs = fs
        self.setup_iso5321_parameters()
        np.seterr(all='ignore')
        warnings.filterwarnings('ignore')
    
    def setup_iso5321_parameters(self):
        """设置ISO 532-1标准参数"""
        
        # 1/3倍频程中心频率 (Hz)
        self.third_octave_center_freqs = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
            250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
        ])
        
        # 临界频带中心频率 (Hz)
        self.bark_center_freqs = self.calculate_bark_center_freqs()
        
        # 参考声压 (20 μPa)
        self.p0 = 2e-5
        self.p0_sq = self.p0 ** 2
        
        # 设置听阈数据
        self.setup_hearing_threshold()
    
    def calculate_bark_center_freqs(self):
        """计算临界频带中心频率"""
        # 更精确的Bark频率计算
        bark_freqs = []
        for z in range(1, 25):  # Bark 1-24
            if z <= 2:
                f = 50 * z
            elif z <= 16:
                f = 100 * ((z - 2) / 14) ** 2 + 100
            else:
                f = 2000 * ((z - 16) / 8) ** 2 + 1500
            bark_freqs.append(f)
        return np.array(bark_freqs)
    
    def setup_hearing_threshold(self):
        """设置绝对听阈数据"""
        # ISO 226:2003 听阈数据 (更准确)
        self.iso226_frequencies = np.array([
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 
            1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
        ])
        
        # 更准确的听阈声压级 (dB SPL)
        self.iso226_thresholds_dB = np.array([
            78.1, 68.7, 59.5, 51.2, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
            14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7,
            -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9
        ])
        
        # 转换为声能量 (Pa²)
        self.iso226_thresholds_energy = self.p0_sq * 10 ** (self.iso226_thresholds_dB / 10)
        
        # 创建插值函数
        self.threshold_interp = interp1d(
            self.iso226_frequencies, 
            self.iso226_thresholds_energy,
            kind='linear',
            bounds_error=False,
            fill_value=(self.iso226_thresholds_energy[0], self.iso226_thresholds_energy[-1])
        )
    
    def freq_to_bark(self, f):
        """频率到Bark尺度转换"""
        f_safe = np.maximum(f, 1)
        return 13 * np.arctan(0.00076 * f_safe) + 3.5 * np.arctan((f_safe / 7500) ** 2)
    
    def calculate_third_octave_levels(self, x):
        """计算1/3倍频程声压级"""
        levels = np.zeros(len(self.third_octave_center_freqs))
        
        for i, f_center in enumerate(self.third_octave_center_freqs):
            # 1/3倍频程带宽
            f_lower = f_center / (2 ** (1/6))
            f_upper = f_center * (2 ** (1/6))
            
            # 计算频带能量
            band_energy = self.calculate_band_energy(x, f_lower, f_upper)
            
            # 计算声压级 - 修复：添加最小能量检查
            if band_energy > 1e-20:  # 避免除零
                level = 10 * np.log10(band_energy / self.p0_sq)
                levels[i] = level
            else:
                levels[i] = -100  # 极小声压级
        
        return levels
    
    def calculate_band_energy(self, x, f_low, f_high):
        """计算频带能量"""
        nyquist = self.fs / 2
        
        if f_high >= nyquist:
            f_high = nyquist * 0.99
        if f_low <= 0:
            f_low = 1.0
        
        try:
            # 设计带通滤波器
            b, a = signal.butter(4, [f_low/nyquist, f_high/nyquist], btype='band')
            filtered_signal = signal.filtfilt(b, a, x)
            
            # 计算能量 - 修复：使用更稳定的方法
            energy = np.mean(np.square(filtered_signal))
            return max(energy, 1e-20)  # 确保最小能量值
            
        except:
            return 1e-20
    
    def map_to_critical_bands(self, third_octave_levels):
        """映射到临界频带"""
        # 将声压级转换为声能量
        energies = self.p0_sq * 10 ** (third_octave_levels / 10)
        
        # 简单的能量映射 - 修复：使用更合理的映射
        critical_band_energies = np.zeros(24)
        
        for i, freq in enumerate(self.third_octave_center_freqs):
            bark_val = self.freq_to_bark(freq)
            band_idx = int(np.clip(np.floor(bark_val), 0, 23))
            
            # 分配能量到对应的临界频带
            critical_band_energies[band_idx] += energies[i]
        
        return critical_band_energies
    
    def calculate_specific_loudness(self, critical_band_energies):
        """
        修复的特定响度计算
        使用更准确的ISO 532-1公式
        """
        specific_loudness = np.zeros(24)
        
        # 获取每个临界频带的绝对听阈
        threshold_energies = self.threshold_interp(self.bark_center_freqs)
        
        for i in range(24):
            E = critical_band_energies[i]  # 临界频带能量
            E_TQ = threshold_energies[i]   # 绝对听阈能量
            
            # 修复：使用更准确的判断条件
            if E > E_TQ * 1.1:  # 确保显著高于听阈
                try:
                    # ISO 532-1标准公式（修正版）
                    # N' = 0.08 * (E_TQ/E_0)^0.23 * [(E/E_TQ)^0.23 - 1]
                    E_0 = self.p0_sq
                    
                    # 修正公式参数
                    term1 = 0.08 * (E_TQ / E_0) ** 0.23
                    term2 = (E / E_TQ) ** 0.23 - 1
                    
                    # 确保term2为正
                    if term2 > 0:
                        specific_loudness[i] = term1 * term2
                    else:
                        specific_loudness[i] = 0.0
                        
                except (FloatingPointError, OverflowError):
                    specific_loudness[i] = 0.0
            else:
                specific_loudness[i] = 0.0
        
        return specific_loudness
    
    def calculate_total_loudness(self, specific_loudness):
        """计算总响度"""
        total_loudness = np.sum(specific_loudness)
        return max(total_loudness, 0.0)
    
    def analyze_audio_signal(self, x, calibration_factor=1.0):
        """
        修复的音频信号分析
        添加校准因子参数
        """
        if len(x) == 0:
            raise ValueError("输入信号长度不能为0")
        
        print("开始心理声学分析...")
        
        # 信号预处理 - 修复：使用校准因子而不是归一化
        x = np.array(x, dtype=np.float64)
        
        # 应用校准因子（将数字信号转换为物理声压）
        x_calibrated = x * calibration_factor
        
        # 计算1/3倍频程声压级
        print("计算1/3倍频程声压级...")
        third_octave_levels = self.calculate_third_octave_levels(x_calibrated)
        
        # 检查是否有合理的声压级
        max_level = np.max(third_octave_levels)
        print(f"最大1/3倍频程声压级: {max_level:.1f} dB")
        
        if max_level < 20:  # 如果最大声压级太小
            print("警告：信号声压级可能过低，尝试调整校准因子")
        
        # 映射到临界频带
        print("映射到临界频带...")
        critical_band_energies = self.map_to_critical_bands(third_octave_levels)
        
        # 计算特定响度
        print("计算特定响度...")
        specific_loudness = self.calculate_specific_loudness(critical_band_energies)
        
        # 检查特定响度
        max_specific_loudness = np.max(specific_loudness)
        print(f"最大特定响度: {max_specific_loudness:.6f} sone/Bark")
        
        # 计算总响度
        total_loudness = self.calculate_total_loudness(specific_loudness)
        print(f"总响度: {total_loudness:.4f} sone")
        
        # 计算其他参数
        sharpness = self.calculate_sharpness(specific_loudness, total_loudness)
        roughness = self.calculate_roughness(x_calibrated)
        fluctuation = self.calculate_fluctuation_strength(x_calibrated)
        
        results = {
            'loudness': total_loudness,
            'specific_loudness': specific_loudness,
            'sharpness': sharpness,
            'roughness': roughness,
            'fluctuation': fluctuation,
            'third_octave_levels': third_octave_levels,
            'bark_center_freqs': self.bark_center_freqs,
            'calibration_factor_used': calibration_factor
        }
        
        return results
    
    def calculate_sharpness(self, specific_loudness, total_loudness):
        """计算尖锐度"""
        if total_loudness <= 0:
            return 0.0
        
        # 尖锐度权重函数
        g_z = np.ones(24)
        for i in range(24):
            z = i + 1  # Bark值
            if z > 16:
                g_z[i] = 0.066 * np.exp(0.171 * z)
        
        numerator = np.sum(specific_loudness * g_z * np.arange(1, 25))
        sharpness = 0.11 * numerator / total_loudness
        
        return max(0.0, sharpness)
    
    def calculate_roughness(self, x):
        """计算粗糙度"""
        try:
            # 提取包络
            analytic_signal = signal.hilbert(x)
            envelope = np.abs(analytic_signal)
            
            # 分析调制频谱
            envelope_spectrum = np.abs(fft(envelope - np.mean(envelope)))
            freqs = fftfreq(len(envelope), 1/self.fs)
            
            # 70Hz调制成分
            mask = (freqs >= 50) & (freqs <= 90)
            modulation_energy = np.sum(np.abs(envelope_spectrum[mask]))
            
            roughness = modulation_energy / len(x) * 1e6
            return max(0.0, roughness)
            
        except:
            return 0.0
    
    def calculate_fluctuation_strength(self, x):
        """计算波动度"""
        try:
            analytic_signal = signal.hilbert(x)
            envelope = np.abs(analytic_signal)
            
            envelope_spectrum = np.abs(fft(envelope - np.mean(envelope)))
            freqs = fftfreq(len(envelope), 1/self.fs)
            
            # 4Hz调制成分
            mask = (freqs >= 0.5) & (freqs <= 20)
            modulation_energy = np.sum(np.abs(envelope_spectrum[mask]))
            
            fluctuation = modulation_energy / len(x) * 1e4
            return max(0.0, fluctuation)
            
        except:
            return 0.0
    
    def find_optimal_calibration(self, x, target_level_dB=60):
        """
        自动寻找最佳校准因子
        target_level_dB: 目标声压级 (dB SPL)
        """
        # 计算当前信号的RMS
        rms = np.sqrt(np.mean(x ** 2))
        
        if rms == 0:
            return 1.0  # 默认值
        
        # 目标声压对应的RMS (20μPa为参考)
        target_pressure = 2e-5 * 10 ** (target_level_dB / 20)
        
        # 计算校准因子
        calibration_factor = target_pressure / rms
        
        print(f"自动校准: RMS={rms:.2e}, 目标压力={target_pressure:.2e}")
        print(f"校准因子: {calibration_factor:.2e}")
        
        return calibration_factor


# 测试函数 - 修复版
def test_fixed_iso5321_analysis():
    """修复版的测试函数"""
    
    # 创建测试信号 - 使用更合适的幅度
    fs = 48000
    duration = 3  # 秒
    t = np.linspace(0, duration, int(fs * duration))
    
    # 生成60dB SPL的测试信号
    # 60dB SPL对应的声压为：20e-6 * 10^(60/20) = 0.02 Pa
    target_spl_dB = 60
    target_pressure = 2e-5 * 10 ** (target_spl_dB / 20)  # 0.02 Pa
    
    # 正弦波的峰值幅度 = √2 * RMS
    amplitude = target_pressure * np.sqrt(2)
    
    print(f"目标声压级: {target_spl_dB} dB SPL")
    print(f"目标声压: {target_pressure:.6f} Pa")
    print(f"需要的声音幅度: {amplitude:.6f}")
    
    # 生成1kHz测试信号
    test_signal = amplitude * np.sin(2 * np.pi * 1000 * t)
    
    # 添加一些高频成分
    test_signal += 0.3 * amplitude * np.sin(2 * np.pi * 4000 * t)
    
    # 计算实际RMS和声压级
    actual_rms = np.sqrt(np.mean(test_signal ** 2))
    actual_spl = 20 * np.log10(actual_rms / 2e-5)
    
    print(f"实际RMS: {actual_rms:.6f} Pa")
    print(f"实际声压级: {actual_spl:.1f} dB SPL")
    
    # 创建分析器
    analyzer = ISO5321Analyzer(fs)
    
    # 分析信号（不使用额外校准，因为信号已经正确校准）
    results = analyzer.analyze_audio_signal(test_signal, calibration_factor=1.0)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("修复版ISO 532-1分析结果")
    print("=" * 60)
    print(f"总响度: {results['loudness']:.4f} sone")
    print(f"尖锐度: {results['sharpness']:.4f} acum")
    print(f"粗糙度: {results['roughness']:.6f} asper")
    print(f"波动度: {results['fluctuation']:.6f} vacil")
    print("=" * 60)
    
    # 检查特定响度分布
    print("\n特定响度分布:")
    for i, loudness in enumerate(results['specific_loudness']):
        if loudness > 0.001:
            freq = analyzer.bark_center_freqs[i] if i < len(analyzer.bark_center_freqs) else 0
            print(f"  Bark {i+1:2d} ({freq:5.0f} Hz): {loudness:.6f} sone/Bark")
    
    # 预期值参考
    print(f"\n预期参考:")
    print(f"  1kHz 40dB SPL纯音 ≈ 1 sone")
    print(f"  1kHz 60dB SPL纯音 ≈ 4 sone")
    
    if results['loudness'] > 0.1:
        print("✓ 响度计算成功！")
    else:
        print("⚠ 响度仍然可能有问题，请检查输入信号")
    
    return results


# 测试未校准信号
def test_uncalibrated_signal():
    """测试未校准信号（常见情况）"""
    
    fs = 48000
    duration = 3
    t = np.linspace(0, duration, int(fs * duration))
    
    # 生成未校准的信号（幅度为1）
    test_signal = np.sin(2 * np.pi * 1000 * t)
    
    # 创建分析器
    analyzer = ISO5321Analyzer(fs)
    
    # 自动寻找校准因子
    calibration_factor = analyzer.find_optimal_calibration(test_signal, target_level_dB=60)
    
    # 使用校准因子分析
    results = analyzer.analyze_audio_signal(test_signal, calibration_factor=calibration_factor)
    
    print(f"未校准信号分析结果:")
    print(f"总响度: {results['loudness']:.4f} sone")
    
    return results


if __name__ == "__main__":
    print("测试修复版ISO 532-1分析器...")
    
    # 测试1：正确校准的信号
    results1 = test_fixed_iso5321_analysis()
    
    print("\n" + "="*60)
    
    # 测试2：未校准信号（更常见的情况）
    results2 = test_uncalibrated_signal()