import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
    # 标准1/3倍频程中心频率 (ISO标准)
    center_freqs = np.array([
        31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
        630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
        10000, 12500, 16000, 20000
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
    nperseg = min(8192, len(y))  # 段长
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
    
    return np.array(valid_center_freqs), np.array(spl_values)

def calculate_1_3_octave_corrected(audio_path, reference_pressure=20e-6):
    """
    主函数：计算音频文件的1/3倍频程频谱
    
    参数:
        audio_path: 音频文件路径
        reference_pressure: 参考声压 (20µPa)
        
    返回:
        字典，包含中心频率和对应的声压级
    """
    # 读取音频文件
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # 计算1/3倍频程频谱
    center_freqs, spl_values = calculate_one_third_octave_spectrum(y, sr, reference_pressure)
    
    # 转换为字典格式返回
    result = dict(zip(center_freqs, spl_values))
    return result

def test_1khz_sine_wave():
    """测试1: 1kHz正弦波测试"""
    print("测试1: 1kHz正弦波测试 (期望: 1kHz处约90.97dB)")
    print("=" * 60)
    
    # 生成1kHz正弦波，振幅1Pa
    duration = 5.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1Pa振幅的正弦波，有效值 = 1/√2 ≈ 0.7071 Pa
    # 声压级 = 20*log10(0.7071/20e-6) ≈ 20*log10(35355) ≈ 20*4.5485 ≈ 90.97 dB
    amplitude = 1.0
    frequency = 1000
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # 保存为临时文件
    temp_file = "test_1khz_sine.wav"
    # librosa.output.write_wav(temp_file, y, sr)
    import soundfile as sf
    sf.write(temp_file, y, sr)
    # 计算1/3倍频程
    results = calculate_1_3_octave_corrected(temp_file)
    
    # 显示1kHz附近的结果
    target_freqs = [800, 1000, 1250]
    for freq in target_freqs:
        if freq in results:
            spl = results[freq]
            print(f"中心频率 {freq}Hz: {spl:.2f} dB")
            
            if freq == 1000:
                expected = 20 * np.log10((amplitude / np.sqrt(2)) / 20e-6)
                error = abs(spl - expected)
                print(f"  理论值: {expected:.2f} dB")
                print(f"  误差: {error:.2f} dB")
                
                if error < 1.0:
                    print(f"  ✅ 精度良好")
                elif error < 3.0:
                    print(f"  ⚠️ 误差稍大，但可接受")
                else:
                    print(f"  ❌ 误差过大")
    
    # 计算并显示所有频带结果
    print("\n所有频带声压级:")
    print("-" * 40)
    print("频率(Hz)    声压级(dB)")
    print("-" * 20)
    
    sorted_freqs = sorted(results.keys())
    for freq in sorted_freqs:
        spl = results[freq]
        if spl > -np.inf:
            print(f"{freq:>8}    {spl:>8.2f}")
    
    # 绘制频谱图
    plot_spectrum(results, "1kHz正弦波 - 1/3倍频程频谱")
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return results

def test_white_noise():
    """测试2: 白噪声测试"""
    print("\n\n测试2: 白噪声测试 (各频带应有近似相等的能量)")
    print("=" * 60)
    
    # 生成白噪声
    duration = 5.0
    sr = 44100
    n_samples = int(sr * duration)
    
    # 生成高斯白噪声
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 0.1, n_samples)  # 标准差0.1
    
    # 保存为临时文件
    temp_file = "test_white_noise.wav"
    # librosa.output.write_wav(temp_file, white_noise, sr)
    import soundfile as sf
    sf.write(temp_file, white_noise, sr)
    
    # 计算1/3倍频程
    results = calculate_1_3_octave_corrected(temp_file)
    
    # 计算统计信息
    spl_values = [spl for spl in results.values() if spl > -np.inf]
    
    if spl_values:
        mean_spl = np.mean(spl_values)
        std_spl = np.std(spl_values)
        max_spl = max(spl_values)
        min_spl = min(spl_values)
        
        print(f"平均声压级: {mean_spl:.2f} dB")
        print(f"标准差: {std_spl:.2f} dB")
        print(f"最大声压级: {max_spl:.2f} dB")
        print(f"最小声压级: {min_spl:.2f} dB")
        print(f"动态范围: {max_spl - min_spl:.2f} dB")
        
        # 白噪声在各1/3倍频程带应有近似相等的能量
        if std_spl < 6.0:  # 标准1/3倍频程带宽内，白噪声的标准差应小于6dB
            print("✅ 白噪声频带分布均匀性良好")
        else:
            print("⚠️ 白噪声频带分布均匀性一般")
    
    # 绘制频谱图
    plot_spectrum(results, "白噪声 - 1/3倍频程频谱")
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return results

def test_pink_noise():
    """测试3: 粉红噪声测试 (每个倍频程能量相等)"""
    print("\n\n测试3: 粉红噪声测试 (每个倍频程应有近似相等的能量)")
    print("=" * 60)
    
    # 生成粉红噪声
    duration = 5.0
    sr = 44100
    n_samples = int(sr * duration)
    
    # 通过滤波白噪声生成粉红噪声
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 0.1, n_samples)
    
    # 简单的粉红噪声滤波器 (1/f滤波器)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    
    # 应用滤波器
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # 归一化
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.1
    
    # 保存为临时文件
    temp_file = "test_pink_noise.wav"
    # librosa.output.write_wav(temp_file, pink_noise, sr)
    import soundfile as sf
    sf.write(temp_file, pink_noise, sr)
    
    # 计算1/3倍频程
    results = calculate_1_3_octave_corrected(temp_file)
    
    # 计算每倍频程的能量衰减
    print("粉红噪声理论特性: 每倍频程衰减3dB")
    
    # 绘制频谱图
    plot_spectrum(results, "粉红噪声 - 1/3倍频程频谱", is_log=True)
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return results

def plot_spectrum(results, title, is_log=False):
    """绘制1/3倍频程频谱图"""
    if not results:
        print("无数据可绘制")
        return
    
    center_freqs = list(results.keys())
    spl_values = list(results.values())
    
    plt.figure(figsize=(12, 6))
    
    # 过滤无效值
    valid_indices = [i for i, spl in enumerate(spl_values) if spl > -np.inf]
    valid_freqs = [center_freqs[i] for i in valid_indices]
    valid_spl = [spl_values[i] for i in valid_indices]
    
    plt.semilogx(valid_freqs, valid_spl, 'o-', linewidth=2, markersize=8, markerfacecolor='white')
    plt.xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
    plt.ylabel('声压级 (dB SPL)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # 设置x轴刻度
    xticks = [31.5, 100, 315, 1000, 3150, 10000, 20000]
    xtick_labels = ['31.5', '100', '315', '1k', '3.15k', '10k', '20k']
    plt.xticks(xticks, xtick_labels)
    
    if valid_freqs:
        plt.xlim(valid_freqs[0] * 0.8, valid_freqs[-1] * 1.2)
    
    plt.tight_layout()
    plt.show()

def analyze_audio_file(audio_path):
    """分析音频文件"""
    print(f"\n分析音频文件: {audio_path}")
    print("=" * 60)
    
    try:
        # 读取音频信息
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr
        
        print(f"音频信息:")
        print(f"  采样率: {sr} Hz")
        print(f"  时长: {duration:.2f} 秒")
        print(f"  采样点数: {len(y)}")
        print()
        
        # 计算1/3倍频程
        results = calculate_1_3_octave_corrected(audio_path)
        
        # 显示结果
        print("1/3倍频程分析结果:")
        print("-" * 40)
        print("中心频率(Hz)  声压级(dB)")
        print("-" * 20)
        
        valid_spl = []
        for freq in sorted(results.keys()):
            spl = results[freq]
            if spl > -np.inf:
                print(f"{freq:>10.1f}   {spl:>10.2f}")
                valid_spl.append(spl)
        
        if valid_spl:
            mean_spl = np.mean(valid_spl)
            max_spl = max(valid_spl)
            min_spl = min(valid_spl)
            
            print("\n统计信息:")
            print(f"平均声压级: {mean_spl:.2f} dB")
            print(f"最大声压级: {max_spl:.2f} dB (在 {list(results.keys())[valid_spl.index(max_spl)]} Hz)")
            print(f"最小声压级: {min_spl:.2f} dB (在 {list(results.keys())[valid_spl.index(min_spl)]} Hz)")
            print(f"动态范围: {max_spl - min_spl:.2f} dB")
        
        # 绘制频谱
        plot_spectrum(results, f"1/3倍频程频谱 - {audio_path}")
        
        return results
        
    except Exception as e:
        print(f"分析失败: {e}")
        return None

def main():
    """主函数"""
    print("1/3倍频程频谱分析工具")
    print("版本: 修正版 2.0")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了音频文件路径，分析该文件
        audio_file = sys.argv[1]
        analyze_audio_file(audio_file)
    else:
        # 运行所有测试
        print("运行测试套件...\n")
        
        # 测试1: 1kHz正弦波
        print("测试1: 1kHz正弦波验证")
        sine_results = test_1khz_sine_wave()
        
        # 测试2: 白噪声
        print("\n" + "="*60)
        print("测试2: 白噪声均匀性验证")
        noise_results = test_white_noise()
        
        # 测试3: 粉红噪声
        print("\n" + "="*60)
        print("测试3: 粉红噪声验证")
        pink_results = test_pink_noise()
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print("\n使用方法:")
        print("1. 运行测试: python one_third_octave_corrected.py")
        print("2. 分析音频文件: python one_third_octave_corrected.py <音频文件路径>")

if __name__ == "__main__":
    main()
