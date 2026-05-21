import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.io.wavfile as wavfile
import os

class AudioSpectrumComparator:
    def __init__(self):
        """初始化音频频谱比较器"""
        pass
    
    def load_audio(self, file_path):
        """加载音频文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        try:
            sr, data = wavfile.read(file_path)
            # 如果是立体声，转换为单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            return data, sr
        except Exception as e:
            raise Exception(f"加载音频文件失败: {str(e)}")
    
    def compute_spectrum(self, audio_data, sr):
        """计算最普通的频谱图（完整傅里叶变换）"""
        # 计算傅里叶变换
        n = len(audio_data)
        freq_data = np.fft.fft(audio_data)
        # 计算频率轴
        freq_axis = np.fft.fftfreq(n, d=1.0/sr)
        # 只取正频率部分
        positive_freq_indices = np.where(freq_axis > 0)
        freq_axis = freq_axis[positive_freq_indices]
        # 计算幅度谱
        amplitude = np.abs(freq_data[positive_freq_indices])
        # 转换为分贝刻度
        amplitude_db = 20 * np.log10(amplitude + 1e-10)  # 添加小值避免对数运算错误
        
        return freq_axis, amplitude_db
    
    def compare_spectrums(self, file1_path, file2_path, output_image="spectrum_comparison.png", title="音频频谱对比"):
        """比较两个音频文件的频谱并保存对比图像"""
        # 加载两个音频文件
        data1, sr1 = self.load_audio(file1_path)
        data2, sr2 = self.load_audio(file2_path)
        
        # 计算频谱
        freq1, amp1 = self.compute_spectrum(data1, sr1)
        freq2, amp2 = self.compute_spectrum(data2, sr2)
        
        # 创建对比图像
        plt.figure(figsize=(14, 8))
        
        # 第一个音频的频谱
        plt.subplot(2, 1, 1)
        plt.plot(freq1, amp1)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度 (dB)')
        plt.title(f'音频1: {os.path.basename(file1_path)}')
        plt.grid(True)
        plt.tight_layout()
        
        # 第二个音频的频谱
        plt.subplot(2, 1, 2)
        plt.plot(freq2, amp2)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度 (dB)')
        plt.title(f'音频2: {os.path.basename(file2_path)}')
        plt.grid(True)
        plt.tight_layout()
        
        # 设置总标题
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        # 保存图像
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"频谱对比图像已保存至: {output_image}")
        
        # 显示图像
        plt.show()
        
        return output_image

if __name__ == "__main__":
    # 示例用法
    comparator = AudioSpectrumComparator()
    
    # 请替换为您的音频文件路径
    file1 = "E:\lyh\paddlespeech\T2X RHD\CVAF-167-20-.wav"  # 第一个音频文件路径
    file2 = "E:\lyh\paddlespeech\checkpoint_step000015357_params_167.0_-19.0_1353.0.wav"  # 第二个音频文件路径
    
    # 比较频谱并生成对比图像
    comparator.compare_spectrums(file1, file2, output_image="spectrum_comparison.png")