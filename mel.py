import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def compute_mel_spectrogram(audio_path, sr=None, n_fft=1024, hop_length=256, n_mels=64, repeat=1):
    """
    计算音频的梅尔频谱图（分贝刻度），支持音频复制拼接
    
    参数:
        audio_path (str): 音频文件路径
        sr (int): 采样率，None表示使用音频原采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        n_mels (int): 梅尔滤波器数量
        repeat (int): 音频复制拼接次数，默认为1（不复制）
        
    返回:
        np.ndarray: 梅尔频谱图数组 (n_mels, t)
    """
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 复制音频并拼接
    if repeat > 1:
        y = np.concatenate([y] * repeat)  # 复制repeat次并拼接
    
    # 验证音频总时长
    duration = librosa.get_duration(y=y, sr=sr)
    expected_duration = 0.25 * repeat  # 原音频0.25秒，复制repeat次后的期望时长
    if not np.isclose(duration, expected_duration, atol=0.001):
        raise ValueError(f"音频总时长必须为{expected_duration}秒，实际时长: {duration:.4f}秒")
    
    # ... 以下为原有代码，保持不变 ...
    # 计算梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # 转换为分贝刻度 (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
def plot_mel_spectrogram(mel_spec_db, title, sr=22050, hop_length=256, save_path=None, show=True):
    """绘制梅尔频谱图并可选显示和保存
    
    参数:
        mel_spec_db (np.ndarray): 梅尔频谱图数组
        title (str): 图像标题
        sr (int): 采样率
        hop_length (int): 帧移大小
        save_path (str): 图像保存路径，为None则不保存
        show (bool): 是否显示图像
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # dpi控制图像分辨率
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像释放内存

def process_audio_folder(input_folder, output_folder=None, show_plots=True, max_plots=5, **mel_kwargs):
    """
    批量处理文件夹内所有音频文件，计算并保存梅尔频谱图
    
    参数:
        input_folder (str): 输入音频文件夹路径
        output_folder (str): 输出结果保存路径，默认在输入文件夹下创建'mel_output'
        show_plots (bool): 是否显示梅尔频谱图，默认为True
        max_plots (int): 最大显示图像数量，默认为5
        **mel_kwargs: 传递给compute_mel_spectrogram的参数
    """
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'mel_output')
    os.makedirs(output_folder, exist_ok=True)
     # 创建梅尔频谱图图像保存目录
    mel_plots_folder = os.path.join(input_folder, 'mel_plots')
    os.makedirs(mel_plots_folder, exist_ok=True)
    
    # 支持的音频文件格式
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma')
    
    # 遍历文件夹内所有文件
    plot_count = 0  # 计数已显示的图像数量
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(audio_extensions):
            audio_path = os.path.join(input_folder, filename)
            try:
                # 计算梅尔频谱图
                mel_spec_db = compute_mel_spectrogram(audio_path, **mel_kwargs)
                print(f"{filename} 的梅尔谱形状: {mel_spec_db.shape}")  # (n_mels, 时间帧数)
                
                # 保存结果为NumPy数组 (.npy)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}_mel.npy")
                np.save(output_path, mel_spec_db)
                # 获取采样率用于绘图
                y, sr = librosa.load(audio_path, sr=mel_kwargs.get('sr'))
                
                # 保存梅尔频谱图为图像文件
                plot_save_path = os.path.join(mel_plots_folder, f"{base_name}_mel.png")
                plot_mel_spectrogram(
                    mel_spec_db, 
                    title=f"梅尔频谱图 - {filename}",
                    sr=sr,
                    hop_length=mel_kwargs.get('hop_length', 256),
                    save_path=plot_save_path,
                    show=show_plots and (plot_count < max_plots)  # 控制是否显示
                )
                
                # 更新显示计数
                if show_plots and (plot_count < max_plots):
                    plot_count += 1
            
                print(f"成功处理: {filename} -> 数组文件: {os.path.basename(output_path)}, 图像文件: {os.path.basename(plot_save_path)}")
            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")
                
if __name__ == "__main__":
    # ==================== 需要用户修改的参数 ====================
    input_folder = "./MAR2 EVA2"  # 音频文件夹路径
    output_folder = "./MAR2 EVA2/mel_output"
    # output_folder = "E:/path/to/save/mel_results"  # 可选：自定义输出路径
    # ==========================================================
    
    # 调用批量处理函数（添加repeat参数控制复制次数）
    process_audio_folder(
        input_folder=input_folder,
        # output_folder=output_folder,  # 取消注释以使用自定义输出路径
        n_fft=1280,       # FFT窗口大小
        hop_length=320,  # 帧移大小
        n_mels=80,       # 梅尔滤波器数量
        max_plots=5,
        repeat=1          # 音频复制次数（0.25秒×40=10秒总时长）
    )