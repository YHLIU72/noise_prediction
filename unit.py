import os

def normalize_wav_filenames(folder_paths):
    """
    标准化多个文件夹中的WAV文件名，确保在.wav扩展名前添加连字符（-）
    
    参数:
        folder_paths (list): 要处理的文件夹路径列表
    """
    for folder in folder_paths:
        # 检查文件夹是否存在
        if not os.path.isdir(folder):
            print(f"警告: 文件夹 '{folder}' 不存在，已跳过")
            continue
            
        # 递归遍历文件夹中的所有文件
        for root, _, files in os.walk(folder):
            for filename in files:
                # 仅处理WAV文件（不区分大小写）
                if filename.lower().endswith('.wav'):
                    # 分离文件名和扩展名
                    name_part, ext = os.path.splitext(filename)
                    
                    # 检查文件名部分最后是否已有连字符
                    if name_part and name_part[-1] != '-':
                        # 构建新文件名
                        new_filename = f"{name_part}-{ext}"
                        old_path = os.path.join(root, filename)
                        new_path = os.path.join(root, new_filename)
                        
                        # 执行重命名
                        try:
                            os.rename(old_path, new_path)
                            print(f"重命名成功: {old_path} -> {new_path}")
                        except Exception as e:
                            print(f"重命名失败: {old_path}, 错误: {str(e)}")

# 使用示例:
# 处理多个文件夹
normalize_wav_filenames([
    r"E:\lyh\paddlespeech\AD02",
    r"E:\lyh\paddlespeech\H37",
    r"E:\lyh\paddlespeech\GEM",
    r"E:\lyh\paddlespeech\T1X",
    r"E:\lyh\paddlespeech\M1E",
    r"E:\lyh\paddlespeech\KKL",
    r"E:\lyh\paddlespeech\SA5H",
    r"E:\lyh\paddlespeech\H97D",
    r"E:\lyh\paddlespeech\MAR2 2Z",
    r"E:\lyh\paddlespeech\MAR2 EVA2",
    r"E:\lyh\paddlespeech\BYD HTH",
    r"E:\lyh\paddlespeech\E0Y 3Z3M",
    r"E:\lyh\paddlespeech\NU2",
    r"E:\lyh\paddlespeech\X03",
    r"E:\lyh\paddlespeech\MBQ",
    r"E:\lyh\paddlespeech\SRH",
    r"E:\lyh\paddlespeech\MEB",
    r"E:\lyh\paddlespeech\T2X RHD",
])