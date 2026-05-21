import os
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
import pandas as pd
from typing import Optional, Tuple

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加wavenet_vocoder到Python路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'wavenet_vocoder'))


def load_condition(mel_path: str) -> np.ndarray:
    """
    读取保存的条件特征 .npy （形状 (T, 80)），并返回 float32 numpy 数组。
    如果需要，可在此处添加与训练时一致的标准化/反标准化逻辑。
    """
    c = np.load(mel_path)
    if c.ndim != 2 or c.shape[1] != 80:
        raise ValueError(f"期望条件形状为 (T, 80)，收到 {c.shape} 来自 {mel_path}")
    return c.astype(np.float32)


def compute_mel_from_wav(audio_path: str,
                         n_mels: int = 80,
                         fft_window: int = 1280,
                         hop_length: int = 320) -> np.ndarray:
    """
    使用 librosa 计算 mel 频谱，与训练保持一致。
    返回形状 (T, 80)。
    """
    # 使用 librosa 加载音频
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 计算 mel 频谱
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=n_mels,
        n_fft=fft_window,
        hop_length=hop_length,
        fmin=125,
        fmax=7600
    )
    
    # 转换为对数尺度
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # 转置为 (T, 80) 格式
    mel = mel.T.astype(np.float32)
    
    return mel


def load_condition_from_csv(csv_path: str,
                            row_index: Optional[int] = None,
                            params: Optional[Tuple[float, float, float]] = None,
                            n_mels: int = 80,
                            fft_window: int = 1280,
                            hop_length: int = 320,
                            tol: float = 1e-6) -> np.ndarray:
    """
    从 CSV 中定位工况参数对应的音频路径，计算真实 mel。
    - 若提供 params=(p5,p6,p7)，将在第5-7列中近似匹配；否则使用 row_index 定位。
    返回 (T,80) 的 numpy 数组。
    """
    df = pd.read_csv(csv_path)
    if params is not None:
        p = np.asarray(params, dtype=float)
        if p.shape != (3,):
            raise ValueError("params 必须是包含3个数的元组/列表，如: --params 1.0,2.0,3.0")
        # 在第 5-7 列匹配（iloc[4:7]）
        mask = (np.abs(df.iloc[:, 4] - p[0]) <= tol) & \
               (np.abs(df.iloc[:, 5] - p[1]) <= tol) & \
               (np.abs(df.iloc[:, 6] - p[2]) <= tol)
        cand = df[mask]
        if len(cand) == 0:
            raise ValueError("未在 CSV 中找到匹配的工况参数行")
        row = cand.iloc[0]
    else:
        if row_index is None:
            raise ValueError("必须提供 --row-index 或 --params 之一")
        row = df.iloc[row_index]
    audio_path = row.iloc[0]
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    return compute_mel_from_wav(audio_path, n_mels, fft_window, hop_length)


def synthesize_with_wavenet(mel_npy: str, ckpt: str, preset: str, out_wav: str, sr: int = 22050, row_index: int = None, params: tuple = None, duration_sec: float = None, extend_condition: bool = True):
    """
    使用本地构建与训练一致的方式进行合成：
    - 解析 hparams
    - 按训练时方式构建 WaveNet
    - incremental_forward 合成波形
    """
    from wavenet_vocoder import WaveNet
    from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_scalar_input
    from hparams import hparams as hp
    from nnmnkwii import preprocessing as P

    # 载入超参
    if preset is not None and os.path.exists(preset):
        with open(preset, "r", encoding="utf-8") as f:
            hp.parse_json(f.read())

    # 条件特征 (T,80) -> (1,80,T)
    c = load_condition(mel_npy)
    
    # 如果需要生成更长时间的音频，扩展条件特征
    if duration_sec is not None:
        original_duration = c.shape[0] * hp.hop_size / hp.sample_rate
        if duration_sec > original_duration:
            # 计算需要重复多少次条件特征
            repeat_factor = int(duration_sec / original_duration) + 1
            # 重复条件特征
            c_extended = np.tile(c, (repeat_factor, 1))
            # 截取到目标长度
            target_frames = int(duration_sec * hp.sample_rate / hp.hop_size)
            c = c_extended[:target_frames]
            print(f"扩展条件特征: {c.shape[0]} 帧 -> {target_frames} 帧")
            print(f"目标时长: {duration_sec:.2f}秒")
        else:
            # 如果目标时长更短，截取条件特征
            target_frames = int(duration_sec * hp.sample_rate / hp.hop_size)
            c = c[:target_frames]
            print(f"截取条件特征: {c.shape[0]} 帧 -> {target_frames} 帧")
    
    c_t = torch.from_numpy(c.T).unsqueeze(0).float()

    # 构建模型（与 train.py 的 build_model 一致）
    upsample_params = dict(hp.upsample_params)
    upsample_params["cin_channels"] = hp.cin_channels
    upsample_params["cin_pad"] = hp.cin_pad
    model = WaveNet(
        out_channels=hp.out_channels,
        layers=hp.layers,
        stacks=hp.stacks,
        residual_channels=hp.residual_channels,
        gate_channels=hp.gate_channels,
        skip_out_channels=hp.skip_out_channels,
        cin_channels=hp.cin_channels,
        gin_channels=hp.gin_channels,
        n_speakers=hp.n_speakers,
        dropout=hp.dropout,
        kernel_size=hp.kernel_size,
        cin_pad=hp.cin_pad,
        upsample_conditional_features=hp.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hp.input_type),
        output_distribution=hp.output_distribution,
    )

    # 加载权重
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    # 提升推理速度（移除权重归一化等）
    try:
        model.make_generation_fast_()
    except Exception:
        pass

    # 初始输入
    if is_mulaw_quantize(hp.input_type):
        initial_value = P.mulaw_quantize(0, hp.quantize_channels - 1)
        initial_input = torch.from_numpy(
            P.to_categorical(initial_value, num_classes=hp.quantize_channels).astype(np.float32)
        ).view(1, 1, hp.quantize_channels)
    elif is_mulaw(hp.input_type):
        initial_value = P.mulaw(0.0, hp.quantize_channels)
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)
    else:
        initial_value = 0.0
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    # 需要生成的采样点数：按 hop_size * 帧数 粗略估计
    T_samples = int(c_t.size(-1) * hp.hop_size)

    with torch.no_grad():
        from tqdm import tqdm as tq
        y_hat = model.incremental_forward(
            initial_input, c=c_t, g=None, T=T_samples, softmax=True,
            quantize=is_mulaw_quantize(hp.input_type), tqdm=tq,
            log_scale_min=hp.log_scale_min,
        )

    # 反量化/反 mu-law
    if is_mulaw_quantize(hp.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hp.quantize_channels - 1)
    elif is_mulaw(hp.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hp.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # 归一化并保存
    y_hat = np.asarray(y_hat, dtype=np.float32)
    if np.max(np.abs(y_hat)) > 0:
        y_hat = y_hat / np.max(np.abs(y_hat))
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    sf.write(out_wav, y_hat, sr)
    
    print(f"音频已保存: {out_wav}")
    print(f"使用checkpoint: {os.path.basename(ckpt)}")
    if row_index is not None:
        print(f"工况行号: {row_index}")
    if params is not None:
        print(f"工况参数: Qv={params[0]}, DP={params[1]}, N={params[2]}")


def main():
    parser = argparse.ArgumentParser(description="Use wavenet_vocoder to synthesize wavs from mel.")
    # 方式一：直接传入 .npy
    parser.add_argument("--mel", required=False, help="输入mel条件 .npy 文件，形状 (T,80)")
    # 方式二：从 CSV 与工况参数/行号计算真实 mel
    parser.add_argument("--csv", required=False, help="CSV 文件路径（第一列为音频路径，第5-7列为工况参数）")
    parser.add_argument("--row-index", type=int, required=False, help="CSV 行号（从 0 开始）")
    parser.add_argument("--params", required=False, help="用逗号分隔的三个工况参数，例如 1.0,2.0,3.0")
    # 方式三：使用 model.py 预测 mel
    parser.add_argument("--use-predictor", action="store_true", default=True, help="使用 model.py 中的预测函数根据 --params 直接生成 mel（默认启用）")
    parser.add_argument("--no-predictor", action="store_true", help="禁用预测接口，回退到 CSV 或 --mel 方式")
    parser.add_argument("--predictor-path", required=False, default="model.py", help="包含预测函数的模块路径，默认 model.py")
    parser.add_argument("--predictor-func", required=False, default="predict_mel", help="预测函数名，签名为 f(qv, dp, rpm) -> np.ndarray[(T,80)]")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--fft-window", type=int, default=1280)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--ckpt", required=False, default=None, help="WaveNet checkpoint 路径（或设置环境变量 WAVENET_CKPT）")
    parser.add_argument("--preset", required=False, default=None, help="hparams 预设 JSON 文件（或设置环境变量 WAVENET_PRESET）")
    parser.add_argument("--out", required=False, default="out.wav", help="输出 wav 路径")
    parser.add_argument("--sr", type=int, default=22050, help="输出采样率")
    parser.add_argument("--duration", "-d", type=float, help="生成音频时长（秒），如果不指定则使用原始mel谱长度")
    parser.add_argument("--extend-condition", action="store_true", default=True, help="是否通过重复条件特征来扩展时长")
    args = parser.parse_args()

    # 若未指定 CSV，尝试使用项目根目录下的默认 CSV 名称
    if not args.csv and os.path.exists("MAR2 EVA2.csv"):
        args.csv = "MAR2 EVA2.csv"
    # 若未提供行号且未提供 params，默认取第 0 行
    if args.row_index is None and not args.params:
        args.row_index = 0

    # 解析 ckpt 与 preset，允许从环境变量获取
    ckpt = args.ckpt or os.getenv("WAVENET_CKPT")
    preset = args.preset or os.getenv("WAVENET_PRESET")
    if not ckpt:
        raise ValueError("请提供 --ckpt 或设置环境变量 WAVENET_CKPT 指向 WaveNet checkpoint")

    # 自动生成输出文件名（基于checkpoint名称）
    if args.out == "out.wav":  # 默认输出文件名
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]  # 去掉扩展名
        if args.row_index is not None:
            args.out = f"{ckpt_name}_row{args.row_index}.wav"
        elif args.params:
            params_str = "_".join([f"{float(p):.1f}" for p in args.params.split(',')])
            args.out = f"{ckpt_name}_params_{params_str}.wav"
        else:
            args.out = f"{ckpt_name}_generated.wav"

    mel_path = None
    params_tuple = None
    # 优先：使用 predictor 直接根据参数生成 mel（默认启用）
    if args.use_predictor and not args.no_predictor:
        if not args.params:
            raise ValueError("--use-predictor 需要同时提供 --params，例如 --params \"Qv,DP,N\"")
        parts = [s.strip() for s in args.params.split(',') if s.strip() != ""]
        if len(parts) != 3:
            raise ValueError("--params 需要三个以逗号分隔的数值，如 1.0,2.0,3.0")
        params_tuple = (float(parts[0]), float(parts[1]), float(parts[2]))
        # 动态导入 predictor
        import importlib.util
        spec = importlib.util.spec_from_file_location("predictor_mod", args.predictor_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从 {args.predictor_path} 加载预测模块")
        predictor_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor_mod)
        if not hasattr(predictor_mod, args.predictor_func):
            raise AttributeError(f"{args.predictor_path} 中未找到函数 {args.predictor_func}")
        predict_fn = getattr(predictor_mod, args.predictor_func)
        mel_arr = predict_fn(params_tuple[0], params_tuple[1], params_tuple[2])
        mel_arr = np.asarray(mel_arr, dtype=np.float32)
        if mel_arr.ndim != 2 or mel_arr.shape[1] != 80:
            raise ValueError(f"预测函数应返回形状 (T,80) 的数组，实际为 {mel_arr.shape}")
        # 写入临时 npy 以复用后续流程
        os.makedirs(".cache", exist_ok=True)
        mel_path = os.path.join(".cache", "cond.npy")
        np.save(mel_path, mel_arr)
    else:
        # 常规：使用 CSV 定位并计算真实 mel
        if args.mel:
            mel_path = args.mel
        else:
            if not args.csv:
                raise ValueError("未提供 --mel，需提供 --csv 并使用 --row-index 或 --params 选择一行")
            if args.params:
                parts = [s.strip() for s in args.params.split(',') if s.strip() != ""]
                if len(parts) != 3:
                    raise ValueError("--params 需要三个以逗号分隔的数值，如 1.0,2.0,3.0")
                params_tuple = (float(parts[0]), float(parts[1]), float(parts[2]))
            mel_arr = load_condition_from_csv(
                args.csv,
                row_index=args.row_index,
                params=params_tuple,
                n_mels=args.n_mels,
                fft_window=args.fft_window,
                hop_length=args.hop_length,
            )  # (T,80)
            # 写入临时 npy 以复用后续流程
            os.makedirs(".cache", exist_ok=True)
            mel_path = os.path.join(".cache", "cond.npy")
            np.save(mel_path, mel_arr)

    synthesize_with_wavenet(mel_path, ckpt, preset, args.out, args.sr, args.row_index, params_tuple, args.duration, args.extend_condition)


if __name__ == "__main__":
    main()


