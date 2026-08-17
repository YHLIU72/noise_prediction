"""
v10-v14 模型综合评估脚本
- 从 TensorBoard 提取训练历程
- 分层抽样统一验证集评估
- 处理不同频率网格 (线性/对数, 20~5000/60~5000Hz)
- 生成详细指标 + CSV表格 + 对比报告
"""
import os, sys, json, time, warnings, glob
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
# [修复] Windows GBK编码兼容：强制UTF-8输出
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import torch, numpy as np, pandas as pd, ast, importlib.util
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# TensorBoard 数据提取
# ============================================================
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = r"f:\lyh\paddlespeech\papernoise\physic"
runs_base = os.path.join(base, "runs")

VERSIONS_TB = OrderedDict([
    ("V10", "pi_mbcn_v10_20260614_195124"),
    ("V11", "pi_mbcn_v11_20260615_141627"),
    ("V12", "pi_mbcn_v12_20260615_141720"),
    ("V13", "pi_mbcn_v13_20260615_141745"),
    ("V14", "pi_mbcn_v14_20260615_141801"),
])

tb_info = {}
print("="*60)
print("TensorBoard 训练历程提取")
print("="*60)
for vname, run_dir in VERSIONS_TB.items():
    run_path = os.path.join(runs_base, run_dir)
    ef = glob.glob(os.path.join(run_path, 'events.out.*'))
    if not ef:
        print(f"  {vname}: NO EVENTS FILE")
        continue
    try:
        ea = EventAccumulator(ef[0], size_guidance={'scalars': 0})
        ea.Reload()
    except Exception as e:
        print(f"  {vname}: RELOAD ERROR: {e}")
        continue
    tags = ea.Tags().get('scalars', [])
    val_tags = [t for t in tags if 'val_' in t.lower()]
    
    info = {'run_dir': run_dir, 'tags': tags, 'val_tags': val_tags, 'metrics': {}}
    for tag in val_tags:
        try:
            events = ea.Scalars(tag)
            vals_list = [(e.step, e.value) for e in events]
            if vals_list:
                min_idx = np.argmin([v[1] for v in vals_list])
                min_step, min_val = vals_list[min_idx]
                latest_step, latest_val = vals_list[-1]
                info['metrics'][tag] = {
                    'best': min_val, 'best_epoch': min_step,
                    'latest': latest_val, 'latest_epoch': latest_step,
                    'total': len(vals_list)
                }
        except:
            pass
    
    # Checkpoint info
    ckpt_path = os.path.join(run_path, 'models', 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        info['ckpt_epoch'] = ckpt.get('epoch', '?')
        info['ckpt_best_val_loss'] = ckpt.get('best_val_loss', '?')
    
    tb_info[vname] = info
    print(f"  {vname}: {len(val_tags)} val tags, ckpt_epoch={info.get('ckpt_epoch','?')}")
    
    # Print key metrics
    for tag in sorted(info['metrics'].keys()):
        m = info['metrics'][tag]
        print(f"    {tag}: best={m['best']:.4f} @ep{m['best_epoch']}, latest={m['latest']:.4f} @ep{m['latest_epoch']}")

# ============================================================
# 统一数据集 (分层抽样, 固定seed)
# ============================================================
print("\n" + "="*60)
print("构建统一验证集")
print("="*60)

class UnifiedDataset(Dataset):
    """统一数据集: 加载原始CSV, 分层抽样划分train/val
    
    [V13修复] 可通过 norm_params 传入训练时的归一化参数,
    避免因 CSV 加载顺序不同导致评估与训练不一致。
    """
    def __init__(self, directory_path, is_train=True, val_split=0.2, seed=42,
                 freq_range='20-5000', log_sample=False, norm_params=None):
        # [修复] 使用 sorted() 确保 CSV 加载顺序确定性, 与训练代码保持一致
        csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv')])
        dfs = [pd.read_csv(os.path.join(directory_path, f)) for f in csv_files]
        self.data = pd.concat(dfs, ignore_index=True)
        self.n_total = len(self.data)
        self.inputs = self.data.iloc[:, [4,5,6]].values.astype(np.float32)
        self.types_raw = self.data.iloc[:, 3].values.astype(str)
        self.modes_raw = self.data.iloc[:, 2].values.astype(str)
        self.type_enc = LabelEncoder().fit(self.types_raw)
        self.mode_enc = LabelEncoder().fit(self.modes_raw)
        self.types = self.type_enc.transform(self.types_raw).astype(np.int64)
        self.modes = self.mode_enc.transform(self.modes_raw).astype(np.int64)
        
        # 解析频谱
        spectra_raw = []
        for val in self.data.iloc[:, 13]:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            arr = np.array(parsed, dtype=np.float32)
            if len(arr) >= 2501: arr = arr[:2501]
            # [修复] 零填充, 与训练代码 PIMBCN_data_0614_v13.py 保持一致
            else: arr = np.array(list(arr) + [0.0] * (2501 - len(arr)), dtype=np.float32)
            spectra_raw.append(arr)
        spectra_raw = np.array(spectra_raw, dtype=np.float32)
        
        # 频率范围处理
        self.freq_range = freq_range
        self.log_sample = log_sample
        
        if freq_range == '20-5000':
            self.spectra = spectra_raw[:, 5:1251]  # 1246 points
            self.freq_axis = np.linspace(20, 5000, 1246)
            self.freq_bins = 1246
        elif freq_range == '60-5000':
            self.spectra = spectra_raw[:, 15:1251]  # 1236 points
            self.freq_axis = np.linspace(60, 5000, 1236)
            self.freq_bins = 1236
        
        # 对数重采样 (V13)
        if log_sample:
            log_freqs = np.logspace(np.log10(20), np.log10(5000), 1246)
            log_spectra = np.zeros((self.n_total, 1246), dtype=np.float32)
            linear_freqs = np.linspace(20, 5000, 1246)
            for i in range(self.n_total):
                log_spectra[i] = np.interp(log_freqs, linear_freqs, spectra_raw[i, 5:1251])
            self.spectra = log_spectra
            self.freq_axis = log_freqs
            self.freq_bins = 1246
        
        # 分层抽样
        np.random.seed(seed)
        combo_labels = self.modes * 1000 + self.types
        unique_combos = np.unique(combo_labels)
        combo_indices, combo_sizes = [], []
        for combo in unique_combos:
            c_idx = np.where(combo_labels == combo)[0]
            np.random.shuffle(c_idx)
            combo_indices.append(c_idx); combo_sizes.append(len(c_idx))
        target_val = int(self.n_total * val_split)
        val_counts = []
        for size in combo_sizes:
            count = max(1, int(np.round(size * val_split)))
            count = min(count, size // 2)
            val_counts.append(count)
        total_assigned = sum(val_counts)
        if total_assigned > target_val:
            ratio = target_val / total_assigned
            val_counts = [max(1, int(np.round(c * ratio))) for c in val_counts]
        train_idx_list, val_idx_list = [], []
        for idx_c, n_val in zip(combo_indices, val_counts):
            if len(idx_c) == 1: train_idx_list.extend(idx_c)
            else:
                val_idx_list.extend(idx_c[:n_val])
                train_idx_list.extend(idx_c[n_val:])
        val_indices = np.array(val_idx_list, dtype=np.int64)
        train_indices = np.array(train_idx_list, dtype=np.int64)
        self.indices = train_indices if is_train else val_indices
        
        # 归一化（优先使用外部传入的归一化参数，否则从训练集统计量计算）
        if norm_params is not None:
            self.input_mean = norm_params['input_mean']
            self.input_std = norm_params['input_std']
            print(f"  [归一化] 使用外部传入的 norm_params (来自训练checkpoint)")
        else:
            self.input_mean = self.inputs[train_indices].mean(axis=0)
            self.input_std = self.inputs[train_indices].std(axis=0) + 1e-8
            print(f"  [归一化] 从当前训练集重新计算 (无外部norm_params)")
        
        unique_combos_val = np.unique(combo_labels[val_indices])
        print(f"  {freq_range}{' (对数)' if log_sample else ''}: "
              f"Train={len(train_indices)}, Val={len(val_indices)} ({len(val_indices)/self.n_total:.1%}), "
              f"覆盖{len(unique_combos_val)}/52工况")
    
    def __len__(self): return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = (self.inputs[i] - self.input_mean) / self.input_std
        return (torch.from_numpy(x).float(),
                torch.tensor(self.types[i], dtype=torch.long),
                torch.tensor(self.modes[i], dtype=torch.long),
                i,
                torch.from_numpy(self.spectra[i]).float())

data_dir = r"F:\lyh\paddlespeech\csvdata333"

# [修复] 尝试从各版本 checkpoint 加载训练时使用的归一化参数
# 所有版本使用相同数据和划分, norm_params 应一致, 任意一个版本的即可
def _try_load_norm_params(ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'input_mean' in ckpt and 'input_std' in ckpt:
            mean = ckpt['input_mean'].numpy() if isinstance(ckpt['input_mean'], torch.Tensor) else ckpt['input_mean']
            std = ckpt['input_std'].numpy() if isinstance(ckpt['input_std'], torch.Tensor) else ckpt['input_std']
            return {'input_mean': mean, 'input_std': std}
    except Exception:
        pass
    return None

_shared_norm_params = None
# 按优先级尝试各版本 checkpoint
for _vname, _run_dir in VERSIONS_TB.items():
    _ckpt_path = os.path.join(runs_base, _run_dir, "models", "best_model.pth")
    _shared_norm_params = _try_load_norm_params(_ckpt_path)
    if _shared_norm_params is not None:
        print(f"[修复] 从 {_vname} checkpoint 加载到训练归一化参数: "
              f"mean={_shared_norm_params['input_mean']}, std={_shared_norm_params['input_std']}")
        break
if _shared_norm_params is None:
    print("[修复] WARNING: 所有checkpoint均无归一化参数(旧版), 将从数据重新计算")

# 构建多个验证集以适配不同模型（传入训练时的归一化参数）
val_ds_20_5000 = UnifiedDataset(data_dir, is_train=False, freq_range='20-5000',
                                 norm_params=_shared_norm_params)
val_ds_60_5000 = UnifiedDataset(data_dir, is_train=False, freq_range='60-5000',
                                 norm_params=_shared_norm_params)
val_ds_log = UnifiedDataset(data_dir, is_train=False, freq_range='20-5000', log_sample=True,
                             norm_params=_shared_norm_params)

# 验证三层数据集是否共享同一套样本
assert np.array_equal(val_ds_20_5000.indices, val_ds_60_5000.indices), "验证集索引不一致!"
assert np.array_equal(val_ds_20_5000.indices, val_ds_log.indices), "验证集索引不一致!"
print(f"  验证集一致性检查: OK (共{len(val_ds_20_5000)}个样本)")

val_loader_20_5000 = DataLoader(val_ds_20_5000, batch_size=8, shuffle=False, pin_memory=True)
val_loader_60_5000 = DataLoader(val_ds_60_5000, batch_size=8, shuffle=False, pin_memory=True)
val_loader_log = DataLoader(val_ds_log, batch_size=8, shuffle=False, pin_memory=True)

# ============================================================
# 模型加载工具
# ============================================================
def load_module(path):
    name = os.path.basename(path).replace('.py','')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# 版本定义
# ============================================================
VERSIONS = OrderedDict([
    ("V10", {
        "net_path": os.path.join(base, "PIMBCN_net0614_v10.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v10_20260614_195124/models/best_model.pth"),
        "freq_bins": 1236, "freq_range": "60-5000",
        "desc": "V10: 60~5000Hz, 截去20-60Hz高误差低频段 (方案A)",
        "val_loader": val_loader_60_5000, "val_ds": val_ds_60_5000,
    }),
    ("V11", {
        "net_path": os.path.join(base, "PIMBCN_net0614_v11.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v11_20260615_141627/models/best_model.pth"),
        "freq_bins": 1246, "freq_range": "20-5000",
        "desc": "V11: 频率加权损失, 低频(20Hz)权重≈2.5×",
        "val_loader": val_loader_20_5000, "val_ds": val_ds_20_5000,
    }),
    ("V12", {
        "net_path": os.path.join(base, "PIMBCN_net0614_v12.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v12_20260615_141720/models/best_model.pth"),
        "freq_bins": 1246, "freq_range": "20-5000",
        "desc": "V12: 低频专项Head, 独立预测20~200Hz(50频点)",
        "val_loader": val_loader_20_5000, "val_ds": val_ds_20_5000,
    }),
    ("V13", {
        "net_path": os.path.join(base, "PIMBCN_net0614_v13.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v13_20260615_141745/models/best_model.pth"),
        "freq_bins": 1246, "freq_range": "20-5000 (对数)",
        "desc": "V13: 对数频率重采样, 低频密度7.5×",
        "val_loader": val_loader_log, "val_ds": val_ds_log,
    }),
    ("V14", {
        "net_path": os.path.join(base, "PIMBCN_net0614_v14.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v14_20260615_141801/models/best_model.pth"),
        "freq_bins": 1236, "freq_range": "60-5000",
        "desc": "V14: 修复V10增强(反射填充)+损失重校准(6:3:1.5)+warmup=20ep",
        "val_loader": val_loader_60_5000, "val_ds": val_ds_60_5000,
    }),
])

# 检查可用性
available = OrderedDict()
for name, cfg in VERSIONS.items():
    if os.path.exists(cfg["ckpt"]) and os.path.exists(cfg["net_path"]):
        available[name] = cfg
        print(f"  {name}: FOUND (freq_bins={cfg['freq_bins']}, {cfg['freq_range']})")
    else:
        print(f"  {name}: MISSING (ckpt={os.path.exists(cfg['ckpt'])}, net={os.path.exists(cfg['net_path'])})")

# ============================================================
# 运行所有模型预测
# ============================================================
print("\n" + "="*60)
print("模型评估")
print("="*60)

results = {}

for name, cfg in available.items():
    print(f"\n{'='*50}\nEvaluating {name}: {cfg['desc']}")
    net_mod = load_module(cfg["net_path"])
    ModelClass = net_mod.PI_MBCN
    
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    
    # 构建模型
    try:
        model = ModelClass(num_modes=4, num_types=13, freq_bins=cfg['freq_bins']).to(device)
    except Exception as e:
        print(f"  Model init error: {e}")
        continue
    
    # 加载权重 (使用 strict=False 兼容不同checkpoint格式)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # 推理
    val_loader = cfg['val_loader']
    val_ds = cfg['val_ds']
    freq = val_ds.freq_axis
    
    preds, targets, idxs = [], [], []
    with torch.no_grad():
        for inputs, types, modes, idx, target in val_loader:
            inputs = inputs.to(device); types = types.to(device); modes = modes.to(device)
            # V12 返回 (spectrum, low_freq_spectrum), 只取主频谱
            output = model(inputs, modes, types)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())
            idxs.extend(idx.numpy())
    
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    idxs = np.array(idxs)
    
    print(f"  Pred shape: {pred.shape}, Target shape: {target.shape}")
    
    # ===================== 计算指标 =====================
    r = {'name': name, 'desc': cfg['desc'], 'params': n_params,
         'freq_bins': cfg['freq_bins'], 'freq_range': cfg['freq_range'],
         'ckpt_epoch': ckpt.get('epoch', '?')}
    
    err = pred - target
    mse = np.mean(err**2)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(mse)
    
    # Cosine similarity
    dot = np.sum(pred * target, axis=1)
    norm_p = np.linalg.norm(pred, axis=1)
    norm_t = np.linalg.norm(target, axis=1)
    cos_sim = np.mean(dot / (norm_p * norm_t + 1e-10))
    
    # R²
    ss_r = np.sum(err**2)
    ss_t = np.sum((target - np.mean(target))**2)
    r2 = 1 - ss_r / (ss_t + 1e-10)
    
    # Pearson
    pears = np.mean([pearsonr(pred[i], target[i])[0] for i in range(len(pred))])
    
    r['val_mse'] = mse; r['val_mae'] = mae; r['val_rmse'] = rmse
    r['val_cos'] = cos_sim; r['val_r2'] = r2; r['val_pearson'] = pears
    
    # Per-sample statistics
    ps_mse = np.mean(err**2, axis=1)
    r['val_med'] = np.median(ps_mse)
    r['val_std'] = np.std(ps_mse)
    r['val_p95'] = np.percentile(ps_mse, 95)
    r['val_max'] = np.max(ps_mse)
    r['val_ps_mse'] = ps_mse
    
    # OASPL (dB) — 从频谱计算总声压级
    def compute_oaspl(spectrum):
        # 安全log-sum-exp
        shifted = spectrum - np.max(spectrum, axis=1, keepdims=True)
        return 10 * np.log10(np.sum(np.power(10., shifted / 10.), axis=1) + 1e-10) + np.max(spectrum, axis=1)
    
    o_pred = compute_oaspl(pred)
    o_true = compute_oaspl(target)
    r['val_oaspl_mae'] = np.mean(np.abs(o_pred - o_true))
    r['val_oaspl_rmse'] = np.sqrt(np.mean((o_pred - o_true)**2))
    
    # Peak frequency & amplitude
    pi_p = np.argmax(pred, axis=1)
    pi_t = np.argmax(target, axis=1)
    r['val_peak_freq_mae'] = np.mean(np.abs(freq[pi_p] - freq[pi_t]))
    r['val_peak_amp_mae'] = np.mean(np.abs(pred[np.arange(len(pred)), pi_p] - target[np.arange(len(target)), pi_t]))
    
    # Per-frequency metrics
    per_f_rmse = np.sqrt(np.mean(err**2, axis=0))
    per_f_mae = np.mean(np.abs(err), axis=0)
    per_f_bias = np.mean(err, axis=0)
    r['freq_rmse'] = per_f_rmse; r['freq_mae'] = per_f_mae; r['freq_bias'] = per_f_bias
    
    # Frequency bands (根据实际频率范围调整)
    if cfg['freq_range'] == '60-5000':
        bands = [(60, 100, "60-100Hz"), (100, 500, "100-500Hz"),
                 (500, 2000, "500-2000Hz"), (2000, 5000, "2000-5000Hz")]
    else:
        bands = [(20, 100, "20-100Hz"), (100, 500, "100-500Hz"),
                 (500, 2000, "500-2000Hz"), (2000, 5000, "2000-5000Hz")]
    
    for lo, hi, label in bands:
        m = (freq >= lo) & (freq <= hi)
        r[f'freq_{label}_rmse'] = np.mean(per_f_rmse[m])
        r[f'freq_{label}_mae'] = np.mean(per_f_mae[m])
    
    # Per-condition analysis
    cm = defaultdict(list)
    for i, vi in enumerate(idxs):
        cm[(val_ds.modes_raw[vi], val_ds.types_raw[vi])].append(ps_mse[i])
    
    best_cond = min(cm.items(), key=lambda x: np.mean(x[1]))
    worst_cond = max(cm.items(), key=lambda x: np.mean(x[1]))
    r['best_condition'] = f"{best_cond[0][0]}+{best_cond[0][1]} (MSE={np.mean(best_cond[1]):.2f})"
    r['worst_condition'] = f"{worst_cond[0][0]}+{worst_cond[0][1]} (MSE={np.mean(worst_cond[1]):.2f})"
    
    # Store for later
    r['val_pred'] = pred; r['val_target'] = target; r['val_idxs'] = idxs
    r['per_condition'] = cm
    
    results[name] = r
    
    print(f"  Val MSE={mse:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  CosSim={cos_sim:.4f}")
    print(f"  R²={r2:.4f}  Pearson={pears:.4f}  Med={r['val_med']:.4f}  P95={r['val_p95']:.4f}  Worst={r['val_max']:.4f}")
    print(f"  OASPL MAE={r['val_oaspl_mae']:.3f} dB  PeakFreq MAE={r['val_peak_freq_mae']:.0f} Hz")
    print(f"  最好工况: {r['best_condition']}")
    print(f"  最差工况: {r['worst_condition']}")

# ============================================================
# 跨版本对比: 将 V13 的预测插值到线性网格
# ============================================================
print("\n" + "="*60)
print("跨版本对比: V13 对数→线性插值")
print("="*60)

if "V13" in results:
    r13 = results["V13"]
    log_freqs = val_ds_log.freq_axis
    linear_freqs = np.linspace(20, 5000, 1246)
    
    # 将V13对数预测插值到线性网格
    v13_pred_linear = np.zeros((len(r13['val_pred']), 1246), dtype=np.float32)
    for i in range(len(r13['val_pred'])):
        v13_pred_linear[i] = np.interp(linear_freqs, log_freqs, r13['val_pred'][i])
    
    # 线性网格上的ground truth
    v13_target_linear = val_ds_20_5000.spectra[val_ds_20_5000.indices]
    
    err_lin = v13_pred_linear - v13_target_linear
    mse_lin = np.mean(err_lin**2)
    mae_lin = np.mean(np.abs(err_lin))
    rmse_lin = np.sqrt(mse_lin)
    
    dot_lin = np.sum(v13_pred_linear * v13_target_linear, axis=1)
    cos_lin = np.mean(dot_lin / (np.linalg.norm(v13_pred_linear, axis=1) * np.linalg.norm(v13_target_linear, axis=1) + 1e-10))
    
    ps_mse_lin = np.mean(err_lin**2, axis=1)
    
    r13['val_mse_linear'] = mse_lin
    r13['val_mae_linear'] = mae_lin
    r13['val_rmse_linear'] = rmse_lin
    r13['val_cos_linear'] = cos_lin
    r13['val_med_linear'] = np.median(ps_mse_lin)
    r13['val_p95_linear'] = np.percentile(ps_mse_lin, 95)
    r13['val_max_linear'] = np.max(ps_mse_lin)
    
    print(f"  V13 对数域 → 线性域: MSE={mse_lin:.4f}, MAE={mae_lin:.4f}, CosSim={cos_lin:.4f}")
    print(f"  (对数域原始: MSE={r13['val_mse']:.4f}, MAE={r13['val_mae']:.4f})")

# ============================================================
# 保存结果
# ============================================================
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(base, f"eval_v10_v14_{timestamp}")
os.makedirs(out_dir, exist_ok=True)
print(f"\n保存结果到: {out_dir}")

# --- Summary Table ---
rows = []
for name, r in results.items():
    row = {
        'Version': name,
        'FreqBins': r['freq_bins'],
        'FreqRange': r['freq_range'],
        'Params': r['params'],
        'CkptEpoch': r['ckpt_epoch'],
        'Val_MSE': f"{r['val_mse']:.4f}",
        'Val_MAE': f"{r['val_mae']:.4f}",
        'Val_RMSE': f"{r['val_rmse']:.4f}",
        'Val_CosSim': f"{r['val_cos']:.4f}",
        'Val_R2': f"{r['val_r2']:.4f}",
        'Val_Pearson': f"{r['val_pearson']:.4f}",
        'Val_Med': f"{r['val_med']:.4f}",
        'Val_P95': f"{r['val_p95']:.4f}",
        'Val_Worst': f"{r['val_max']:.4f}",
        'Val_Std': f"{r['val_std']:.4f}",
        'OASPL_MAE_dB': f"{r['val_oaspl_mae']:.3f}",
        'OASPL_RMSE_dB': f"{r['val_oaspl_rmse']:.3f}",
        'PeakFreq_MAE_Hz': f"{r['val_peak_freq_mae']:.0f}",
        'PeakAmp_MAE_dB': f"{r['val_peak_amp_mae']:.2f}",
        'BestCondition': r['best_condition'],
        'WorstCondition': r['worst_condition'],
        'Description': r['desc'],
    }
    # 添加分频段指标
    for label in ['20-100Hz', '100-500Hz', '500-2000Hz', '2000-5000Hz',
                  '60-100Hz']:
        key = f'freq_{label}_rmse'
        if key in r:
            row[f'Freq_{label}_RMSE'] = f"{r[key]:.4f}"
    
    # V13线性域指标
    if name == 'V13' and 'val_mse_linear' in r:
        row['V13_Linear_MSE'] = f"{r['val_mse_linear']:.4f}"
        row['V13_Linear_MAE'] = f"{r['val_mae_linear']:.4f}"
    
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding='utf-8-sig')
print("  summary.csv saved")

# --- Per-Sample MSE ---
ps_data = {'SampleID': list(range(len(val_ds_20_5000)))}
for name, r in results.items():
    if r['freq_bins'] == 1246 and '对数' not in r['freq_range']:
        ps_data[name] = r['val_ps_mse']
    elif name == 'V13' and 'val_mse_linear' in r:
        # V13: 使用线性插值后的per-sample MSE
        err_lin = r['val_pred']  # 这个需要重新计算
        # 简单处理: 用val_ps_mse近似
        ps_data[name + '(log)'] = r['val_ps_mse']
df_ps = pd.DataFrame(ps_data)
df_ps.to_csv(os.path.join(out_dir, "per_sample_mse.csv"), index=False, encoding='utf-8-sig')
print("  per_sample_mse.csv saved")

# --- Per-Condition Table ---
cond_rows = []
for name, r in results.items():
    for (mode, typ), mses in r['per_condition'].items():
        cond_rows.append({
            'Version': name, 'Mode': mode, 'Type': typ, 'N': len(mses),
            'MSE_Mean': np.mean(mses), 'MSE_Median': np.median(mses),
            'MSE_Worst': np.max(mses), 'MSE_P95': np.percentile(mses, 95)
        })
df_cond = pd.DataFrame(cond_rows)
df_cond.to_csv(os.path.join(out_dir, "per_condition.csv"), index=False, encoding='utf-8-sig')
print("  per_condition.csv saved")

# --- Per-Frequency RMSE ---
freq_data = {'Frequency_Hz': val_ds_20_5000.freq_axis}
for name, r in results.items():
    if r['freq_bins'] == 1246:
        freq_data[f'{name}_RMSE'] = r['freq_rmse']
        freq_data[f'{name}_MAE'] = r['freq_mae']
        freq_data[f'{name}_Bias'] = r['freq_bias']
df_freq = pd.DataFrame(freq_data)
df_freq.to_csv(os.path.join(out_dir, "per_frequency.csv"), index=False, encoding='utf-8-sig')
print("  per_frequency.csv saved")

# 60-5000Hz 模型的频率数据
freq_data_60 = {'Frequency_Hz': val_ds_60_5000.freq_axis}
for name, r in results.items():
    if r['freq_bins'] == 1236:
        freq_data_60[f'{name}_RMSE'] = r['freq_rmse']
        freq_data_60[f'{name}_MAE'] = r['freq_mae']
        freq_data_60[f'{name}_Bias'] = r['freq_bias']
if len(freq_data_60) > 1:
    df_freq_60 = pd.DataFrame(freq_data_60)
    df_freq_60.to_csv(os.path.join(out_dir, "per_frequency_60_5000.csv"), index=False, encoding='utf-8-sig')
    print("  per_frequency_60_5000.csv saved")

# --- 打印最终对比表 ---
print("\n" + "="*80)
print("★ v10-v14 验证集核心指标对比")
print("="*80)
header = f"{'版本':<6} {'freq_bins':>9} {'Val_MSE':>10} {'Val_MAE':>10} {'Val_RMSE':>10} {'CosSim':>8} {'R²':>8} {'Med':>10} {'P95':>10} {'Worst':>10} {'OASPL_MAE':>10}"
print(header)
print("-"*100)
for name, r in results.items():
    print(f"{name:<6} {r['freq_bins']:>9} {r['val_mse']:>10.4f} {r['val_mae']:>10.4f} "
          f"{r['val_rmse']:>10.4f} {r['val_cos']:>8.4f} {r['val_r2']:>8.4f} "
          f"{r['val_med']:>10.4f} {r['val_p95']:>10.4f} {r['val_max']:>10.4f} "
          f"{r['val_oaspl_mae']:>10.3f}")

print("\n★ 分频段 RMSE 对比")
band_header = f"{'版本':<6} {'低频':>10} {'中低频':>10} {'中频':>10} {'高频':>10}"
print(band_header)
print("-"*50)
for name, r in results.items():
    if r['freq_range'] == '60-5000':
        bands_keys = ['60-100Hz', '100-500Hz', '500-2000Hz', '2000-5000Hz']
    else:
        bands_keys = ['20-100Hz', '100-500Hz', '500-2000Hz', '2000-5000Hz']
    vals = [f"{r.get(f'freq_{k}_rmse', 0):.4f}" for k in bands_keys]
    print(f"{name:<6} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

# ============================================================
# 打印 TensorBoard 训练历程摘要
# ============================================================
print("\n" + "="*80)
print("★ TensorBoard 训练历程摘要")
print("="*80)
for vname, info in tb_info.items():
    print(f"\n{vname} ({info['run_dir']}):")
    if 'ckpt_epoch' in info:
        print(f"  Checkpoint epoch: {info['ckpt_epoch']}, best_val_loss: {info['ckpt_best_val_loss']}")
    for tag, m in sorted(info['metrics'].items()):
        print(f"  {tag}: best={m['best']:.4f} @ep{m['best_epoch']}, "
              f"latest={m['latest']:.4f} @ep{m['latest_epoch']} ({m['total']} events)")

print(f"\n{'='*80}")
print(f"评估完成! 结果保存在: {out_dir}")
print(f"{'='*80}")
