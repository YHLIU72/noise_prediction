"""
全版本模型综合评估脚本 (V4, V5, V6, V7, V7-fix, V8)
生成详细指标 + CSV表格 + 对比报告
"""
import os, sys, json, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
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
# 统一数据集（所有版本共用同一个数据划分）
# ============================================================
class UnifiedDataset(Dataset):
    def __init__(self, directory_path, is_train=True, val_split=0.2, seed=42, norm_params=None):
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
        spectra = []
        for val in self.data.iloc[:, 13]:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            arr = np.array(parsed, dtype=np.float32)
            if len(arr) >= 2501: arr = arr[:2501]
            # [修复] 零填充, 与训练代码保持一致
            else: arr = np.array(list(arr) + [0.0] * (2501 - len(arr)), dtype=np.float32)
            spectra.append(arr[5:1251])
        self.spectra = np.array(spectra, dtype=np.float32)
        self.freq_axis = np.linspace(20, 5000, 1246)
        # [fix] 分层抽样: 保证每种 mode×type 组合至少有1个验证样本, 共52种全覆盖
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
        print(f"Stratified split: {len(unique_combos)} conditions, Train={len(train_indices)}, Val={len(val_indices)} ({len(val_indices)/self.n_total:.1%})")
        # [fix] 优先使用外部传入的归一化参数，否则从训练集计算
        if norm_params is not None:
            self.input_mean = norm_params['input_mean']
            self.input_std = norm_params['input_std']
            print(f"  [归一化] 使用外部传入的 norm_params (来自训练checkpoint)")
        else:
            self.input_mean = self.inputs[train_indices].mean(axis=0)
            self.input_std = self.inputs[train_indices].std(axis=0) + 1e-8
            print(f"  [归一化] 从当前训练集重新计算 (无外部norm_params)")
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = (self.inputs[i] - self.input_mean) / self.input_std
        return (torch.from_numpy(x).float(), torch.tensor(self.types[i], dtype=torch.long),
                torch.tensor(self.modes[i], dtype=torch.long), i, torch.from_numpy(self.spectra[i]).float())

data_dir = r"F:\lyh\paddlespeech\csvdata333"

# [修复] 辅助函数：尝试从checkpoint加载归一化参数
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

train_ds = UnifiedDataset(data_dir, is_train=True)
val_ds = UnifiedDataset(data_dir, is_train=False)
freq = val_ds.freq_axis
print(f"Data: {train_ds.n_total} total, Train={len(train_ds)}, Val={len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, pin_memory=True)

# ============================================================
# 版本定义：名称, checkpoint路径, 网络模块, 网络class
# ============================================================
def load_module(path):
    name = os.path.basename(path).replace('.py','')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = r"f:\lyh\paddlespeech\papernoise\physic"
runs_base = os.path.join(base, "runs")

versions = OrderedDict([
    ("V4", {
        "net_path": os.path.join(base, "PIMBCN_net0612_v4.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v4_20260611_215207/models/best_model.pth"),
        "desc": "V4: 1共享Head, Dropout 0.5, 5:2:3, embed=16, ch=64"
    }),
    ("V5", {
        "net_path": os.path.join(base, "PIMBCN_net0612_v5.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v5_20260612_141720/models/best_model.pth"),
        "desc": "V5: 强正则化, DropPath, 1:300:1, wd=5e-3"
    }),
    ("V6", {
        "net_path": os.path.join(base, "PIMBCN_net0612_v6.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v6_20260612_154757/models/best_model.pth"),
        "desc": "V6: 折中正则化, DropPath减半, wd=2e-3"
    }),
    ("V7", {
        "net_path": os.path.join(base, "PIMBCN_net0612_v7.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v7_20260612_185911/models/best_model.pth"),
        "desc": "V7: 回归基线, 5:10:3, wd=1e-3, patience=500截杀"
    }),
    ("V7-fix", {
        "net_path": os.path.join(base, "PIMBCN_net0612_v7.py"),
        "ckpt": os.path.join(runs_base, "pi_mbcn_v7_20260612_185911/models/best_model.pth"),
        "desc": "V7-fix: 续训至ep1513, patience=2000"
    }),
])

# Check which models exist
available = OrderedDict()
for name, cfg in versions.items():
    if os.path.exists(cfg["ckpt"]) and os.path.exists(cfg["net_path"]):
        available[name] = cfg
        print(f"  {name}: FOUND")
    else:
        print(f"  {name}: MISSING")

# ============================================================
# 运行所有模型预测
# ============================================================
results = {}

for name, cfg in available.items():
    print(f"\n{'='*50}\nEvaluating {name}...")
    net_mod = load_module(cfg["net_path"])
    ModelClass = net_mod.PI_MBCN
    
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    
    # Auto-detect model params
    try:
        model = ModelClass(num_modes=4, num_types=13, freq_bins=1246).to(device)
    except:
        try:
            model = ModelClass(num_modes=4, num_types=13, freq_bins=1246, embed_dim=32).to(device)
        except:
            model = ModelClass(num_modes=4, num_types=13, freq_bins=1246, embed_dim=16).to(device)
    
    model.load_state_dict(state, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    
    # Predict
    def predict(loader):
        preds, targets, idxs = [], [], []
        with torch.no_grad():
            for inputs, types, modes, idx, target in loader:
                inputs=inputs.to(device); types=types.to(device); modes=modes.to(device)
                pred = model(inputs, modes, types)
                preds.append(pred.cpu().numpy()); targets.append(target.numpy()); idxs.extend(idx.numpy())
        return np.concatenate(preds), np.concatenate(targets), np.array(idxs)
    
    tp, tt, ti = predict(train_loader)
    vp, vt, vi = predict(val_loader)
    
    # ===================== 计算全部指标 =====================
    r = {'name': name, 'desc': cfg['desc'], 'params': n_params, 'ckpt_epoch': ckpt.get('epoch', '?')}
    
    for split, pred, target, idxs in [("train", tp, tt, ti), ("val", vp, vt, vi)]:
        err = pred - target
        mse = np.mean(err**2); mae = np.mean(np.abs(err)); rmse = np.sqrt(mse)
        dot = np.sum(pred*target, axis=1)
        cos_sim = np.mean(dot/(np.linalg.norm(pred,axis=1)*np.linalg.norm(target,axis=1)+1e-10))
        ss_r = np.sum(err**2); ss_t = np.sum((target-np.mean(target))**2); r2 = 1-ss_r/(ss_t+1e-10)
        pears = np.mean([pearsonr(pred[i],target[i])[0] for i in range(len(pred))])
        
        r[f'{split}_mse'] = mse; r[f'{split}_mae'] = mae; r[f'{split}_rmse'] = rmse
        r[f'{split}_cos'] = cos_sim; r[f'{split}_r2'] = r2; r[f'{split}_pearson'] = pears
        
        # Per-sample
        ps_mse = np.mean(err**2, axis=1)
        r[f'{split}_med'] = np.median(ps_mse); r[f'{split}_std'] = np.std(ps_mse)
        r[f'{split}_p95'] = np.percentile(ps_mse, 95); r[f'{split}_max'] = np.max(ps_mse)
        r[f'{split}_ps_mse'] = ps_mse
        
        # OASPL
        o_pred = 10*np.log10(np.sum(np.power(10.,(pred-np.max(pred,1,keepdims=True))/10.),1)+1e-10)+np.max(pred,1)
        o_true = 10*np.log10(np.sum(np.power(10.,(target-np.max(target,1,keepdims=True))/10.),1)+1e-10)+np.max(target,1)
        r[f'{split}_oaspl_mae'] = np.mean(np.abs(o_pred-o_true))
        r[f'{split}_oaspl_rmse'] = np.sqrt(np.mean((o_pred-o_true)**2))
        
        # Peak
        pi_p = np.argmax(pred,1); pi_t = np.argmax(target,1)
        r[f'{split}_peak_freq_mae'] = np.mean(np.abs(freq[pi_p]-freq[pi_t]))
        r[f'{split}_peak_amp_mae'] = np.mean(np.abs(pred[np.arange(len(pred)),pi_p]-target[np.arange(len(target)),pi_t]))
        
        # Frequency bands (validation only)
        if split == 'val':
            per_f_rmse = np.sqrt(np.mean(err**2, axis=0))
            per_f_mae = np.mean(np.abs(err), axis=0)
            per_f_bias = np.mean(err, axis=0)
            r['freq_rmse'] = per_f_rmse; r['freq_mae'] = per_f_mae; r['freq_bias'] = per_f_bias
            bands = [(20,100,"20-100Hz"),(100,500,"100-500Hz"),(500,2000,"500-2000Hz"),(2000,5000,"2000-5000Hz")]
            for lo,hi,label in bands:
                m = (freq>=lo)&(freq<=hi)
                r[f'freq_{label}_rmse'] = np.mean(per_f_rmse[m])
                r[f'freq_{label}_mae'] = np.mean(per_f_mae[m])
        
        # Store per-sample data for per-condition analysis
        r[f'{split}_pred'] = pred; r[f'{split}_target'] = target; r[f'{split}_idxs'] = idxs
    
    # Generalization
    r['gen_mse_ratio'] = r['val_mse'] / (r['train_mse'] + 1e-10)
    r['gen_mae_ratio'] = r['val_mae'] / (r['train_mae'] + 1e-10)
    
    results[name] = r
    print(f"  Val MSE={r['val_mse']:.4f}  MAE={r['val_mae']:.4f}  Med={r['val_med']:.4f}  Worst={r['val_max']:.4f}  GenRatio={r['gen_mse_ratio']:.2f}x")

# ============================================================
# 保存结果到输出目录
# ============================================================
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(base, f"eval_all_versions_{timestamp}")
os.makedirs(out_dir, exist_ok=True)
print(f"\nSaving results to: {out_dir}")

# --- Summary Table ---
rows = []
for name, r in results.items():
    rows.append({
        'Version': name, 'Params': r['params'], 'CkptEpoch': r['ckpt_epoch'],
        'Train_MSE': f"{r['train_mse']:.4f}", 'Val_MSE': f"{r['val_mse']:.4f}",
        'Val_MAE': f"{r['val_mae']:.4f}", 'Val_RMSE': f"{r['val_rmse']:.4f}",
        'Val_CosSim': f"{r['val_cos']:.4f}", 'Val_R2': f"{r['val_r2']:.4f}",
        'Val_Med': f"{r['val_med']:.4f}", 'Val_P95': f"{r['val_p95']:.4f}",
        'Val_Worst': f"{r['val_max']:.4f}", 'Val_Std': f"{r['val_std']:.4f}",
        'Train_MAE': f"{r['train_mae']:.4f}", 'Gen_MSE_Ratio': f"{r['gen_mse_ratio']:.2f}x",
        'OASPL_MAE': f"{r['val_oaspl_mae']:.3f}", 'OASPL_RMSE': f"{r['val_oaspl_rmse']:.3f}",
        'PeakFreq_MAE': f"{r['val_peak_freq_mae']:.0f}", 'PeakAmp_MAE': f"{r['val_peak_amp_mae']:.2f}",
        'Freq_20-100Hz_RMSE': f"{r['freq_20-100Hz_rmse']:.4f}",
        'Freq_100-500Hz_RMSE': f"{r['freq_100-500Hz_rmse']:.4f}",
        'Freq_500-2000Hz_RMSE': f"{r['freq_500-2000Hz_rmse']:.4f}",
        'Freq_2000-5000Hz_RMSE': f"{r['freq_2000-5000Hz_rmse']:.4f}",
        'Description': r['desc']
    })
df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding='utf-8-sig')
print("  summary.csv saved")

# --- Per-Sample Comparison ---
ps_data = {'SampleID': list(range(len(val_ds)))}
for name, r in results.items():
    ps_data[name] = r['val_ps_mse']
df_ps = pd.DataFrame(ps_data)
df_ps.to_csv(os.path.join(out_dir, "per_sample_mse.csv"), index=False, encoding='utf-8-sig')
print("  per_sample_mse.csv saved")

# --- Per-Condition Table ---
cond_rows = []
for name, r in results.items():
    cm = defaultdict(list)
    for i, vi in enumerate(r['val_idxs']):
        cm[(val_ds.modes_raw[vi], val_ds.types_raw[vi])].append(r['val_ps_mse'][i])
    for (mode, typ), mses in cm.items():
        cond_rows.append({
            'Version': name, 'Mode': mode, 'Type': typ, 'N': len(mses),
            'MSE_Mean': np.mean(mses), 'MSE_Median': np.median(mses),
            'MSE_Worst': np.max(mses), 'MSE_P95': np.percentile(mses, 95)
        })
df_cond = pd.DataFrame(cond_rows)
df_cond.to_csv(os.path.join(out_dir, "per_condition.csv"), index=False, encoding='utf-8-sig')
print("  per_condition.csv saved")

# --- Per-Frequency RMSE ---
freq_data = {'Frequency_Hz': freq}
for name, r in results.items():
    freq_data[f'{name}_RMSE'] = r['freq_rmse']
    freq_data[f'{name}_MAE'] = r['freq_mae']
    freq_data[f'{name}_Bias'] = r['freq_bias']
df_freq = pd.DataFrame(freq_data)
df_freq.to_csv(os.path.join(out_dir, "per_frequency.csv"), index=False, encoding='utf-8-sig')
print("  per_frequency.csv saved")

# --- Charts ---
# 1. Per-sample MSE distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# 1a. Val MSE bar chart
ax = axes[0]
versions_list = list(results.keys())
val_mses = [results[v]['val_mse'] for v in versions_list]
colors = ['#2ecc71' if v == 'V4' else '#3498db' for v in versions_list]
bars = ax.bar(versions_list, val_mses, color=colors, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, val_mses): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.4f}', ha='center', fontsize=8)
ax.set_title('Val MSE (lower=better)'); ax.set_ylabel('MSE (dB^2)')

# 1b. Val MSE boxplot
ax = axes[1]
box_data = [np.clip(results[v]['val_ps_mse'], 0, 8) for v in versions_list]
bp = ax.boxplot(box_data, labels=versions_list, patch_artist=True)
for patch, c in zip(bp['boxes'], colors): patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_title('Per-Sample MSE Distribution'); ax.set_ylabel('MSE (dB^2)')

# 1c. Train vs Val MSE
ax = axes[2]
x = np.arange(len(versions_list)); w = 0.35
ax.bar(x-w/2, [results[v]['train_mse'] for v in versions_list], w, label='Train', color='#e74c3c', alpha=0.7)
ax.bar(x+w/2, [results[v]['val_mse'] for v in versions_list], w, label='Val', color='#3498db', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(versions_list); ax.legend(); ax.set_title('Train vs Val MSE')

# 1d. Generalization ratio
ax = axes[3]
ratios = [results[v]['gen_mse_ratio'] for v in versions_list]
ax.bar(versions_list, ratios, color=['#2ecc71' if r < 1.1 else '#f39c12' if r < 1.5 else '#e74c3c' for r in ratios])
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal 1.0')
ax.axhline(y=1.2, color='orange', linestyle='--', alpha=0.5, label='Good <1.2')
ax.set_title('Generalization Ratio (Val/Train MSE)'); ax.legend(fontsize=7)

# 1e. Per-frequency RMSE
ax = axes[4]
for name, r in results.items():
    ax.semilogy(freq, r['freq_rmse'], label=name, linewidth=0.8, alpha=0.8)
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('RMSE (dB)'); ax.set_title('Per-Frequency RMSE')
ax.legend(fontsize=7, ncol=2); ax.set_xlim(20, 5000)

# 1f. OASPL error
ax = axes[5]
oaspl_mae = [results[v]['val_oaspl_mae'] for v in versions_list]
ax.bar(versions_list, oaspl_mae, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('OASPL MAE (dB)'); ax.set_ylabel('MAE (dB)')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "comparison_charts.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  comparison_charts.png saved")

# 2. Per-condition heatmap
fig, axes = plt.subplots(1, len(versions_list), figsize=(4*len(versions_list), 12))
if len(versions_list) == 1: axes = [axes]
for ax, name in zip(axes, versions_list):
    r = results[name]
    cm = defaultdict(lambda: defaultdict(list))
    for i, vi in enumerate(r['val_idxs']):
        cm[val_ds.modes_raw[vi]][val_ds.types_raw[vi]].append(r['val_ps_mse'][i])
    modes_list = sorted(cm.keys())
    types_list = sorted(set(val_ds.types_raw))
    heatmap = np.full((len(modes_list), len(types_list)), np.nan)
    for mi, mode in enumerate(modes_list):
        for ti, typ in enumerate(types_list):
            if typ in cm[mode]: heatmap[mi, ti] = np.mean(cm[mode][typ])
    # [fix] Use gray for missing conditions instead of white
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='#cccccc')
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(types_list))); ax.set_xticklabels(types_list, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(modes_list))); ax.set_yticklabels(modes_list, fontsize=7)
    ax.set_title(f'{name} MSE by Condition'); plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "condition_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  condition_heatmap.png saved")

# ============================================================
# 最终报告
# ============================================================
print(f"\n{'='*80}")
print(f"  FINAL SUMMARY: 全版本综合评估")
print(f"{'='*80}")
print(f"\n{'Version':<10s} {'ValMSE':>8s} {'ValMAE':>8s} {'Median':>8s} {'Worst':>8s} {'P95':>8s} {'GenRatio':>8s} {'OASPL':>8s} {'PeakFreq':>8s} {'20-100Hz':>10s} {'100-500Hz':>10s} {'500-2kHz':>10s} {'2-5kHz':>10s}")
print("-"*120)
for name in versions_list:
    r = results[name]
    print(f"{name:<10s} {r['val_mse']:8.4f} {r['val_mae']:8.4f} {r['val_med']:8.4f} {r['val_max']:8.4f} {r['val_p95']:8.4f} {r['gen_mse_ratio']:7.2f}x {r['val_oaspl_mae']:7.3f} {r['val_peak_freq_mae']:7.0f} {r['freq_20-100Hz_rmse']:10.4f} {r['freq_100-500Hz_rmse']:10.4f} {r['freq_500-2000Hz_rmse']:10.4f} {r['freq_2000-5000Hz_rmse']:10.4f}")

# Find best in each category
best_mse = min(versions_list, key=lambda v: results[v]['val_mse'])
best_med = min(versions_list, key=lambda v: results[v]['val_med'])
best_worst = min(versions_list, key=lambda v: results[v]['val_max'])
best_gen = min(versions_list, key=lambda v: results[v]['gen_mse_ratio'])
print(f"\n  Best Val MSE:  {best_mse} ({results[best_mse]['val_mse']:.4f})")
print(f"  Best Median:   {best_med} ({results[best_med]['val_med']:.4f})")
print(f"  Best Worst:    {best_worst} ({results[best_worst]['val_max']:.4f})")
print(f"  Best Gen Ratio:{best_gen} ({results[best_gen]['gen_mse_ratio']:.2f}x)")

# Save report
with open(os.path.join(out_dir, "report.txt"), 'w', encoding='utf-8') as f:
    f.write(f"全版本综合评估报告\n{'='*80}\n\n")
    f.write(df_summary.to_string(index=False))
    f.write(f"\n\nBest in category:\n")
    f.write(f"  Val MSE: {best_mse}\n  Median: {best_med}\n  Worst-case: {best_worst}\n  Generalization: {best_gen}\n")

print(f"\nAll results saved to: {out_dir}")
print("Done!")
