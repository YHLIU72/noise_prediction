"""
V13 (对数频率重采样) 模型预测效果图 + 架构图
研究生毕业论文风格 · 中文学术排版
批量保存至 plot_thesis/ 文件夹
"""
import os, sys, warnings, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import torch, numpy as np, pandas as pd, ast, importlib.util
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, gaussian_kde
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# ============ 中文学术风格全局设置 ============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'legend.fontsize': 8,
    'legend.framealpha': 0.8,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = r"f:\lyh\paddlespeech\papernoise\physic"
data_dir = r"F:\lyh\paddlespeech\csvdata333"
out_dir = os.path.join(base, "plot_thesis")
os.makedirs(out_dir, exist_ok=True)

# ============ 数据加载 ============
class EvalDataset(Dataset):
    def __init__(self, directory_path, is_train=True, val_split=0.2, seed=42, start_idx=5, end_idx=1251):
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
        spectra_raw = []
        for val in self.data.iloc[:, 13]:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            arr = np.array(parsed, dtype=np.float32)
            if len(arr) >= 2501: arr = arr[:2501]
            else: arr = np.pad(arr, (0, 2501-len(arr)), 'edge')
            spectra_raw.append(arr)
        spectra_raw = np.array(spectra_raw, dtype=np.float32)
        self.spectra_all = spectra_raw[:, start_idx:end_idx]
        self.freq_bins = end_idx - start_idx
        np.random.seed(seed)
        combo_labels = self.modes * 1000 + self.types
        unique_combos = np.unique(combo_labels)
        combo_indices = [np.where(combo_labels == c)[0] for c in unique_combos]
        target_val = int(self.n_total * val_split)
        val_counts = [max(1, min(int(np.round(len(c) * val_split)), len(c)//2)) for c in combo_indices]
        total_assigned = sum(val_counts)
        if total_assigned > target_val:
            ratio = target_val / total_assigned
            val_counts = [max(1, int(np.round(c * ratio))) for c in val_counts]
        train_idx_list, val_idx_list = [], []
        for idx_c, n_val in zip(combo_indices, val_counts):
            if len(idx_c) == 1: train_idx_list.extend(idx_c)
            else: val_idx_list.extend(idx_c[:n_val]); train_idx_list.extend(idx_c[n_val:])
        val_indices = np.array(val_idx_list, dtype=np.int64)
        train_indices = np.array(train_idx_list, dtype=np.int64)
        self.indices = train_indices if is_train else val_indices
        self.input_mean = self.inputs[train_indices].mean(axis=0)
        self.input_std = self.inputs[train_indices].std(axis=0) + 1e-8
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = (self.inputs[i] - self.input_mean) / self.input_std
        return (torch.from_numpy(x).float(), torch.tensor(self.types[i], dtype=torch.long),
                torch.tensor(self.modes[i], dtype=torch.long), i,
                torch.from_numpy(self.spectra_all[i]).float())

def load_module(path):
    name = os.path.basename(path).replace('.py','')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load data and model
print("Loading data and V13 model...")
ds = EvalDataset(data_dir, is_train=False, start_idx=5, end_idx=1251)
loader = DataLoader(ds, batch_size=8, shuffle=False, pin_memory=True)

# Log frequency interpolation
linear_freqs = np.linspace(20, 5000, 1246)
log_freqs = np.logspace(np.log10(20), np.log10(5000), 1246)

# Load V13 model
net_mod = load_module(os.path.join(base, "PIMBCN_net0614_v13.py"))
ModelClass = net_mod.PI_MBCN
ckpt = torch.load(os.path.join(base, "runs", "pi_mbcn_v13_20260615_141745/models/best_model.pth"), map_location=device)
state = ckpt.get('model_state_dict', ckpt)
model = ModelClass(num_modes=4, num_types=13, freq_bins=1246).to(device)
model.load_state_dict(state, strict=False)
model.eval()

# Inference
preds_log, targets_linear, idxs = [], [], []
with torch.no_grad():
    for inputs, types, modes, idx, target in loader:
        inputs, types, modes = inputs.to(device), types.to(device), modes.to(device)
        pred = model(inputs, modes, types)
        preds_log.append(pred.cpu().numpy())
        targets_linear.append(target.numpy())
        idxs.extend(idx.numpy())

pred_log_all = np.concatenate(preds_log)
target_all = np.concatenate(targets_linear)
idxs = np.array(idxs)

# Interpolate predictions to linear grid
pred_linear = np.zeros_like(pred_log_all)
for i in range(len(pred_log_all)):
    pred_linear[i] = np.interp(linear_freqs, log_freqs, pred_log_all[i])

print(f"Predictions: {pred_linear.shape}, Targets: {target_all.shape}")
print(f"Val samples: {len(idxs)}")

# Compute per-sample metrics
err = pred_linear - target_all
ps_mse = np.mean(err**2, axis=1)

def compute_oaspl(s):
    shifted = s - np.max(s, axis=1, keepdims=True)
    return 10 * np.log10(np.sum(np.power(10., shifted / 10.), axis=1) + 1e-10) + np.max(s, axis=1)

o_pred = compute_oaspl(pred_linear)
o_true = compute_oaspl(target_all)

# ================================================================
# 图1: 最佳/中等/最差 三个典型样本预测对比
# ================================================================
print("Plotting Fig1: Best/Median/Worst samples...")
sorted_idx = np.argsort(ps_mse)
best_idx = sorted_idx[0]
med_idx = sorted_idx[len(sorted_idx)//2]
worst_idx = sorted_idx[-1]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
titles = ['(a) 最优预测样本', '(b) 中等预测样本', '(c) 最差预测样本']
sample_idxs = [best_idx, med_idx, worst_idx]
colors_sample = ['#2E7D32', '#1565C0', '#C62828']

for ax, s_idx, title, color in zip(axes, sample_idxs, titles, colors_sample):
    vi = idxs[s_idx]
    mode_name = ds.modes_raw[vi]
    type_name = ds.types_raw[vi]
    mse_val = ps_mse[s_idx]
    
    ax.semilogx(linear_freqs, target_all[s_idx], color='#333333', linewidth=1.8, 
                label='真实值 (Ground Truth)', alpha=0.85)
    ax.semilogx(linear_freqs, pred_linear[s_idx], color=color, linewidth=1.8, 
                label='预测值 (Prediction)', alpha=0.85)
    
    ax.fill_between(linear_freqs, target_all[s_idx], pred_linear[s_idx], 
                     alpha=0.15, color=color)
    
    ax.set_xlabel('频率 / Hz', fontsize=9)
    ax.set_ylabel('声压级 / dB', fontsize=9)
    ax.set_title(f'{title}\n{type_name} ({mode_name})  MSE={mse_val:.2f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(20, 5000)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig01_best_median_worst_prediction.png'), dpi=300)
plt.close()
print("  -> fig01 saved")

# ================================================================
# 图2: 多工况随机样本预测网格 (3×3)
# ================================================================
print("Plotting Fig2: Multi-condition grid...")
np.random.seed(42)
rand_idx = np.random.choice(len(idxs), 9, replace=False)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.flatten()

for i, s_idx in enumerate(rand_idx):
    ax = axes[i]
    vi = idxs[s_idx]
    mode_name = ds.modes_raw[vi][:8] if len(ds.modes_raw[vi]) > 8 else ds.modes_raw[vi]
    type_name = ds.types_raw[vi][:8] if len(ds.types_raw[vi]) > 8 else ds.types_raw[vi]
    
    ax.semilogx(linear_freqs, target_all[s_idx], color='#333333', linewidth=1.3, label='真实值')
    ax.semilogx(linear_freqs, pred_linear[s_idx], color='#1565C0', linewidth=1.3, label='预测值', alpha=0.8)
    ax.fill_between(linear_freqs, target_all[s_idx], pred_linear[s_idx], alpha=0.08, color='#1565C0')
    
    ax.set_title(f'{type_name} | {mode_name}\nMSE={ps_mse[s_idx]:.2f}', fontsize=8)
    ax.set_xlim(20, 5000)
    ax.tick_params(labelsize=6)
    if i >= 6: ax.set_xlabel('频率 / Hz', fontsize=7)
    if i % 3 == 0: ax.set_ylabel('SPL / dB', fontsize=7)

# Single legend
lines1, _ = axes[0].get_legend_handles_labels()
fig.legend(lines1, ['真实值 (Ground Truth)', '预测值 (Prediction)'], 
           loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('V13 模型多工况预测效果对比（随机抽取9个验证集样本）', fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig02_multi_condition_grid.png'), dpi=300)
plt.close()
print("  -> fig02 saved")

# ================================================================
# 图3: 逐频率误差分析 (RMSE + Bias + 频段标注)
# ================================================================
print("Plotting Fig3: Per-frequency error analysis...")
per_f_rmse = np.sqrt(np.mean(err**2, axis=0))
per_f_mae = np.mean(np.abs(err), axis=0)
per_f_bias = np.mean(err, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# RMSE + MAE
ax = axes[0]
ax.semilogx(linear_freqs, per_f_rmse, color='#C62828', linewidth=1.2, label='RMSE')
ax.semilogx(linear_freqs, per_f_mae, color='#1565C0', linewidth=1.2, label='MAE', alpha=0.7)
ax.set_ylabel('误差 / dB', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.set_title('(a) 逐频率验证集误差分布', fontsize=11, fontweight='bold')

# Add band annotations
band_colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#F3E5F5']
band_labels = ['低频段', '中低频段', '中频段', '高频段']
bands = [(20, 100), (100, 500), (500, 2000), (2000, 5000)]
for (lo, hi), color, label in zip(bands, band_colors, band_labels):
    ax.axvspan(lo, hi, alpha=0.2, color=color, zorder=0)
    mid = np.sqrt(lo * hi)
    ax.text(mid, ax.get_ylim()[1]*0.95, label, ha='center', va='top', fontsize=7, 
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

# Bias
ax2 = axes[1]
ax2.semilogx(linear_freqs, per_f_bias, color='#E65100', linewidth=1.2)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax2.fill_between(linear_freqs, 0, per_f_bias, 
                  where=(per_f_bias > 0), alpha=0.2, color='#C62828', label='正偏差 (高估)')
ax2.fill_between(linear_freqs, 0, per_f_bias, 
                  where=(per_f_bias < 0), alpha=0.2, color='#1565C0', label='负偏差 (低估)')
ax2.set_xlabel('频率 / Hz', fontsize=10)
ax2.set_ylabel('偏差 / dB', fontsize=10)
ax2.set_title('(b) 逐频率预测偏差 (Bias)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)

for ax in axes:
    ax.set_xlim(20, 5000)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig03_per_frequency_error.png'), dpi=300)
plt.close()
print("  -> fig03 saved")

# ================================================================
# 图4: OASPL 散点图 + 误差分布
# ================================================================
print("Plotting Fig4: OASPL scatter + error distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter
ax = axes[0]
oaspl_min = min(o_true.min(), o_pred.min()) - 2
oaspl_max = max(o_true.max(), o_pred.max()) + 2
ax.scatter(o_true, o_pred, c=ps_mse, cmap='RdYlBu_r', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
ax.plot([oaspl_min, oaspl_max], [oaspl_min, oaspl_max], 'k--', linewidth=1, label='y=x')
ax.set_xlabel('真实 OASPL / dB', fontsize=10)
ax.set_ylabel('预测 OASPL / dB', fontsize=10)
ax.set_title('(a) OASPL 预测散点图', fontsize=11, fontweight='bold')
cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
cbar.set_label('频谱 MSE', fontsize=8)
ax.legend(fontsize=8)
ax.set_xlim(oaspl_min, oaspl_max); ax.set_ylim(oaspl_min, oaspl_max)
ax.set_aspect('equal')

# Annotate stats
oaspl_err = o_pred - o_true
mae_o = np.mean(np.abs(oaspl_err))
rmse_o = np.sqrt(np.mean(oaspl_err**2))
ax.text(0.03, 0.97, f'MAE = {mae_o:.2f} dB\nRMSE = {rmse_o:.2f} dB\n$R^2$ = {np.corrcoef(o_true, o_pred)[0,1]**2:.4f}',
        transform=ax.transAxes, va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Error histogram
ax2 = axes[1]
ax2.hist(oaspl_err, bins=30, color='#1565C0', alpha=0.7, edgecolor='white', density=True)
# KDE
kde = gaussian_kde(oaspl_err)
x_kde = np.linspace(oaspl_err.min(), oaspl_err.max(), 200)
ax2.plot(x_kde, kde(x_kde), 'r-', linewidth=1.5, label='核密度估计')
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
ax2.axvline(x=np.mean(oaspl_err), color='#C62828', linestyle='-', linewidth=1, 
            label=f'均值 = {np.mean(oaspl_err):.3f} dB')
ax2.set_xlabel('OASPL 预测误差 / dB', fontsize=10)
ax2.set_ylabel('概率密度', fontsize=10)
ax2.set_title('(b) OASPL 误差分布', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig04_oaspl_scatter_hist.png'), dpi=300)
plt.close()
print("  -> fig04 saved")

# ================================================================
# 图5: Per-Condition MSE 热力图 (Mode × Type)
# ================================================================
print("Plotting Fig5: Per-condition MSE heatmap...")
cm = defaultdict(list)
for i, vi in enumerate(idxs):
    cm[(ds.modes_raw[vi], ds.types_raw[vi])].append(ps_mse[i])

all_modes = sorted(set(ds.modes_raw))
all_types = sorted(set(ds.types_raw))
heatmap_data = np.full((len(all_modes), len(all_types)), np.nan)
for mi, mode in enumerate(all_modes):
    for ti, typ in enumerate(all_types):
        key = (mode, typ)
        if key in cm:
            heatmap_data[mi, ti] = np.mean(cm[key])

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', vmin=np.nanmin(heatmap_data), vmax=np.nanpercentile(heatmap_data[~np.isnan(heatmap_data)], 95))

ax.set_xticks(range(len(all_types)))
ax.set_yticks(range(len(all_modes)))
ax.set_xticklabels(all_types, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(all_modes, fontsize=8)
ax.set_xlabel('车辆类型 (Type)', fontsize=10)
ax.set_ylabel('工况模式 (Mode)', fontsize=10)
ax.set_title('V13 模型各工况验证集 MSE 热力图', fontsize=12, fontweight='bold')

# Annotate
for mi in range(len(all_modes)):
    for ti in range(len(all_types)):
        val = heatmap_data[mi, ti]
        if not np.isnan(val):
            color = 'white' if val > np.nanmean(heatmap_data) else 'black'
            ax.text(ti, mi, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('MSE / dB²', fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig05_per_condition_heatmap.png'), dpi=300)
plt.close()
print("  -> fig05 saved")

# ================================================================
# 图6: 误差分布直方图 (全体验证集)
# ================================================================
print("Plotting Fig6: Error distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Per-sample MSE histogram
ax = axes[0]
ax.hist(ps_mse, bins=35, color='#1565C0', alpha=0.7, edgecolor='white')
ax.axvline(x=np.median(ps_mse), color='#C62828', linestyle='-', linewidth=1.5, label=f'中位数 = {np.median(ps_mse):.2f}')
ax.axvline(x=np.mean(ps_mse), color='#E65100', linestyle='--', linewidth=1.5, label=f'均值 = {np.mean(ps_mse):.2f}')
ax.axvline(x=np.percentile(ps_mse, 95), color='#7B1FA2', linestyle=':', linewidth=1.5, label=f'P95 = {np.percentile(ps_mse, 95):.2f}')
ax.set_xlabel('Per-Sample MSE / dB²', fontsize=10)
ax.set_ylabel('样本数量', fontsize=10)
ax.set_title('(a) 验证集 MSE 分布', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)

# QQ-plot-like: sorted per-sample MSE
ax2 = axes[1]
sorted_mse = np.sort(ps_mse)
n = len(sorted_mse)
cum_frac = (np.arange(1, n+1)) / n
ax2.plot(cum_frac, sorted_mse, 'b-', linewidth=1.5)
ax2.axhline(y=np.median(ps_mse), color='#C62828', linestyle='--', linewidth=1, alpha=0.6)
ax2.fill_between([0.5, 1.0], 0, sorted_mse.max(), alpha=0.1, color='#FF9800')
ax2.text(0.75, sorted_mse.max()*0.9, '后50%样本', ha='center', fontsize=8, color='#E65100')
ax2.fill_between([0.9, 1.0], 0, sorted_mse.max(), alpha=0.15, color='#F44336')
ax2.text(0.95, sorted_mse.max()*0.8, 'P10', ha='center', fontsize=8, color='#C62828')
ax2.set_xlabel('累积样本比例', fontsize=10)
ax2.set_ylabel('Per-Sample MSE / dB²', fontsize=10)
ax2.set_title('(b) MSE 累积分布', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig06_error_distribution.png'), dpi=300)
plt.close()
print("  -> fig06 saved")

# ================================================================
# 图7: 综合仪表盘 (4合1摘要)
# ================================================================
print("Plotting Fig7: Summary dashboard...")
fig = plt.figure(figsize=(14, 10))

# (a) Overall metrics bar chart
ax1 = fig.add_subplot(2, 2, 1)
metrics_names = ['MSE', 'MAE', 'RMSE', '中位数\nMSE', 'P95\nMSE', '最差\nMSE']
metrics_vals = [np.mean(ps_mse), np.mean(np.abs(err)), np.sqrt(np.mean(err**2)),
                np.median(ps_mse), np.percentile(ps_mse, 95), np.max(ps_mse)]
bar_colors = ['#1565C0', '#1976D2', '#1E88E5', '#43A047', '#FB8C00', '#E53935']
bars = ax1.bar(range(len(metrics_names)), metrics_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(metrics_names)))
ax1.set_xticklabels(metrics_names, fontsize=8)
ax1.set_ylabel('dB² / dB', fontsize=9)
ax1.set_title('(a) 核心指标概览', fontsize=11, fontweight='bold')
for bar, val in zip(bars, metrics_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, f'{val:.2f}',
             ha='center', va='bottom', fontsize=8)

# (b) Band RMSE
ax2 = fig.add_subplot(2, 2, 2)
band_names = ['20-100Hz\n(低频)', '100-500Hz\n(中低频)', '500-2000Hz\n(中频)', '2000-5000Hz\n(高频)']
per_f_rmse_arr = np.sqrt(np.mean(err**2, axis=0))
band_vals = []
for lo, hi in [(20,100), (100,500), (500,2000), (2000,5000)]:
    m = (linear_freqs >= lo) & (linear_freqs <= hi)
    band_vals.append(np.mean(per_f_rmse_arr[m]))
band_colors_4 = ['#1B5E20', '#4CAF50', '#8BC34A', '#CDDC39']
bars2 = ax2.bar(range(4), band_vals, color=band_colors_4, edgecolor='white')
ax2.set_xticks(range(4))
ax2.set_xticklabels(band_names, fontsize=8)
ax2.set_ylabel('RMSE / dB', fontsize=9)
ax2.set_title('(b) 分频段 RMSE', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, band_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# (c) OASPL error by condition count
ax3 = fig.add_subplot(2, 2, 3)
cond_oaspl = defaultdict(list)
for i, vi in enumerate(idxs):
    key = f"{ds.modes_raw[vi][:4]}+{ds.types_raw[vi][:8]}"
    cond_oaspl[key].append(abs(oaspl_err[i]))
cond_names_sorted = sorted(cond_oaspl.keys(), key=lambda x: np.mean(cond_oaspl[x]))[:12]
cond_means = [np.mean(cond_oaspl[c]) for c in cond_names_sorted]
ax3.barh(range(len(cond_names_sorted)), cond_means, color='#FF9800', edgecolor='white', alpha=0.8)
ax3.set_yticks(range(len(cond_names_sorted)))
ax3.set_yticklabels(cond_names_sorted, fontsize=7)
ax3.set_xlabel('OASPL MAE / dB', fontsize=9)
ax3.set_title('(c) 各工况 OASPL MAE (最优12个)', fontsize=11, fontweight='bold')
ax3.invert_yaxis()

# (d) Sample quality pie
ax4 = fig.add_subplot(2, 2, 4)
labels_pie = ['优秀 (MSE<0.5)', '良好 (0.5-1.0)', '一般 (1.0-2.0)', '较差 (2.0-5.0)', '差 (MSE>5.0)']
counts = [
    np.sum(ps_mse < 0.5), np.sum((ps_mse >= 0.5) & (ps_mse < 1.0)),
    np.sum((ps_mse >= 1.0) & (ps_mse < 2.0)), np.sum((ps_mse >= 2.0) & (ps_mse < 5.0)),
    np.sum(ps_mse >= 5.0)
]
pie_colors = ['#1B5E20', '#4CAF50', '#FFC107', '#FF9800', '#F44336']
explode = (0.02, 0.02, 0.02, 0.02, 0.05)
wedges, texts, autotexts = ax4.pie(counts, explode=explode, labels=labels_pie, colors=pie_colors,
                                     autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
for at in autotexts: at.set_fontsize(8)
ax4.set_title('(d) 样本预测质量分布', fontsize=11, fontweight='bold')

fig.suptitle('V13 模型验证集评估总览（对数频率重采样 · 1共享Head）', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig07_summary_dashboard.png'), dpi=300)
plt.close()
print("  -> fig07 saved")

print(f"\n所有预测效果图已保存至: {out_dir}")
print("预测效果图绘制完成!")
