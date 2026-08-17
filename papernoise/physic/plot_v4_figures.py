"""
V4 模型预测效果图 — 研究生毕业论文风格
批量保存至 plot_thesis/ 文件夹 (V4_ 前缀)
"""
import os, sys, warnings, io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'; os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch, numpy as np, importlib.util
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 10, 'axes.unicode_minus': False,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1, 'axes.grid': True, 'grid.alpha': 0.3,
    'grid.linestyle': '--', 'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
    'legend.fontsize': 8, 'legend.framealpha': 0.8,
})

device = torch.device("cuda")
base = r"f:\lyh\paddlespeech\papernoise\physic"
data_dir = r"F:\lyh\paddlespeech\csvdata333"
out_dir = os.path.join(base, "plot_thesis")

def load_module(path):
    name = os.path.basename(path).replace('.py','')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ===== 加载数据 & 模型 =====
print("Loading V4 model...")
from PIMBCN_data_0612_v4 import PIMBCNDataset
train_ds = PIMBCNDataset(directory_path=data_dir, is_validation=False, augment=False)
val_ds = train_ds.get_validation_dataset()
val_loader = DataLoader(val_ds, batch_size=214, shuffle=False)

# V4数据类的原始字符串需要通过data DataFrame获取
types_raw_v4 = val_ds.data.iloc[:, 3].values.astype(str)
modes_raw_v4 = val_ds.data.iloc[:, 2].values.astype(str)

net_v4 = load_module(os.path.join(base, "PIMBCN_net0612_v4.py"))
ckpt = torch.load(os.path.join(base, "runs", "pi_mbcn_v4_20260611_215207/models/best_model.pth"), map_location=device)
model = net_v4.PI_MBCN(num_modes=4, num_types=13, freq_bins=1246).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
n_params = sum(p.numel() for p in model.parameters())

for batch in val_loader:
    inputs, type_idx, mode_idx, oaspl_v, octave_v, spectrum = batch
    break
inputs = inputs.to(device); type_idx = type_idx.to(device); mode_idx = mode_idx.to(device)
target_all = spectrum.numpy()
val_indices = val_ds.indices
freq = val_ds.freq_axis

with torch.no_grad():
    pred_all = model(inputs, mode_idx, type_idx).cpu().numpy()

print(f"Pred: {pred_all.shape}, Target: {target_all.shape}")

# ===== 指标计算 =====
err = pred_all - target_all
ps_mse = np.mean(err**2, axis=1)

def oaspl(s):
    shifted = s - np.max(s, axis=1, keepdims=True)
    return 10*np.log10(np.sum(np.power(10., shifted/10.), axis=1)+1e-10) + np.max(s, axis=1)
o_pred = oaspl(pred_all); o_true = oaspl(target_all)

# Peak bias
peak_biases = np.array([pred_all[i, np.argmax(target_all[i])] - target_all[i, np.argmax(target_all[i])] for i in range(len(pred_all))])

print(f"MSE={np.mean(err**2):.4f} MAE={np.mean(np.abs(err)):.4f} OASPL_MAE={np.mean(np.abs(o_pred-o_true)):.3f}dB PeakBias={np.mean(peak_biases):+.2f}dB")

# ================================================================
# 图1: 最佳/中等/最差 三个典型样本预测对比
# ================================================================
print("Fig1: Best/Median/Worst...")
sorted_idx = np.argsort(ps_mse)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, s_idx, title, color in zip(axes, 
    [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]],
    ['(a) 最优预测样本', '(b) 中等预测样本', '(c) 最差预测样本'],
    ['#2E7D32', '#1565C0', '#C62828']):
    vi = val_indices[s_idx]
    ax.semilogx(freq, target_all[s_idx], color='#333333', linewidth=1.8, label='Ground Truth')
    ax.semilogx(freq, pred_all[s_idx], color=color, linewidth=1.8, label='Prediction', alpha=0.85)
    ax.fill_between(freq, target_all[s_idx], pred_all[s_idx], alpha=0.15, color=color)
    ax.set_xlabel('Frequency / Hz', fontsize=9)
    ax.set_ylabel('SPL / dB', fontsize=9)
    ax.set_title(f'{title}\n{types_raw_v4[vi]} ({modes_raw_v4[vi][:8]})  MSE={ps_mse[s_idx]:.2f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.set_xlim(20, 5000); ax.tick_params(labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig01_best_median_worst.png'), dpi=300)
plt.close()

# ================================================================
# 图2: 3x3 随机样本网格
# ================================================================
print("Fig2: Multi-condition grid...")
np.random.seed(42)
rand_idx = np.random.choice(len(pred_all), 9, replace=False)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, s_idx in enumerate(rand_idx):
    ax = axes.flat[i]; vi = val_indices[s_idx]
    ax.semilogx(freq, target_all[s_idx], color='#333333', linewidth=1.3, label='Ground Truth')
    ax.semilogx(freq, pred_all[s_idx], color='#1565C0', linewidth=1.3, label='Prediction', alpha=0.8)
    ax.fill_between(freq, target_all[s_idx], pred_all[s_idx], alpha=0.08, color='#1565C0')
    ax.set_title(f'{types_raw_v4[vi][:8]} | {modes_raw_v4[vi][:8]}\nMSE={ps_mse[s_idx]:.2f}', fontsize=8)
    ax.set_xlim(20, 5000); ax.tick_params(labelsize=6)
    if i >= 6: ax.set_xlabel('Frequency / Hz', fontsize=7)
    if i % 3 == 0: ax.set_ylabel('SPL / dB', fontsize=7)
lines1, _ = axes.flat[0].get_legend_handles_labels()
fig.legend(lines1, ['Ground Truth', 'Prediction'], loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('V4 Model: Multi-Condition Prediction Grid (9 Random Validation Samples)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig02_multi_condition_grid.png'), dpi=300)
plt.close()

# ================================================================
# 图3: 逐频率误差
# ================================================================
print("Fig3: Per-frequency error...")
per_f_rmse = np.sqrt(np.mean(err**2, axis=0))
per_f_bias = np.mean(err, axis=0)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax = axes[0]
ax.semilogx(freq, per_f_rmse, color='#C62828', linewidth=1.2, label='RMSE')
ax.set_ylabel('Error / dB', fontsize=10); ax.legend(fontsize=9)
ax.set_title('(a) Per-Frequency Validation RMSE', fontsize=11, fontweight='bold')
bands = [(20,100),(100,500),(500,2000),(2000,5000)]
colors_b = ['#E8F5E9','#FFF3E0','#E3F2FD','#F3E5F5']
labels_b = ['Low','Mid-Low','Mid','High']
for (lo,hi),c,l in zip(bands,colors_b,labels_b):
    ax.axvspan(lo,hi,alpha=0.2,color=c,zorder=0)
    ax.text(np.sqrt(lo*hi), ax.get_ylim()[1]*0.95, l, ha='center',va='top',fontsize=7,
            bbox=dict(boxstyle='round',facecolor=c,alpha=0.7))
ax2 = axes[1]
ax2.semilogx(freq, per_f_bias, color='#E65100', linewidth=1.2)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax2.fill_between(freq, 0, per_f_bias, where=(per_f_bias>0), alpha=0.2, color='#C62828', label='Over-predict')
ax2.fill_between(freq, 0, per_f_bias, where=(per_f_bias<0), alpha=0.2, color='#1565C0', label='Under-predict')
ax2.set_xlabel('Frequency / Hz', fontsize=10); ax2.set_ylabel('Bias / dB', fontsize=10)
ax2.set_title('(b) Per-Frequency Prediction Bias', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
for ax in axes: ax.set_xlim(20,5000); ax.tick_params(labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig03_per_frequency_error.png'), dpi=300)
plt.close()

# ================================================================
# 图4: OASPL 散点图 + 误差分布
# ================================================================
print("Fig4: OASPL scatter...")
oaspl_err = o_pred - o_true
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
sc = ax.scatter(o_true, o_pred, c=ps_mse, cmap='RdYlBu_r', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
o_min, o_max = min(o_true.min(),o_pred.min())-2, max(o_true.max(),o_pred.max())+2
ax.plot([o_min,o_max],[o_min,o_max],'k--',linewidth=1)
ax.set_xlabel('True OASPL / dB'); ax.set_ylabel('Predicted OASPL / dB')
ax.set_title('(a) OASPL Prediction Scatter', fontsize=11, fontweight='bold')
ax.set_aspect('equal'); ax.set_xlim(o_min,o_max); ax.set_ylim(o_min,o_max)
cbar=plt.colorbar(sc,ax=ax,shrink=0.8); cbar.set_label('Spectrum MSE',fontsize=8)
ax.text(0.03,0.97,f'MAE={np.mean(np.abs(oaspl_err)):.2f} dB\nRMSE={np.sqrt(np.mean(oaspl_err**2)):.2f} dB\nR2={np.corrcoef(o_true,o_pred)[0,1]**2:.4f}',
        transform=ax.transAxes,va='top',fontsize=8,bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
ax2 = axes[1]
ax2.hist(oaspl_err, bins=30, color='#1565C0', alpha=0.7, edgecolor='white', density=True)
kde = gaussian_kde(oaspl_err)
x_kde = np.linspace(oaspl_err.min(), oaspl_err.max(), 200)
ax2.plot(x_kde, kde(x_kde), 'r-', linewidth=1.5, label='KDE')
ax2.axvline(x=0,color='gray',linestyle='--',linewidth=0.8)
ax2.axvline(x=np.mean(oaspl_err),color='#C62828',linestyle='-',linewidth=1,label=f'Mean={np.mean(oaspl_err):.3f} dB')
ax2.set_xlabel('OASPL Error / dB'); ax2.set_ylabel('Density')
ax2.set_title('(b) OASPL Error Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig04_oaspl_scatter.png'), dpi=300)
plt.close()

# ================================================================
# 图5: Per-Condition MSE 热力图
# ================================================================
print("Fig5: Per-condition heatmap...")
cm = defaultdict(list)
for i, vi in enumerate(val_indices):
    cm[(modes_raw_v4[vi], types_raw_v4[vi])].append(ps_mse[i])
all_modes = sorted(set(modes_raw_v4)); all_types = sorted(set(types_raw_v4))
heatmap = np.full((len(all_modes), len(all_types)), np.nan)
for mi, mode in enumerate(all_modes):
    for ti, typ in enumerate(all_types):
        if (mode, typ) in cm: heatmap[mi, ti] = np.mean(cm[(mode, typ)])
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heatmap, aspect='auto', cmap='RdYlBu_r', vmin=np.nanmin(heatmap), vmax=np.nanpercentile(heatmap[~np.isnan(heatmap)], 95))
ax.set_xticks(range(len(all_types))); ax.set_yticks(range(len(all_modes)))
ax.set_xticklabels(all_types, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(all_modes, fontsize=8)
ax.set_xlabel('Vehicle Type'); ax.set_ylabel('Mode')
ax.set_title('V4 Model: Per-Condition Validation MSE Heatmap', fontsize=12, fontweight='bold')
for mi in range(len(all_modes)):
    for ti in range(len(all_types)):
        v = heatmap[mi,ti]
        if not np.isnan(v):
            ax.text(ti, mi, f'{v:.1f}', ha='center', va='center', fontsize=7, color='white' if v>np.nanmean(heatmap) else 'black')
cbar=plt.colorbar(im,ax=ax,shrink=0.8); cbar.set_label('MSE / dB2',fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig05_per_condition_heatmap.png'), dpi=300)
plt.close()

# ================================================================
# 图6: 误差分布
# ================================================================
print("Fig6: Error distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.hist(ps_mse, bins=35, color='#1565C0', alpha=0.7, edgecolor='white')
ax.axvline(x=np.median(ps_mse), color='#C62828', linestyle='-', linewidth=1.5, label=f'Median={np.median(ps_mse):.2f}')
ax.axvline(x=np.mean(ps_mse), color='#E65100', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(ps_mse):.2f}')
ax.axvline(x=np.percentile(ps_mse,95), color='#7B1FA2', linestyle=':', linewidth=1.5, label=f'P95={np.percentile(ps_mse,95):.2f}')
ax.set_xlabel('Per-Sample MSE / dB2'); ax.set_ylabel('Count')
ax.set_title('(a) Validation MSE Distribution', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax2 = axes[1]
sorted_mse = np.sort(ps_mse); n = len(sorted_mse)
ax2.plot(np.arange(1,n+1)/n, sorted_mse, 'b-', linewidth=1.5)
ax2.axhline(y=np.median(ps_mse), color='#C62828', linestyle='--', linewidth=1, alpha=0.6)
ax2.fill_between([0.5,1.0], 0, sorted_mse.max(), alpha=0.1, color='#FF9800')
ax2.fill_between([0.9,1.0], 0, sorted_mse.max(), alpha=0.15, color='#F44336')
ax2.text(0.75, sorted_mse.max()*0.9, 'Bottom 50%', ha='center', fontsize=8, color='#E65100')
ax2.text(0.95, sorted_mse.max()*0.8, 'P10', ha='center', fontsize=8, color='#C62828')
ax2.set_xlabel('Cumulative Sample Fraction'); ax2.set_ylabel('Per-Sample MSE / dB2')
ax2.set_title('(b) MSE Cumulative Distribution', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig06_error_distribution.png'), dpi=300)
plt.close()

# ================================================================
# 图7: 综合仪表盘
# ================================================================
print("Fig7: Summary dashboard...")
fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, 1)
metrics_names = ['MSE', 'MAE', 'RMSE', 'Med MSE', 'P95 MSE', 'Worst MSE']
metrics_vals = [np.mean(ps_mse), np.mean(np.abs(err)), np.sqrt(np.mean(err**2)),
                np.median(ps_mse), np.percentile(ps_mse,95), np.max(ps_mse)]
bar_colors = ['#1565C0','#1976D2','#1E88E5','#43A047','#FB8C00','#E53935']
bars = ax1.bar(range(6), metrics_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(6)); ax1.set_xticklabels(metrics_names, fontsize=8)
ax1.set_ylabel('dB2 / dB', fontsize=9)
ax1.set_title('(a) Core Metrics Overview', fontsize=11, fontweight='bold')
for bar, val in zip(bars, metrics_vals):
    ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

ax2 = fig.add_subplot(2, 2, 2)
per_f_rmse_arr = np.sqrt(np.mean(err**2, axis=0))
band_vals = [np.mean(per_f_rmse_arr[(freq>=lo)&(freq<=hi)]) for lo,hi in [(20,100),(100,500),(500,2000),(2000,5000)]]
ax2.bar(range(4), band_vals, color=['#1B5E20','#4CAF50','#8BC34A','#CDDC39'], edgecolor='white')
ax2.set_xticks(range(4)); ax2.set_xticklabels(['20-100Hz\n(Low)','100-500Hz\n(Mid-Low)','500-2000Hz\n(Mid)','2000-5000Hz\n(High)'], fontsize=8)
ax2.set_ylabel('RMSE / dB', fontsize=9)
ax2.set_title('(b) Per-Band RMSE', fontsize=11, fontweight='bold')
for i, v in enumerate(band_vals): ax2.text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

ax3 = fig.add_subplot(2, 2, 3)
cond_oaspl = defaultdict(list)
for i, vi in enumerate(val_indices):
    cond_oaspl[f"{modes_raw_v4[vi][:4]}+{types_raw_v4[vi][:8]}"].append(abs(oaspl_err[i]))
cond_sorted = sorted(cond_oaspl.keys(), key=lambda x: np.mean(cond_oaspl[x]))[:12]
ax3.barh(range(12), [np.mean(cond_oaspl[c]) for c in cond_sorted], color='#FF9800', edgecolor='white', alpha=0.8)
ax3.set_yticks(range(12)); ax3.set_yticklabels(cond_sorted, fontsize=7)
ax3.set_xlabel('OASPL MAE / dB', fontsize=9); ax3.invert_yaxis()
ax3.set_title('(c) OASPL MAE by Condition (Best 12)', fontsize=11, fontweight='bold')

ax4 = fig.add_subplot(2, 2, 4)
labels_pie = ['Excellent\n(MSE<0.5)','Good\n(0.5-1.0)','Fair\n(1.0-2.0)','Poor\n(2.0-5.0)','Bad\n(MSE>5.0)']
counts = [np.sum(ps_mse<0.5), np.sum((ps_mse>=0.5)&(ps_mse<1.0)), np.sum((ps_mse>=1.0)&(ps_mse<2.0)), np.sum((ps_mse>=2.0)&(ps_mse<5.0)), np.sum(ps_mse>=5.0)]
ax4.pie(counts, labels=labels_pie, colors=['#1B5E20','#4CAF50','#FFC107','#FF9800','#F44336'],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize':8})
ax4.set_title('(d) Sample Quality Distribution', fontsize=11, fontweight='bold')

fig.suptitle('V4 Model Validation Set Evaluation Dashboard (1 Shared Head)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V4_fig07_summary_dashboard.png'), dpi=300)
plt.close()

print(f"\nAll V4 plots saved to: {out_dir}")
print("Done!")
