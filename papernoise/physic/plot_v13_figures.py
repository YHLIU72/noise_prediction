"""
V13 模型预测效果图 — 对数频率重采样方案
批量保存至 plot_thesis/ 文件夹 (V13_ 前缀)
"""
import os, sys, warnings, io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'; os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import torch, numpy as np, importlib.util, pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==== 样式设置 ====
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
os.makedirs(out_dir, exist_ok=True)

def load_module(path):
    name = os.path.basename(path).replace('.py','')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ===== 加载 V13 数据 =====
print("Loading V13 data pipeline...")
from PIMBCN_data_0614_v13 import PIMBCNDataset
train_ds = PIMBCNDataset(directory_path=data_dir, is_validation=False, augment=False)
val_ds = train_ds.get_validation_dataset()
val_loader = DataLoader(val_ds, batch_size=214, shuffle=False)

types_raw = val_ds.data.iloc[:, 3].values.astype(str)
modes_raw = val_ds.data.iloc[:, 2].values.astype(str)
freq = val_ds.freq_axis  # V13: logspace(20, 5000, 1246)

print(f"  Frequency axis: {freq[0]:.1f} ~ {freq[-1]:.0f} Hz, {len(freq)} points (log-spaced)")
print(f"  Validation samples: {len(val_ds)}")

# ===== 加载 V13 模型 =====
print("Loading V13 model...")
net_v13 = load_module(os.path.join(base, "PIMBCN_net0614_v13.py"))
ckpt_path = os.path.join(base, "runs", "pi_mbcn_v13_20260615_141745", "models", "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device)
model = net_v13.PI_MBCN(num_modes=4, num_types=13, freq_bins=1246).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,} | Checkpoint epoch: {ckpt.get('epoch', '?')}")

# ===== 批量推理 =====
for batch in val_loader:
    inputs, type_idx, mode_idx, oaspl_v, octave_v, spectrum = batch
    break
inputs = inputs.to(device); type_idx = type_idx.to(device); mode_idx = mode_idx.to(device)
target_all = spectrum.numpy()
val_indices = val_ds.indices

with torch.no_grad():
    pred_all = model(inputs, mode_idx, type_idx).cpu().numpy()

print(f"  Pred shape: {pred_all.shape}, Target shape: {target_all.shape}")

# ===== 指标计算 =====
err = pred_all - target_all
ps_mse = np.mean(err**2, axis=1)

def oaspl(s):
    shifted = s - np.max(s, axis=1, keepdims=True)
    return 10*np.log10(np.sum(np.power(10., shifted/10.), axis=1)+1e-10) + np.max(s, axis=1)
o_pred = oaspl(pred_all); o_true = oaspl(target_all)
oaspl_err = o_pred - o_true
peak_biases = np.array([pred_all[i, np.argmax(target_all[i])] - target_all[i, np.argmax(target_all[i])] for i in range(len(pred_all))])

print(f"\n=== V13 验证集指标 ===")
print(f"  MSE={np.mean(err**2):.4f}  MAE={np.mean(np.abs(err)):.4f}  RMSE={np.sqrt(np.mean(err**2)):.4f}")
print(f"  CosSim={np.mean([np.dot(p,t)/(np.linalg.norm(p)*np.linalg.norm(t)+1e-10) for p,t in zip(pred_all,target_all)]):.4f}")
print(f"  Median MSE={np.median(ps_mse):.4f}  P95={np.percentile(ps_mse,95):.4f}  Worst={np.max(ps_mse):.4f}")
print(f"  OASPL MAE={np.mean(np.abs(oaspl_err)):.3f} dB  RMSE={np.sqrt(np.mean(oaspl_err**2)):.3f} dB")
print(f"  Peak Bias={np.mean(peak_biases):+.2f} dB")

# ================================================================
# 图1: 最佳/中等/最差 三个典型样本
# ================================================================
print("\nFig1: Best/Median/Worst prediction curves...")
sorted_idx = np.argsort(ps_mse)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, s_idx, title, color in zip(axes, 
    [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]],
    ['(a) Best Prediction', '(b) Median Prediction', '(c) Worst Prediction'],
    ['#2E7D32', '#1565C0', '#C62828']):
    vi = val_indices[s_idx]
    ax.semilogx(freq, target_all[s_idx], color='#333333', linewidth=1.8, label='Ground Truth')
    ax.semilogx(freq, pred_all[s_idx], color=color, linewidth=1.8, label='Prediction', alpha=0.85)
    ax.fill_between(freq, target_all[s_idx], pred_all[s_idx], alpha=0.15, color=color)
    ax.set_xlabel('Frequency / Hz', fontsize=9)
    ax.set_ylabel('SPL / dB', fontsize=9)
    ax.set_title(f'{title}\n{types_raw[vi]} ({modes_raw[vi][:8]})  MSE={ps_mse[s_idx]:.2f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7); ax.set_xlim(20, 5000); ax.tick_params(labelsize=8)
plt.suptitle('V13 (Log-Freq Resampling): Best / Median / Worst Validation Samples', fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig01_best_median_worst.png'), dpi=300)
plt.close()
print("  -> V13_fig01_best_median_worst.png")

# ================================================================
# 图2: 3x3 随机样本网格
# ================================================================
print("Fig2: Multi-condition prediction grid...")
np.random.seed(42)
rand_idx = np.random.choice(len(pred_all), 9, replace=False)
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, s_idx in enumerate(rand_idx):
    ax = axes.flat[i]; vi = val_indices[s_idx]
    ax.semilogx(freq, target_all[s_idx], color='#333333', linewidth=1.3, label='Ground Truth')
    ax.semilogx(freq, pred_all[s_idx], color='#1565C0', linewidth=1.3, label='Prediction', alpha=0.8)
    ax.fill_between(freq, target_all[s_idx], pred_all[s_idx], alpha=0.08, color='#1565C0')
    ax.set_title(f'{types_raw[vi][:8]} | {modes_raw[vi][:8]}\nMSE={ps_mse[s_idx]:.2f}', fontsize=8)
    ax.set_xlim(20, 5000); ax.tick_params(labelsize=6)
    if i >= 6: ax.set_xlabel('Frequency / Hz', fontsize=7)
    if i % 3 == 0: ax.set_ylabel('SPL / dB', fontsize=7)
lines1, _ = axes.flat[0].get_legend_handles_labels()
fig.legend(lines1, ['Ground Truth', 'Prediction'], loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('V13 (Log-Freq Resampling): Multi-Condition Prediction Grid', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig02_multi_condition_grid.png'), dpi=300)
plt.close()
print("  -> V13_fig02_multi_condition_grid.png")

# ================================================================
# 图3: 逐频率 RMSE + Bias
# ================================================================
print("Fig3: Per-frequency error analysis...")
per_f_rmse = np.sqrt(np.mean(err**2, axis=0))
per_f_bias = np.mean(err, axis=0)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax = axes[0]
ax.semilogx(freq, per_f_rmse, color='#C62828', linewidth=1.2, label='RMSE')
ax.set_ylabel('Error / dB', fontsize=10); ax.legend(fontsize=9)
ax.set_title('(a) Per-Frequency Validation RMSE (V13 Log-Freq)', fontsize=11, fontweight='bold')
bands = [(20,100),(100,500),(500,2000),(2000,5000)]
colors_b = ['#E8F5E9','#FFF3E0','#E3F2FD','#F3E5F5']
labels_b = ['Low (20-100Hz)','Mid-Low (100-500Hz)','Mid (500-2000Hz)','High (2000-5000Hz)']
for (lo,hi),c,l in zip(bands,colors_b,labels_b):
    ax.axvspan(lo,hi,alpha=0.2,color=c,zorder=0)
    ax.text(np.sqrt(lo*hi), ax.get_ylim()[1]*0.95, l[:12], ha='center',va='top',fontsize=7,
            bbox=dict(boxstyle='round',facecolor=c,alpha=0.7))
# Add band RMSE annotations
for (lo,hi),l in zip(bands,['20-100','100-500','500-2k','2k-5k']):
    m = (freq>=lo)&(freq<=hi)
    band_rmse = np.mean(per_f_rmse[m])
    ax.text(np.sqrt(lo*hi), 0.5, f'{band_rmse:.2f} dB', ha='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

ax2 = axes[1]
ax2.semilogx(freq, per_f_bias, color='#E65100', linewidth=1.2)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax2.fill_between(freq, 0, per_f_bias, where=(per_f_bias>0), alpha=0.2, color='#C62828', label='Over-predict')
ax2.fill_between(freq, 0, per_f_bias, where=(per_f_bias<0), alpha=0.2, color='#1565C0', label='Under-predict')
ax2.set_xlabel('Frequency / Hz', fontsize=10); ax2.set_ylabel('Bias / dB', fontsize=10)
ax2.set_title('(b) Per-Frequency Prediction Bias (V13 Log-Freq)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
for ax in axes: ax.set_xlim(20,5000); ax.tick_params(labelsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig03_per_frequency_error.png'), dpi=300)
plt.close()
print("  -> V13_fig03_per_frequency_error.png")

# ================================================================
# 图4: OASPL 散点图 + 误差分布
# ================================================================
print("Fig4: OASPL scatter + error distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
sc = ax.scatter(o_true, o_pred, c=ps_mse, cmap='RdYlBu_r', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
o_min, o_max = min(o_true.min(),o_pred.min())-2, max(o_true.max(),o_pred.max())+2
ax.plot([o_min,o_max],[o_min,o_max],'k--',linewidth=1)
ax.set_xlabel('True OASPL / dB'); ax.set_ylabel('Predicted OASPL / dB')
ax.set_title('(a) OASPL Prediction Scatter (V13)', fontsize=11, fontweight='bold')
ax.set_aspect('equal'); ax.set_xlim(o_min,o_max); ax.set_ylim(o_min,o_max)
cbar=plt.colorbar(sc,ax=ax,shrink=0.8); cbar.set_label('Spectrum MSE',fontsize=8)
ax.text(0.03,0.97,f'MAE={np.mean(np.abs(oaspl_err)):.3f} dB\nRMSE={np.sqrt(np.mean(oaspl_err**2)):.3f} dB\nR2={np.corrcoef(o_true,o_pred)[0,1]**2:.4f}',
        transform=ax.transAxes,va='top',fontsize=8,bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
ax2 = axes[1]
ax2.hist(oaspl_err, bins=30, color='#1565C0', alpha=0.7, edgecolor='white', density=True)
kde = gaussian_kde(oaspl_err)
x_kde = np.linspace(oaspl_err.min(), oaspl_err.max(), 200)
ax2.plot(x_kde, kde(x_kde), 'r-', linewidth=1.5, label='KDE')
ax2.axvline(x=0,color='gray',linestyle='--',linewidth=0.8)
ax2.axvline(x=np.mean(oaspl_err),color='#C62828',linestyle='-',linewidth=1,label=f'Mean={np.mean(oaspl_err):.3f} dB')
ax2.set_xlabel('OASPL Error / dB'); ax2.set_ylabel('Density')
ax2.set_title('(b) OASPL Error Distribution (V13)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig04_oaspl_scatter.png'), dpi=300)
plt.close()
print("  -> V13_fig04_oaspl_scatter.png")

# ================================================================
# 图5: Per-Condition MSE 热力图
# ================================================================
print("Fig5: Per-condition MSE heatmap...")
cm = defaultdict(list)
for i, vi in enumerate(val_indices):
    cm[(modes_raw[vi], types_raw[vi])].append(ps_mse[i])
all_modes = sorted(set(modes_raw)); all_types = sorted(set(types_raw))
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
ax.set_title('V13 (Log-Freq Resampling): Per-Condition Validation MSE Heatmap', fontsize=12, fontweight='bold')
for mi in range(len(all_modes)):
    for ti in range(len(all_types)):
        v = heatmap[mi,ti]
        if not np.isnan(v):
            ax.text(ti, mi, f'{v:.1f}', ha='center', va='center', fontsize=7, color='white' if v>np.nanmean(heatmap) else 'black')
cbar=plt.colorbar(im,ax=ax,shrink=0.8); cbar.set_label('MSE',fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig05_per_condition_heatmap.png'), dpi=300)
plt.close()
print("  -> V13_fig05_per_condition_heatmap.png")

# ================================================================
# 图6: 误差分布直方图 + 累积分布
# ================================================================
print("Fig6: Error distribution + cumulative...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.hist(ps_mse, bins=35, color='#1565C0', alpha=0.7, edgecolor='white')
ax.axvline(x=np.median(ps_mse), color='#C62828', linestyle='-', linewidth=1.5, label=f'Median={np.median(ps_mse):.2f}')
ax.axvline(x=np.mean(ps_mse), color='#E65100', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(ps_mse):.2f}')
ax.axvline(x=np.percentile(ps_mse,95), color='#7B1FA2', linestyle=':', linewidth=1.5, label=f'P95={np.percentile(ps_mse,95):.2f}')
ax.set_xlabel('Per-Sample MSE'); ax.set_ylabel('Count')
ax.set_title('(a) Validation MSE Distribution (V13)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax2 = axes[1]
sorted_mse = np.sort(ps_mse); n = len(sorted_mse)
ax2.plot(np.arange(1,n+1)/n, sorted_mse, 'b-', linewidth=1.5)
ax2.axhline(y=np.median(ps_mse), color='#C62828', linestyle='--', linewidth=1, alpha=0.6)
ax2.fill_between([0.5,1.0], 0, sorted_mse.max(), alpha=0.1, color='#FF9800')
ax2.fill_between([0.9,1.0], 0, sorted_mse.max(), alpha=0.15, color='#F44336')
ax2.text(0.75, sorted_mse.max()*0.9, 'Bottom 50%', ha='center', fontsize=8, color='#E65100')
ax2.text(0.95, sorted_mse.max()*0.8, 'P10', ha='center', fontsize=8, color='#C62828')
ax2.set_xlabel('Cumulative Sample Fraction'); ax2.set_ylabel('Per-Sample MSE')
ax2.set_title('(b) MSE Cumulative Distribution (V13)', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig06_error_distribution.png'), dpi=300)
plt.close()
print("  -> V13_fig06_error_distribution.png")

# ================================================================
# 图7: 综合仪表盘 (4合1)
# ================================================================
print("Fig7: Comprehensive dashboard...")
fig = plt.figure(figsize=(14, 10))

# 7a: Best/Worst curves
ax1 = plt.subplot(2, 2, 1)
for s_idx, color, label in [(sorted_idx[0], '#2E7D32', f'Best (MSE={ps_mse[sorted_idx[0]]:.1f})'),
                               (sorted_idx[-1], '#C62828', f'Worst (MSE={ps_mse[sorted_idx[-1]]:.1f})')]:
    ax1.semilogx(freq, target_all[s_idx], color='#333', linewidth=1.2, alpha=0.7)
    ax1.semilogx(freq, pred_all[s_idx], color=color, linewidth=1.5, label=label)
ax1.set_xlabel('Frequency / Hz'); ax1.set_ylabel('SPL / dB')
ax1.set_title('Best vs Worst Prediction Curves', fontweight='bold')
ax1.legend(fontsize=7); ax1.set_xlim(20, 5000)

# 7b: Per-frequency RMSE (log-scaled x)
ax2 = plt.subplot(2, 2, 2)
ax2.semilogx(freq, per_f_rmse, color='#C62828', linewidth=1)
for (lo,hi),c in zip(bands,colors_b):
    ax2.axvspan(lo,hi,alpha=0.15,color=c,zorder=0)
ax2.set_xlabel('Frequency / Hz'); ax2.set_ylabel('RMSE / dB')
ax2.set_title('Per-Frequency RMSE', fontweight='bold')
ax2.set_xlim(20, 5000)

# 7c: OASPL scatter
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(o_true, o_pred, c=ps_mse, cmap='RdYlBu_r', alpha=0.6, s=20, edgecolors='white', linewidth=0.2)
ax3.plot([o_min,o_max],[o_min,o_max],'k--',linewidth=0.8)
ax3.set_xlabel('True OASPL / dB'); ax3.set_ylabel('Predicted OASPL / dB')
ax3.set_title(f'OASPL: MAE={np.mean(np.abs(oaspl_err)):.3f} dB', fontweight='bold')
ax3.set_aspect('equal')

# 7d: MSE cumulative
ax4 = plt.subplot(2, 2, 4)
ax4.plot(np.arange(1,n+1)/n, sorted_mse, 'b-', linewidth=1.2)
ax4.axhline(y=np.median(ps_mse), color='#C62828', linestyle='--', linewidth=0.8)
ax4.set_xlabel('Cumulative Fraction'); ax4.set_ylabel('Per-Sample MSE')
ax4.set_title(f'MSE Distribution: Median={np.median(ps_mse):.2f}', fontweight='bold')

plt.suptitle('V13 Model (Log-Frequency Resampling) — Comprehensive Evaluation Dashboard', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'V13_fig07_dashboard.png'), dpi=300)
plt.close()
print("  -> V13_fig07_dashboard.png")

# ================================================================
print(f"\nAll figures saved to: {out_dir}/")
print("Generated files:")
for f in sorted(os.listdir(out_dir)):
    if f.startswith('V13_'):
        print(f"  {f}")
print("Done!")
