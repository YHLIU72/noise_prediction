"""V15 模型评估脚本 — 混合频率采样 (低频对数+高频线性)"""
import os, sys, io, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import torch, numpy as np, importlib.util
from torch.utils.data import DataLoader
from collections import defaultdict
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = r"f:\lyh\paddlespeech\papernoise\physic"
data_dir = r"F:\lyh\paddlespeech\csvdata333"
out_file = os.path.join(base, 'v15_eval_results.txt')
results_lines = []
def log(msg):
    print(msg)
    results_lines.append(msg)

log(f"Device: {device}")

# ===== 加载 V15 数据 =====
from PIMBCN_data_0713_v15 import PIMBCNDataset
train_ds = PIMBCNDataset(directory_path=data_dir, is_validation=False, augment=False)
val_ds = train_ds.get_validation_dataset()
val_loader = DataLoader(val_ds, batch_size=214, shuffle=False)
freq = val_ds.freq_axis
log(f"V15 混合频率轴: {len(freq)} 点, {freq[0]:.1f}~{freq[-1]:.0f}Hz")

types_raw = val_ds.data.iloc[:, 3].values.astype(str)
modes_raw = val_ds.data.iloc[:, 2].values.astype(str)
val_indices = val_ds.indices

# ===== 加载 V15 模型 =====
net_mod_path = os.path.join(base, "PIMBCN_net0713_v15.py")
spec = importlib.util.spec_from_file_location("net_v15", net_mod_path)
net_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_mod)

# 使用最新的完整训练 run
runs = [d for d in os.listdir(os.path.join(base, "runs")) if d.startswith("pi_mbcn_v15_")]
runs.sort()
run_dir = runs[-1]
ckpt_path = os.path.join(base, "runs", run_dir, "models", "best_model.pth")
log(f"Checkpoint: {ckpt_path}")
# [修复] 用 CPU 加载 checkpoint, 避免 CUDA 张量 .numpy() 崩溃
ckpt = torch.load(ckpt_path, map_location='cpu')
log(f"Checkpoint epoch: {ckpt.get('epoch')}, best_val_loss: {ckpt.get('best_val_loss'):.4f}")
if 'input_mean' in ckpt:
    log(f"checkpoint norm: mean={ckpt['input_mean'].numpy()}, std={ckpt['input_std'].numpy()}")

model = net_mod.PI_MBCN(num_modes=4, num_types=13, freq_bins=1246).to(device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# ===== 推理 =====
for batch in val_loader:
    inputs, type_idx, mode_idx, _, _, spectrum = batch
    break
inputs = inputs.to(device); type_idx = type_idx.to(device); mode_idx = mode_idx.to(device)
target_all = spectrum.numpy()

with torch.no_grad():
    pred_all = model(inputs, mode_idx, type_idx).cpu().numpy()

log(f"Pred: {pred_all.shape}, Target: {target_all.shape}")

# ===== 指标计算 =====
err = pred_all - target_all
mse = np.mean(err**2)
mae = np.mean(np.abs(err))
rmse = np.sqrt(mse)
ps_mse = np.mean(err**2, axis=1)

dot = np.sum(pred_all * target_all, axis=1)
cos_sim = np.mean(dot / (np.linalg.norm(pred_all, axis=1) * np.linalg.norm(target_all, axis=1) + 1e-10))
ss_r = np.sum(err**2); ss_t = np.sum((target_all - np.mean(target_all))**2)
r2 = 1 - ss_r / (ss_t + 1e-10)
pears = np.mean([pearsonr(pred_all[i], target_all[i])[0] for i in range(len(pred_all))])

def oaspl(s):
    shifted = s - np.max(s, axis=1, keepdims=True)
    return 10*np.log10(np.sum(np.power(10., shifted/10.), axis=1)+1e-10) + np.max(s, axis=1)
o_pred = oaspl(pred_all); o_true = oaspl(target_all)
oaspl_mae = np.mean(np.abs(o_pred - o_true))

peak_biases = np.array([pred_all[i, np.argmax(target_all[i])] - target_all[i, np.argmax(target_all[i])] for i in range(len(pred_all))])

log(f"\n=== V15 验证集指标 (混合频率域) ===")
log(f"Val MSE:  {mse:.4f}")
log(f"Val MAE:  {mae:.4f}")
log(f"Val RMSE: {rmse:.4f}")
log(f"CosSim:   {cos_sim:.4f}")
log(f"R2:       {r2:.4f}")
log(f"Pearson:  {pears:.4f}")
log(f"Median:   {np.median(ps_mse):.4f}")
log(f"P95:      {np.percentile(ps_mse, 95):.4f}")
log(f"Worst:    {np.max(ps_mse):.4f}")
log(f"OASPL MAE: {oaspl_mae:.3f} dB")
log(f"Peak Bias: {np.mean(peak_biases):+.2f} dB")

# ===== 分频段 RMSE =====
bands = [(20, 100, '20-100Hz'), (100, 300, '100-300Hz'), (300, 800, '300-800Hz'),
         (800, 2000, '800-2000Hz'), (2000, 3500, '2000-3500Hz'), (3500, 5000, '3500-5000Hz')]
log(f"\n=== 分频段 RMSE (V15 混合域) ===")
for lo, hi, label in bands:
    mask = (freq >= lo) & (freq <= hi)
    band_rmse = np.sqrt(np.mean(err[:, mask]**2))
    band_bias = np.mean(err[:, mask])
    log(f"  {label}: RMSE={band_rmse:.4f}, Bias={band_bias:+.4f}")

# ===== 对比 V4/V13 (从上次评估读取) =====
log(f"\n=== 对比: V4 vs V13 vs V15 ===")
log(f"  模型    Val_MSE   Val_MAE   Val_Med   Worst   OASPL_MAE")
log(f"  V4     1.5216    0.8895    0.9261    10.93   0.517")
log(f"  V13    1.4962    0.8721    1.0021    11.72   0.487")
log(f"  V15    {mse:.4f}    {mae:.4f}    {np.median(ps_mse):.4f}    {np.max(ps_mse):.2f}   {oaspl_mae:.3f}")

# ===== 分频段对比 =====
log(f"\n=== 分频段 RMSE 对比 (V4线性 vs V13对数 vs V15混合) ===")
log(f"  频段        V4       V13      V15")
v4_bands = { '20-100Hz': 2.3238, '100-300Hz': 1.3817, '300-800Hz': 1.0957,
             '800-2000Hz': 1.1700, '2000-3500Hz': 1.2332, '3500-5000Hz': 1.1936 }
v13_bands = { '20-100Hz': 1.1774, '100-300Hz': 1.2058, '300-800Hz': 1.4496,
              '800-2000Hz': 1.2974, '2000-3500Hz': 1.1094, '3500-5000Hz': 1.1713 }
for lo, hi, label in bands:
    mask = (freq >= lo) & (freq <= hi)
    v15_band = np.sqrt(np.mean(err[:, mask]**2))
    log(f"  {label:<15} {v4_bands[label]:.4f}   {v13_bands[label]:.4f}   {v15_band:.4f}")

# ===== 分工况 =====
cm = defaultdict(list)
for i, vi in enumerate(val_indices):
    cm[(modes_raw[vi], types_raw[vi])].append(ps_mse[i])
best_cond = min(cm.items(), key=lambda x: np.mean(x[1]))
worst_cond = max(cm.items(), key=lambda x: np.mean(x[1]))
log(f"\n=== 分工况 ===")
log(f"  最好工况: {best_cond[0][0]}+{best_cond[0][1]} (MSE={np.mean(best_cond[1]):.2f})")
log(f"  最差工况: {worst_cond[0][0]}+{worst_cond[0][1]} (MSE={np.mean(worst_cond[1]):.2f})")

# ===== 结论 =====
v4_mse, v13_mse = 1.5216, 1.4962
log(f"\n=== 结论 ===")
log(f"  V15 vs V4:  {(v4_mse - mse)/v4_mse*100:+.1f}%")
log(f"  V15 vs V13: {(v13_mse - mse)/v13_mse*100:+.1f}%")

with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results_lines))
print(f"\nResults saved to: {out_file}")
