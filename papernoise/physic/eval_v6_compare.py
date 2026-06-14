"""
Quick V4/V5/V6 comparison evaluation
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Simple dataset that loads all data
# ============================================================
class EvalDataset(Dataset):
    def __init__(self, directory_path, is_train=True, val_split=0.2, seed=42):
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        dfs = [pd.read_csv(os.path.join(directory_path, f)) for f in csv_files]
        data = pd.concat(dfs, ignore_index=True)
        print(f"Total records: {len(data)}")
        
        # Parse data
        self.inputs = data.iloc[:, [4,5,6]].values.astype(np.float32)
        self.types = LabelEncoder().fit_transform(data.iloc[:, 3]).astype(np.int64)
        self.modes = LabelEncoder().fit_transform(data.iloc[:, 2]).astype(np.int64)
        
        spectra = []
        for val in data.iloc[:, 13]:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            arr = np.array(parsed, dtype=np.float32)
            if len(arr) >= 2501: arr = arr[:2501]
            else: arr = np.pad(arr, (0, 2501-len(arr)), 'edge')
            spectra.append(arr[5:1251])  # 20~5000Hz, 1246 points
        self.spectra = np.array(spectra, dtype=np.float32)
        
        # Split
        np.random.seed(seed)
        n = len(data)
        idx = np.random.permutation(n)
        split = int(n * (1 - val_split))
        if is_train:
            self.indices = idx[:split]
        else:
            self.indices = idx[split:]
        
        # Normalize inputs
        self.input_mean = self.inputs[self.indices].mean(axis=0)
        self.input_std = self.inputs[self.indices].std(axis=0) + 1e-8
        print(f"{'Train' if is_train else 'Val'} samples: {len(self.indices)}")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = (self.inputs[i] - self.input_mean) / self.input_std
        return (torch.from_numpy(x).float(),
                torch.tensor(self.types[i], dtype=torch.long),
                torch.tensor(self.modes[i], dtype=torch.long),
                torch.tensor(0), torch.tensor(0.0),
                torch.from_numpy(self.spectra[i]).float())

# ============================================================
# Load datasets
# ============================================================
data_dir = r"F:\lyh\paddlespeech\csvdata333"
train_ds = EvalDataset(data_dir, is_train=True)
val_ds = EvalDataset(data_dir, is_train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, pin_memory=True)

# ============================================================
# Import models
# ============================================================
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

net_v4 = load_module("net_v4", r"f:\lyh\paddlespeech\papernoise\physic\PIMBCN_net0612_v4.py")
net_v5 = load_module("net_v5", r"f:\lyh\paddlespeech\papernoise\physic\PIMBCN_net0612_v5.py")
net_v6 = load_module("net_v6", r"f:\lyh\paddlespeech\papernoise\physic\PIMBCN_net0612_v6.py")

versions = {
    "V4": (r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_v4_20260611_215207\models\best_model.pth", net_v4.PI_MBCN),
    "V5": (r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_v5_20260612_141720\models\best_model.pth", net_v5.PI_MBCN),
    "V6": (r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_v6_20260612_154757\models\best_model.pth", net_v6.PI_MBCN),
}

results = {}

for name, (model_path, ModelClass) in versions.items():
    print(f"\n{'='*50}")
    print(f"Evaluating {name}...")
    
    model = ModelClass(num_modes=4, num_types=13, freq_bins=1246).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    
    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    
    # ---- Validation ----
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, types, modes, _, _, target in val_loader:
            inputs = inputs.to(device); types = types.to(device)
            modes = modes.to(device); target = target.to(device)
            pred = model(inputs, modes, types)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    val_preds = np.concatenate(all_preds)
    val_targets = np.concatenate(all_targets)
    
    # ---- Training ----
    all_preds_t, all_targets_t = [], []
    with torch.no_grad():
        for inputs, types, modes, _, _, target in train_loader:
            inputs = inputs.to(device); types = types.to(device)
            modes = modes.to(device); target = target.to(device)
            pred = model(inputs, modes, types)
            all_preds_t.append(pred.cpu().numpy())
            all_targets_t.append(target.cpu().numpy())
    train_preds = np.concatenate(all_preds_t)
    train_targets = np.concatenate(all_targets_t)
    
    val_mse = np.mean((val_preds - val_targets)**2)
    val_mae = np.mean(np.abs(val_preds - val_targets))
    val_rmse = np.sqrt(val_mse)
    train_mse = np.mean((train_preds - train_targets)**2)
    
    # Per-sample stats
    per_sample = np.mean((val_preds - val_targets)**2, axis=1)
    worst_idx = np.argmax(per_sample)
    best_idx = np.argmin(per_sample)
    
    # Cosine similarity
    dot = np.sum(val_preds * val_targets, axis=1)
    norm_p = np.linalg.norm(val_preds, axis=1)
    norm_t = np.linalg.norm(val_targets, axis=1)
    cos_sim = np.mean(dot / (norm_p * norm_t + 1e-10))
    
    # R²
    ss_res = np.sum((val_targets - val_preds)**2)
    ss_tot = np.sum((val_targets - np.mean(val_targets))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    results[name] = {
        'params': n_params,
        'train_mse': train_mse, 'val_mse': val_mse, 'val_mae': val_mae,
        'val_rmse': val_rmse, 'cos_sim': cos_sim, 'r2': r2,
        'worst_mse': per_sample[worst_idx], 'best_mse': per_sample[best_idx],
        'gap_ratio': train_mse / val_mse if val_mse > 0 else 0,
        'per_sample_mse': per_sample,
    }
    
    print(f"  Params: {n_params:,}")
    print(f"  Train MSE: {train_mse:.4f}  |  Val MSE: {val_mse:.4f}  |  Val MAE: {val_mae:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}  |  Cosine: {cos_sim:.4f}  |  R²: {r2:.4f}")
    print(f"  Gap ratio: {train_mse/val_mse:.2f}x")
    print(f"  Worst/Best sample MSE: {per_sample[worst_idx]:.2f} / {per_sample[best_idx]:.2f}")

# ============================================================
# Summary table
# ============================================================
print(f"\n{'='*80}")
print(f"  SUMMARY: V4 → V5 → V6 对比")
print(f"{'='*80}")
print(f"{'Metric':<22s} {'V4':>10s} {'V5':>10s} {'V6':>10s} {'V5vsV4':>10s} {'V6vsV4':>10s}")
print(f"{'-'*72}")
for metric, label in [('val_mse', 'Val MSE (↓)'), ('val_mae', 'Val MAE (↓)'), 
                       ('val_rmse', 'Val RMSE (↓)'), ('train_mse', 'Train MSE (↓)'),
                       ('cos_sim', 'Cosine Sim (↑)'), ('r2', 'R² (↑)'),
                       ('worst_mse', 'Worst MSE (↓)'), ('gap_ratio', 'Train/Val Gap')]:
    v4 = results['V4'][metric]
    v5 = results['V5'][metric]
    v6 = results['V6'][metric]
    c5 = f"{(v5-v4)/abs(v4)*100:+.1f}%" if v4 != 0 else "N/A"
    c6 = f"{(v6-v4)/abs(v4)*100:+.1f}%" if v4 != 0 else "N/A"
    print(f"{label:<22s} {v4:10.4f} {v5:10.4f} {v6:10.4f} {c5:>10s} {c6:>10s}")

# ============================================================
# Per-sample distribution comparison
# ============================================================
print(f"\n{'='*80}")
print(f"  逐样本MSE分布统计")
print(f"{'='*80}")
for name in ['V4', 'V5', 'V6']:
    mse = results[name]['per_sample_mse']
    print(f"  {name}: mean={np.mean(mse):.4f}, median={np.median(mse):.4f}, "
          f"std={np.std(mse):.4f}, min={np.min(mse):.4f}, max={np.max(mse):.4f}, "
          f"P95={np.percentile(mse, 95):.4f}")

# Identify samples where V6 is better/worse than V4
v4_mse = results['V4']['per_sample_mse']
v6_mse = results['V6']['per_sample_mse']
diffs = v6_mse - v4_mse
better = np.sum(diffs < 0)
worse = np.sum(diffs > 0)
print(f"\n  V6 vs V4 per-sample: {better} better, {worse} worse (out of {len(diffs)})")
print(f"  Mean Δ: {np.mean(diffs):+.4f}, Median Δ: {np.median(diffs):+.4f}")
print(f"  V6 比 V4 好的样本平均改善: {np.mean(diffs[diffs<0]):.4f}" if better > 0 else "")
print(f"  V6 比 V4 差的样本平均恶化: {np.mean(diffs[diffs>0]):.4f}" if worse > 0 else "")
