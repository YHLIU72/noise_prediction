"""检查 V15 训练结果"""
import os, glob, torch, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = r'f:\lyh\paddlespeech\papernoise\physic'
out_path = os.path.join(base, 'v15_result_summary.txt')
out_lines = []

def log(msg):
    print(msg)
    out_lines.append(msg)

runs = sorted(glob.glob(os.path.join(base, 'runs', 'pi_mbcn_v15_*')))
log("V15 runs found:")
for r in runs:
    log(f"  {os.path.basename(r)}")

if runs:
    run_dir = runs[-1]
    ckpt_path = os.path.join(run_dir, 'models', 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        log(f"\nBest checkpoint: epoch={ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f}")
        log(f"Has norm params: {'input_mean' in ckpt and 'input_std' in ckpt}")
        if 'input_mean' in ckpt:
            log(f"input_mean: {ckpt['input_mean'].numpy()}")
            log(f"input_std:  {ckpt['input_std'].numpy()}")

    ef = glob.glob(os.path.join(run_dir, 'events.out.*'))
    if ef:
        ea = EventAccumulator(ef[0], size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        log(f"\nTensorBoard tags ({len(tags)}):")
        for tag in sorted(tags):
            events = ea.Scalars(tag)
            vals = [(e.step, e.value) for e in events]
            if vals:
                min_idx = np.argmin([v[1] for v in vals])
                log(f"  {tag}: best={vals[min_idx][1]:.4f} @ep{vals[min_idx][0]}, "
                    f"final={vals[-1][1]:.4f} @ep{vals[-1][0]}")

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print(f"Results written to {out_path}")
