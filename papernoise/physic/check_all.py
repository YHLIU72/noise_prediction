import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 检查所有运行中的训练
base = r"f:\lyh\paddlespeech\papernoise\physic\runs"
runs_found = []
for d in os.listdir(base):
    dp = os.path.join(base, d)
    if not os.path.isdir(dp): continue
    ef = glob.glob(os.path.join(dp, "events.out.*"))
    if not ef: continue
    ea = EventAccumulator(ef[0], size_guidance={"scalars": 0})
    ea.Reload()
    val_tags = sorted([t for t in ea.Tags().get("scalars", []) if "val" in t.lower()])
    if not val_tags: continue
    # get latest epoch and val_mse_db
    mse_tag = [t for t in val_tags if "mse_db" in t.lower() or "mse_spec" in t.lower()]
    if not mse_tag: continue
    events = ea.Scalars(mse_tag[0])
    vals = [(e.step, e.value) for e in events]
    if not vals: continue
    min_step, min_val = min(vals, key=lambda x: x[1])
    latest_step = vals[-1][0]
    latest_val = vals[-1][1]
    trend = "↓" if latest_val < min_val * 1.05 else ("↑过拟合" if latest_val > min_val * 1.1 else "→横盘")
    runs_found.append((os.path.basename(d), min_step, min_val, latest_step, latest_val, trend))

runs_found.sort(key=lambda x: x[2])  # sort by val_mse
print(f"{'Run':<50} {'best@':>6} {'val_mse':>8} {'now@':>6} {'now_mse':>8} {'趋势':>8}")
print("-"*95)
for name, bs, bv, ls, lv, tr in runs_found:
    print(f"{name:<50} {bs:>6} {bv:>8.4f} {ls:>6} {lv:>8.4f} {tr:>8}")
