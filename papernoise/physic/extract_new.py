import os, json, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "V0610_lossV2": r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_lossV2_20260610_201508_epochs50000_bs8_wd0.001",
    "V0610_final":  r"f:\lyh\paddlespeech\papernoise\physic\runs\pi_mbcn_final_20260610_230212",
}

for ver, run_path in RUNS.items():
    print(f"\n{'='*60}")
    print(f"  {ver}: {os.path.basename(run_path)}")
    print(f"{'='*60}")
    if not os.path.isdir(run_path):
        print("  [ERROR] 目录不存在")
        continue
    ef = glob.glob(os.path.join(run_path, "events.out.*"))
    if not ef:
        print("  [ERROR] 无事件文件")
        continue
    ea = EventAccumulator(ef[0], size_guidance={"scalars": 0})
    ea.Reload()
    val_tags = sorted([t for t in ea.Tags().get("scalars", []) if "val" in t.lower()])
    if not val_tags:
        print("  [WARN] 无验证标量")
        continue
    for tag in val_tags:
        events = ea.Scalars(tag)
        vals = [(e.step, e.value) for e in events]
        if not vals: continue
        min_step, min_val = min(vals, key=lambda x: x[1])
        last = vals[-5:]
        latest_step = vals[-1][0]
        print(f"  {tag}:")
        print(f"    min={min_val:.4f} @epoch={min_step}")
        print(f"    latest epoch={latest_step} ({'进行中' if latest_step < 50000 else '已完成'})")
        print(f"    last5={[f'{v:.4f}' for _,v in last]}")
