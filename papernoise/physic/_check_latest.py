import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = r"f:\lyh\paddlespeech\papernoise\physic\runs"
targets = [
    'pi_mbcn_v8_20260612_215015',
    'pi_mbcn_v9_20260612_230539',
    'pi_mbcn_v7_20260612_185911',
]

for t in targets:
    dp = os.path.join(base, t)
    if not os.path.isdir(dp):
        print(f"[SKIP] {t} not found")
        continue
    ef = glob.glob(os.path.join(dp, "events.out.*"))
    if not ef:
        print(f"[SKIP] {t} no events")
        continue
    print(f"\n{'='*60}")
    print(f"  {t}")
    print(f"{'='*60}")
    ea = EventAccumulator(ef[0], size_guidance={"scalars": 0})
    ea.Reload()
    val_tags = sorted([tag for tag in ea.Tags().get("scalars", []) if "val" in tag.lower()])
    if not val_tags:
        print("  (no val tags)")
        continue
    for tag in val_tags:
        events = ea.Scalars(tag)
        vals = [(e.step, e.value) for e in events]
        if not vals: continue
        min_step, min_val = min(vals, key=lambda x: x[1])
        latest_step = vals[-1][0]
        latest_val = vals[-1][1]
        print(f"  {tag}: min={min_val:.4f} @{min_step}  last={latest_val:.4f} @{latest_step}")
    
    # check model files
    model_dir = os.path.join(dp, "models")
    if os.path.isdir(model_dir):
        models = os.listdir(model_dir)
        print(f"  models/: {models}")

print("\nDone.")
