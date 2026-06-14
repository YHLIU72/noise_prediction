"""
从 TensorBoard 事件文件中提取所有运行的最佳验证集指标
"""
import os
import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = r"F:\lyh\paddlespeech\papernoise\physic\runs"

# 版本映射：run_name -> version label
VERSION_MAP = {
    # V1 - 初始多分支物理约束模型 (05-06)
    "pi_mbcn_hvac_20260506_150414_epochs200_bs8_lr0.001_dir_csvdata333": "V1",
    "pi_mbcn_hvac_20260504_181809_epochs200_bs8_lr0.001_dir_csvdata333": "V1 (备选/早期)",
    "pi_mbcn_hvac_20260504_184559_epochs200_bs8_lr0.001_dir_csvdata333": "V1 (备选/早期)",

    # V1.1 - 编码器扩展+物理退火 (05-06)
    "pi_mbcn_hvac_20260506_183039_epochs200_bs8_lr0.001_dir_csvdata333": "V1.1",

    # V1.1.1 - 移除OASPL/倍频程头 (05-06)
    "pi_mbcn_hvac_20260506_171933_epochs200_bs8_lr0.001_dir_csvdata333": "V1.1.1",

    # V1.1.1.1 - 残差转置卷积+纯数据驱动 (05-09)
    "pi_mbcn_hvac_20260509_200108_epochs300_bs4_lr0.0005_dir_csvdata333": "V1.1.1.1",

    # V1.1.1.1.1 - HPS-MTL 架构 (05-11)
    "pi_mbcn_hvac_MTL_20260511_123415_epochs300_bs16_lr0.0005": "V1.1.1.1.1 (300ep)",
    "pi_mbcn_hvac_MTL_20260511_211708_epochs1500_bs16_lr0.0005": "V1.1.1.1.1 (1500ep)",

    # V514 - SE注意力+自适应权重 (05-14)
    "pi_mbcn_hvac_MTL_20260514_160650_epochs7500_bs16_lr0.0005": "V514",

    # V0521 - 性能优化版 (05-21)
    "pi_mbcn_hvac_MTL_20260521_220418_epochs7500_bs16_lr0.0005_opt0521": "V0521 (run1)",
    "pi_mbcn_hvac_MTL_20260521_220730_epochs7500_bs16_lr0.0005_opt0521": "V0521 (run2)",

    # V0529 - 小样本高鲁棒版 (05-29)
    "pi_mbcn_WinFast_20260529_162937_epochs50000_bs8_wd0.005": "V0529 (run1)",
    "pi_mbcn_WinFast_20260529_224446_epochs50000_bs8_wd0.005": "V0529 (run2)",

    # V0601 - 极端瘦身版 (06-01)
    "pi_mbcn_WinFast_20260601_183111_epochs50000_bs8_wd0.005": "V0601 (run1, hidden=512)",
    "pi_mbcn_WinFast_20260601_194754_epochs50000_bs8_wd0.005": "V0601 (run2, hidden=512)",
    "pi_mbcn_WinFast_20260601_205659_epochs50000_bs8_wd0.005": "V0601 (run3, hidden=256)",

    # V0602 - 权重衰减调优 (06-02)
    "pi_mbcn_WinFast_20260602_164724_epochs50000_bs8_wd0.01": "V0602 (wd=0.01, run1)",
    "pi_mbcn_WinFast_20260602_173246_epochs50000_bs8_wd0.01": "V0602 (wd=0.01, run2)",
    "pi_mbcn_WinFast_20260602_185145_epochs50000_bs8_wd0.0005": "V0602 (wd=0.0005)",
    "pi_mbcn_WinFast_20260602_210337_epochs50000_bs8_wd0.005": "V0602 (wd=0.005)",

    # Other/early runs
    "pi_mbcn_WinFast_20260608_152940_epochs50000_bs8_wd0.005": "V0608 (中间实验)",

    # V0609 - 20~5000Hz (06-09)
    "pi_mbcn_20to5000_20260609_172149_epochs50000_bs8_wd0.005": "V0609 (wd=0.005)",
    "pi_mbcn_20to5000_20260609_220932_epochs50000_bs8_wd0.001": "V0609 (wd=0.001)",

    # Early/Old runs
    "pi_mbcn_hvac_20260428_161913_epochs100_bs16_lr0.001_dir_csvdata333": "Early (04-28 run1)",
    "pi_mbcn_hvac_20260428_163839_epochs100_bs16_lr0.001_dir_csvdata333": "Early (04-28 run2)",
    "pi_mbcn_hvac_20260428_213447_epochs200_bs16_lr0.001_dir_csvdata333": "Early (04-28 run3)",
    "pi_mbcn_hvac_epochs100_bs16_lr0.001_dir_csvdata333": "Early (04-28 run4)",
}


def extract_run(run_path, run_name):
    """提取单个运行的 TensorBoard 数据"""
    event_files = [f for f in os.listdir(run_path) if f.startswith("events.out.tfevents")]
    if not event_files:
        return None

    event_file = os.path.join(run_path, event_files[0])
    try:
        ea = EventAccumulator(event_file, size_guidance={})
        ea.Reload()
    except Exception as e:
        print(f"  [ERROR] 无法加载 {event_file}: {e}")
        return None

    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"  [WARN] 无标量数据")
        return None

    # 提取所有 val_ 开头的指标
    val_tags = [t for t in tags if 'val_' in t.lower()]
    train_tags = [t for t in tags if 'train_' in t.lower() and t not in val_tags]
    other_tags = [t for t in tags if t not in val_tags and t not in train_tags]

    result = {
        'run_name': run_name,
        'version': VERSION_MAP.get(run_name, 'Unknown'),
        'num_events': len(tags),
        'val_metrics': {},
        'train_metrics': {},
        'other_metrics': {},
    }

    # 验证集指标：找每个指标的最小值（对于loss）和/或记录所有epoch的best
    for tag in val_tags:
        try:
            events = ea.Scalars(tag)
            if not events:
                continue
            values = [(e.step, e.value) for e in events]
            # 找最小值及其epoch（对于loss类指标）
            min_val = min(values, key=lambda x: x[1])
            # 也记录最后一个epoch的值
            last_val = values[-1]
            result['val_metrics'][tag] = {
                'best': min_val[1],
                'best_epoch': min_val[0],
                'last': last_val[1],
                'last_epoch': last_val[0],
                'all_values': values  # 保留所有值用于后续分析
            }
        except Exception as e:
            print(f"  [WARN] 无法读取 {tag}: {e}")

    # 训练集指标
    for tag in train_tags:
        try:
            events = ea.Scalars(tag)
            if not events:
                continue
            values = [(e.step, e.value) for e in events]
            min_val = min(values, key=lambda x: x[1])
            last_val = values[-1]
            result['train_metrics'][tag] = {
                'best': min_val[1],
                'best_epoch': min_val[0],
                'last': last_val[1],
                'last_epoch': last_val[0],
            }
        except Exception as e:
            pass

    # 其他指标（如学习率、动态权重等）
    for tag in other_tags:
        try:
            events = ea.Scalars(tag)
            if not events:
                continue
            values = [(e.step, e.value) for e in events]
            last_val = values[-1]
            result['other_metrics'][tag] = {
                'last': last_val[1],
                'last_epoch': last_val[0],
            }
        except Exception as e:
            pass

    return result


def find_overall_best(results):
    """
    对于每个版本，综合考虑 val_total 或 val_spectrum 或 val_mse_spec 
    找出最佳的那个 epoch 对应的所有指标值
    """
    enhanced = {}
    for run_name, data in results.items():
        if data is None:
            continue
        version = data['version']
        val_metrics = data['val_metrics']

        # 决定用哪个指标来找最优epoch
        # 优先级: val_total > val_spectrum > val_mse_spec
        best_tag = None
        for candidate in ['Loss/val_total', 'Loss/val_spectrum', 'Loss/val_mse_spec', 'Loss/val_mse_db']:
            if candidate in val_metrics:
                best_tag = candidate
                break

        if best_tag is None and val_metrics:
            # 随便选第一个val指标
            best_tag = list(val_metrics.keys())[0]

        if best_tag is None:
            continue

        best_epoch = val_metrics[best_tag]['best_epoch']
        best_val = val_metrics[best_tag]['best']

        # 提取该epoch附近的所有指标值（从all_values中）
        summary = {
            'version': version,
            'best_tag': best_tag,
            'best_val': best_val,
            'best_epoch': best_epoch,
            'metrics_at_best': {},
            'all_val_metrics': {},
        }

        for tag, info in val_metrics.items():
            # 找最接近 best_epoch 的值
            all_vals = info.get('all_values', [])
            closest = None
            for step, val in all_vals:
                if step == best_epoch:
                    closest = val
                    break
            if closest is None and all_vals:
                # 取该tag自己的best
                closest = info['best']

            summary['metrics_at_best'][tag] = closest
            summary['all_val_metrics'][tag] = {
                'best': info['best'],
                'best_epoch': info['best_epoch'],
                'last': info['last'],
            }

        enhanced[run_name] = summary

    return enhanced


def main():
    print("=" * 80)
    print("  提取 TensorBoard 验证集指标")
    print("=" * 80)

    all_results = {}
    run_dirs = sorted(os.listdir(RUNS_DIR))

    for run_dir in run_dirs:
        run_path = os.path.join(RUNS_DIR, run_dir)
        if not os.path.isdir(run_path):
            continue
        # 跳过 models 子目录（万一有）
        if run_dir == 'models':
            continue

        version_label = VERSION_MAP.get(run_dir, 'Unknown')
        print(f"\n[{version_label}] {run_dir}")
        result = extract_run(run_path, run_dir)
        if result:
            all_results[run_dir] = result
            # 打印简略信息
            val_tags = list(result['val_metrics'].keys())
            print(f"  验证指标 ({len(val_tags)}): {val_tags}")
            for tag, info in result['val_metrics'].items():
                print(f"    {tag}: best={info['best']:.4f} @epoch={info['best_epoch']}, last={info['last']:.4f}")

    # 增强：找每个版本的最佳epoch
    enhanced = find_overall_best(all_results)

    # 输出汇总到文件
    output_path = r"F:\lyh\paddlespeech\papernoise\physic\tb_extracted_results.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  TensorBoard 验证集指标提取结果\n")
        f.write("=" * 80 + "\n\n")

        # 按版本排序
        sorted_runs = sorted(all_results.items(), key=lambda x: x[0])

        for run_name, data in sorted_runs:
            if data is None:
                continue
            version = data['version']
            f.write(f"\n{'─' * 70}\n")
            f.write(f"  [{version}]  {run_name}\n")
            f.write(f"{'─' * 70}\n")

            # 训练指标简要
            if data['train_metrics']:
                f.write(f"  训练集最终指标:\n")
                for tag, info in data['train_metrics'].items():
                    f.write(f"    {tag}: {info['last']:.4f} @epoch={info['last_epoch']}\n")

            # 验证集指标
            if data['val_metrics']:
                f.write(f"  验证集指标:\n")
                for tag, info in data['val_metrics'].items():
                    f.write(f"    {tag}: best={info['best']:.4f} @epoch={info['best_epoch']}, "
                            f"last={info['last']:.4f} @epoch={info['last_epoch']}\n")

            # 其他指标
            if data['other_metrics']:
                f.write(f"  其他指标:\n")
                for tag, info in data['other_metrics'].items():
                    f.write(f"    {tag}: last={info['last']:.4f} @epoch={info['last_epoch']}\n")

        # 最佳epoch汇总
        f.write(f"\n\n{'=' * 80}\n")
        f.write(f"  各版本最佳Epoch汇总\n")
        f.write(f"{'=' * 80}\n")

        for run_name, enh in sorted(enhanced.items(), key=lambda x: x[0]):
            f.write(f"\n[{enh['version']}] best_epoch={enh['best_epoch']} "
                    f"(via {enh['best_tag']}={enh['best_val']:.4f})\n")
            for tag, val in enh['metrics_at_best'].items():
                f.write(f"  {tag}: {val:.4f}\n")

    print(f"\n\n结果已写入: {output_path}")
    return all_results, enhanced


if __name__ == '__main__':
    all_results, enhanced = main()
