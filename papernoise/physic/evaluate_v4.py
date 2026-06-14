"""
PIMBCN V4 模型全面评估程序
============================
基于 PIMBCN_train0612_v4.py 训练脚本，对训练好的模型进行全面评估。

评估维度:
  1. 全局指标: MSE(dB), MAE, RMSE, Cosine相似度, Pearson r, R²
  2. 物理损失维度: 多分辨率MSE(1×/2×/4×), Sobolev梯度损失, 线性峰值损失
  3. OASPL 预测精度
  4. 逐频率误差统计 (均值误差, 标准差, RMSE)
  5. 按工况( Mode × Type )分组评估
  6. 峰值频率/幅度误差
  7. 最佳/最差样本可视化
  8. 误差分布直方图
  9. 预测 vs 真实频谱叠加图
  10. 训练集/验证集对比

用法:
  python evaluate_v4.py
  python evaluate_v4.py --model runs/pi_mbcn_v4_xxx/models/best_model.pth
  python evaluate_v4.py --model best_model.pth --csv_dir csvdata333
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ---------- 导入 V4 模型与数据集 ----------
from PIMBCN_net0612_v4 import PI_MBCN, PhysicsLossWrapper
from PIMBCN_data_0612_v4 import PIMBCNDataset

# 全局配置
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
                      'axes.grid': True, 'grid.alpha': 0.3})
FREQ_BINS = 1246                    # 20~5000Hz 频率点数
FREQ_AXIS = np.linspace(20, 5000, FREQ_BINS)

# ============================================================================
# 工具函数
# ============================================================================

def compute_oaspl(spectrum_db, freq_axis=None):
    """从dB频谱计算OASPL (Overall A-weighted SPL 近似)"""
    if freq_axis is None:
        freq_axis = FREQ_AXIS
    # 使用线性dB求和: OASPL = 10*log10( sum( 10^(Lp/10) ) )
    # 这里简化为直接对线性值求和（假设等频率间隔）
    ref = torch.max(spectrum_db, dim=-1, keepdim=True)[0]
    linear = torch.pow(10.0, (spectrum_db - ref) / 10.0)
    oaspl = 10.0 * torch.log10(torch.sum(linear, dim=-1) + 1e-10) + ref.squeeze(-1)
    return oaspl


# ============================================================================
# 核心评估类
# ============================================================================

class ModelEvaluator:
    def __init__(self, model_path, csv_dir, save_dir, device=None):
        self.model_path = model_path
        self.csv_dir = csv_dir
        self.save_dir = save_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

        print(f"评估设备: {self.device}")
        self._load_data()
        self._load_model()

        # 获取 mode/type 映射
        self.mode_names = {i: name for i, name in
                           enumerate(self.train_dataset.mode_encoder.classes_)}
        self.type_names = {i: name for i, name in
                           enumerate(self.train_dataset.type_encoder.classes_)}

    # ----- 数据加载 -----
    def _load_data(self):
        print("\n" + "=" * 60)
        print("加载数据集...")
        self.train_dataset = PIMBCNDataset(
            directory_path=self.csv_dir,
            input_cols=[4, 5, 6], oaspl_col=11, octave_col=12, spectrum_col=13,
            type_col=3, mode_col=2, val_split=0.2,
            is_validation=False, augment=False)
        self.val_dataset = self.train_dataset.get_validation_dataset()

        print(f"  训练集: {len(self.train_dataset)} 条")
        print(f"  验证集: {len(self.val_dataset)} 条")

    # ----- 模型加载 -----
    def _load_model(self):
        print("\n加载模型...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请使用 --model 参数指定正确的路径, 例如:\n"
                f"  python evaluate_v4.py --model runs/pi_mbcn_v4_xxx/models/best_model.pth")
        
        self.model = PI_MBCN(num_modes=4, num_types=13, freq_bins=FREQ_BINS).to(self.device)
        self.loss_wrapper = PhysicsLossWrapper().to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.loss_wrapper.eval()

        epoch = checkpoint.get('epoch', '?')
        best_val = checkpoint.get('best_val_loss', '?')
        print(f"  模型加载成功! 训练轮次: {epoch}, 最佳验证损失: {best_val}")

    # ----- 批量推理 -----
    @torch.no_grad()
    def _infer_dataset(self, dataset, desc="Infer"):
        """对完整数据集做推理，返回 pred, target, modes, types, oaspl, octave"""
        all_pred = []
        all_target = []
        all_modes = []
        all_types = []
        all_oaspl = []
        all_octave = []

        for i in range(len(dataset)):
            inputs, types, modes, oaspl, octave, target = dataset[i]
            inp = inputs.unsqueeze(0).to(self.device)
            mt = torch.tensor([modes], dtype=torch.int64, device=self.device)
            tt = torch.tensor([types], dtype=torch.int64, device=self.device)

            pred = self.model(inp, mt, tt)
            all_pred.append(pred.cpu())
            # target 来自 torch.from_numpy，shape=[1246]
            all_target.append(target.unsqueeze(0) if target.dim() == 1 else target)
            all_modes.append(modes)
            all_types.append(types)
            all_oaspl.append(oaspl.item() if isinstance(oaspl, torch.Tensor) else float(oaspl))
            all_octave.append(octave.numpy() if isinstance(octave, torch.Tensor) else octave)

        pred_tensor = torch.cat(all_pred, dim=0)                     # [N, 1246]
        target_tensor = torch.cat(all_target, dim=0)                 # [N, 1246]
        return (pred_tensor, target_tensor,
                np.array(all_modes), np.array(all_types),
                np.array(all_oaspl), np.array(all_octave))

    # ----- 指标计算 -----
    def _compute_all_metrics(self, pred, target, modes, types, name):
        """计算完整指标集"""
        results = {'name': name}
        N = pred.shape[0]

        # --- 基础指标 ---
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        rmse = np.sqrt(mse)

        # Cosine 相似度
        cos_sim = F.cosine_similarity(pred, target, dim=1).mean().item()

        # Pearson & R² (展平)
        p_flat = pred.detach().cpu().numpy().flatten()
        t_flat = target.detach().cpu().numpy().flatten()
        pearson_r, pearson_p = pearsonr(p_flat, t_flat)
        r2 = r2_score(t_flat, p_flat)

        results['MSE_dB'] = mse
        results['MAE_dB'] = mae
        results['RMSE_dB'] = rmse
        results['Cosine_Sim'] = cos_sim
        results['Pearson_r'] = pearson_r
        results['R2'] = r2

        # --- 多分辨率 MSE ---
        ms_mse_1x = F.mse_loss(pred, target).item()
        p2 = F.avg_pool1d(pred.unsqueeze(1), 2, 2).squeeze(1)
        t2 = F.avg_pool1d(target.unsqueeze(1), 2, 2).squeeze(1)
        ms_mse_2x = F.mse_loss(p2, t2).item()
        p4 = F.avg_pool1d(pred.unsqueeze(1), 4, 4).squeeze(1)
        t4 = F.avg_pool1d(target.unsqueeze(1), 4, 4).squeeze(1)
        ms_mse_4x = F.mse_loss(p4, t4).item()
        results['MS_MSE_1x'] = ms_mse_1x
        results['MS_MSE_2x'] = ms_mse_2x
        results['MS_MSE_4x'] = ms_mse_4x

        # --- Sobolev 梯度损失 ---
        grad_pred = pred[:, 1:] - pred[:, :-1]
        grad_target = target[:, 1:] - target[:, :-1]
        grad_loss = F.mse_loss(grad_pred, grad_target).item()
        results['Sobolev_Grad'] = grad_loss

        # --- 线性峰值损失 ---
        ref = torch.max(target, dim=1, keepdim=True)[0]
        pred_lin = torch.pow(10.0, (pred - ref) / 20.0)
        target_lin = torch.pow(10.0, (target - ref) / 20.0)
        linear_peak_loss = F.mse_loss(pred_lin, target_lin).item()
        results['LinearPeak'] = linear_peak_loss

        # --- OASPL ---
        pred_oaspl = compute_oaspl(pred)
        target_oaspl = compute_oaspl(target)
        oaspl_mae = F.l1_loss(pred_oaspl, target_oaspl).item()
        oaspl_rmse = torch.sqrt(F.mse_loss(pred_oaspl, target_oaspl)).item()
        results['OASPL_MAE'] = oaspl_mae
        results['OASPL_RMSE'] = oaspl_rmse

        # --- 逐频率误差 ---
        per_freq_error = (pred - target).detach().cpu().numpy()          # [N, F]
        freq_mean_error = per_freq_error.mean(axis=0)           # [F]
        freq_std_error = per_freq_error.std(axis=0)            # [F]
        freq_rmse = np.sqrt((per_freq_error ** 2).mean(axis=0)) # [F]
        results['freq_mean_error'] = freq_mean_error
        results['freq_std_error'] = freq_std_error
        results['freq_rmse'] = freq_rmse

        # --- 峰值频率 & 幅度误差 ---
        pred_peak_idx = torch.argmax(pred, dim=1).detach().cpu().numpy()
        target_peak_idx = torch.argmax(target, dim=1).detach().cpu().numpy()
        peak_freq_error_hz = np.mean(np.abs(
            FREQ_AXIS[pred_peak_idx] - FREQ_AXIS[target_peak_idx]))
        pred_peak_val = torch.max(pred, dim=1)[0].detach().cpu().numpy()
        target_peak_val = torch.max(target, dim=1)[0].detach().cpu().numpy()
        peak_val_mae = np.mean(np.abs(pred_peak_val - target_peak_val))
        results['PeakFreq_Error_Hz'] = peak_freq_error_hz
        results['PeakVal_MAE_dB'] = peak_val_mae

        # --- 每样本 MSE (用于最佳/最差) ---
        per_sample_mse = F.mse_loss(pred, target, reduction='none').mean(dim=1).detach().cpu().numpy()
        results['per_sample_mse'] = per_sample_mse

        return results

    # ----- 按工况分组评估 -----
    def _eval_by_condition(self, pred, target, modes, types):
        """按 (Mode, Type) 组合分组评估"""
        cond_results = {}
        for md in np.unique(modes):
            for tp in np.unique(types):
                mask = (modes == md) & (types == tp)
                n = mask.sum()
                if n < 1:
                    continue
                p = pred[mask]
                t = target[mask]
                mse_val = F.mse_loss(p, t).item()
                cos_val = F.cosine_similarity(p, t, dim=1).mean().item()
                cond_results[(int(md), int(tp))] = {
                    'count': int(n),
                    'MSE_dB': round(mse_val, 4),
                    'Cosine_Sim': round(cos_val, 4),
                }
        return cond_results

    # ----- 主评估流程 -----
    def run_full_evaluation(self):
        print("\n" + "=" * 60)
        print("开始全面评估...")

        # ---- 1. 推理 ----
        print("\n[1/7] 推理训练集...")
        pred_train, target_train, modes_train, types_train, oaspl_train, octave_train = \
            self._infer_dataset(self.train_dataset, "Train")
        print(f"  训练集预测完成: {pred_train.shape}")

        print("\n[2/7] 推理验证集...")
        # 验证集: 直接用 val_dataset 逐条推理
        pred_val_list, target_val_list = [], []
        modes_val_list, types_val_list = [], []
        oaspl_val_list, octave_val_list = [], []

        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                inputs, types, modes, oaspl, octave, target = self.val_dataset[i]
                inp = inputs.unsqueeze(0).to(self.device)
                mt = torch.tensor([modes], dtype=torch.int64, device=self.device)
                tt = torch.tensor([types], dtype=torch.int64, device=self.device)
                pred = self.model(inp, mt, tt)
                pred_val_list.append(pred.cpu())                          # [1, 1246]
                target_val_list.append(target.unsqueeze(0) if target.dim() == 1 else target)  # → [1, 1246]
                modes_val_list.append(modes)
                types_val_list.append(types)
                oaspl_val_list.append(oaspl.item() if isinstance(oaspl, torch.Tensor) else float(oaspl))
                octave_val_list.append(octave.numpy() if isinstance(octave, torch.Tensor) else octave)

        pred_val = torch.cat(pred_val_list, dim=0)                  # [N_val, 1246]
        target_val = torch.cat(target_val_list, dim=0)              # [N_val, 1246]
        modes_val = np.array(modes_val_list)
        types_val = np.array(types_val_list)
        oaspl_val = np.array(oaspl_val_list)
        octave_val = np.array(octave_val_list)
        print(f"  验证集预测完成: {pred_val.shape}")

        # ---- 3. 计算指标 ----
        print("\n[3/7] 计算训练集指标...")
        self.train_metrics = self._compute_all_metrics(
            pred_train, target_train, modes_train, types_train, "训练集")

        print("[4/7] 计算验证集指标...")
        self.val_metrics = self._compute_all_metrics(
            pred_val, target_val, modes_val, types_val, "验证集")

        # ---- 4. 按工况评估 ----
        print("\n[5/7] 按工况分组评估...")
        self.train_cond = self._eval_by_condition(pred_train, target_train, modes_train, types_train)
        self.val_cond = self._eval_by_condition(pred_val, target_val, modes_val, types_val)

        # ---- 5. 打印报告 ----
        print("\n[6/7] 生成文本报告...")
        self._print_report()

        # ---- 6. 绘图 ----
        print("\n[7/7] 生成可视化图表...")
        self._generate_plots(pred_train, target_train, modes_train, types_train, "train")
        self._generate_plots(pred_val, target_val, modes_val, types_val, "val")
        self._plot_train_vs_val(pred_train, target_train, pred_val, target_val)
        self._plot_condition_heatmap()
        self._plot_best_worst(pred_val, target_val, modes_val, types_val)

        # ---- 7. 保存详细结果 ----
        self._save_detailed_csv()

        print("\n" + "=" * 60)
        print(f"评估完成! 所有结果已保存至: {self.save_dir}")
        print("=" * 60)

        return self.train_metrics, self.val_metrics

    # ----- 报告打印 -----
    def _print_report(self):
        """格式化打印评估报告"""
        report_path = os.path.join(self.save_dir, 'evaluation_report.txt')
        lines = []
        def p(s=""):
            lines.append(s)
            print(s)

        p("=" * 75)
        p("  PIMBCN V4 模型全面评估报告")
        p("=" * 75)
        p(f"  模型路径: {self.model_path}")
        p(f"  数据目录: {self.csv_dir}")
        p(f"  训练样本: {len(self.train_dataset)} 条")
        p(f"  验证样本: {len(self.val_dataset)} 条")
        p()

        for metrics, title in [(self.train_metrics, "训练集"),
                                (self.val_metrics, "验证集")]:
            p(f"  {'─' * 60}")
            p(f"  【{title}】")
            p(f"  {'─' * 60}")
            p(f"  基础指标:")
            p(f"    MSE (dB):        {metrics['MSE_dB']:.4f}")
            p(f"    MAE (dB):        {metrics['MAE_dB']:.4f}")
            p(f"    RMSE (dB):       {metrics['RMSE_dB']:.4f}")
            p(f"    Cosine 相似度:   {metrics['Cosine_Sim']:.4f}")
            p(f"    Pearson r:       {metrics['Pearson_r']:.4f}")
            p(f"    R²:              {metrics['R2']:.4f}")
            p()
            p(f"  多分辨率 MSE:")
            p(f"    原分辨率 (1×):   {metrics['MS_MSE_1x']:.4f}")
            p(f"    2× 下采样:       {metrics['MS_MSE_2x']:.4f}")
            p(f"    4× 下采样:       {metrics['MS_MSE_4x']:.4f}")
            p()
            p(f"  物理损失:")
            p(f"    Sobolev 梯度:    {metrics['Sobolev_Grad']:.6f}")
            p(f"    线性峰值:        {metrics['LinearPeak']:.6f}")
            p()
            p(f"  OASPL 预测:")
            p(f"    MAE:             {metrics['OASPL_MAE']:.4f} dB")
            p(f"    RMSE:            {metrics['OASPL_RMSE']:.4f} dB")
            p()
            p(f"  峰值误差:")
            p(f"    峰值频率 MAE:    {metrics['PeakFreq_Error_Hz']:.2f} Hz")
            p(f"    峰值幅度 MAE:    {metrics['PeakVal_MAE_dB']:.2f} dB")
            p()

        # 泛化差距
        p(f"  {'─' * 60}")
        p(f"  【泛化差距 (Val - Train)】")
        p(f"  {'─' * 60}")
        for k in ['MSE_dB', 'MAE_dB', 'RMSE_dB', 'Cosine_Sim', 'R2',
                   'OASPL_MAE', 'OASPL_RMSE', 'PeakVal_MAE_dB']:
            gap = self.val_metrics[k] - self.train_metrics[k]
            direction = "↑ 恶化" if gap > 0 else "↓ 改善"
            p(f"    {k:<20s}: {gap:+.4f}  {direction}")
        p()

        # 按工况汇总
        p(f"  {'─' * 60}")
        p(f"  【验证集按工况 (Mode, Type) 分组 — MSE(dB) / Cosine】")
        p(f"  {'─' * 60}")
        p(f"  {'Mode':<8s} {'Type':<8s} {'N':>5s} {'MSE_dB':>10s} {'Cosine':>8s}")
        p(f"  {'-' * 45}")
        sorted_conds = sorted(self.val_cond.items(),
                              key=lambda x: x[1]['MSE_dB'])
        for (md, tp), v in sorted_conds:
            md_name = self.mode_names.get(md, f"Mode{md}")
            tp_name = self.type_names.get(tp, f"Type{tp}")
            p(f"  {md_name:<8s} {tp_name:<8s} {v['count']:>5d} "
              f"{v['MSE_dB']:>10.4f} {v['Cosine_Sim']:>8.4f}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n报告已保存至: {report_path}")

    # ----- 可视化 -----
    def _generate_plots(self, pred, target, modes, types, tag):
        """为单个数据集生成图表"""
        plot_dir = os.path.join(self.save_dir, 'plots')
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        N = pred_np.shape[0]

        # ---- (a) 随机样本对比 ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idxs = np.random.choice(N, min(4, N), replace=False)
        for ax, idx in zip(axes.flat, idxs):
            md = self.mode_names.get(modes[idx], f"M{modes[idx]}")
            tp = self.type_names.get(types[idx], f"T{types[idx]}")
            ax.plot(FREQ_AXIS, target_np[idx], 'b-', alpha=0.7, label='Ground Truth', linewidth=1)
            ax.plot(FREQ_AXIS, pred_np[idx], 'r--', alpha=0.7, label='Prediction', linewidth=1)
            ax.set_title(f'{tag.upper()} | {md}-{tp} (idx={idx})')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('SPL (dB)')
            ax.legend(fontsize=8)
        fig.suptitle(f'{tag.upper()} — 预测 vs 真实频谱 (随机采样)', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{tag}_sample_comparison.png'))
        plt.close(fig)

        # ---- (b) 逐频率误差带 ----
        per_freq_err = pred_np - target_np
        mean_err = per_freq_err.mean(axis=0)
        std_err = per_freq_err.std(axis=0)
        rmse_freq = np.sqrt((per_freq_err ** 2).mean(axis=0))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.plot(FREQ_AXIS, mean_err, 'b-', linewidth=1, label='Mean Error')
        ax1.fill_between(FREQ_AXIS, mean_err - std_err, mean_err + std_err,
                          alpha=0.25, color='blue', label='±1 Std')
        ax1.fill_between(FREQ_AXIS, mean_err - 2*std_err, mean_err + 2*std_err,
                          alpha=0.1, color='blue', label='±2 Std')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Error (dB)')
        ax1.set_title(f'{tag.upper()} — 逐频率误差分布 (均值 ± 标准差)')
        ax1.legend(fontsize=8)

        ax2.plot(FREQ_AXIS, rmse_freq, 'g-', linewidth=1, label='RMSE per Frequency')
        ax2.fill_between(FREQ_AXIS, 0, rmse_freq, alpha=0.15, color='green')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('RMSE (dB)')
        ax2.set_title(f'{tag.upper()} — 逐频率 RMSE')
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{tag}_freq_error.png'))
        plt.close(fig)

        # ---- (c) 误差直方图 ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(per_freq_err.flatten(), bins=100, color='steelblue',
                edgecolor='white', alpha=0.8, density=True)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Prediction Error (dB)')
        ax.set_ylabel('Density')
        ax.set_title(f'{tag.upper()} — 误差分布直方图')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{tag}_error_hist.png'))
        plt.close(fig)

        # ---- (d) OASPL 散点图 ----
        pred_oaspl = compute_oaspl(pred).detach().cpu().numpy()
        target_oaspl = compute_oaspl(target).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(target_oaspl, pred_oaspl, alpha=0.5, s=10, c='steelblue', edgecolors='none')
        mn = min(target_oaspl.min(), pred_oaspl.min())
        mx = max(target_oaspl.max(), pred_oaspl.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='y=x')
        ax.set_xlabel('Target OASPL (dB)')
        ax.set_ylabel('Predicted OASPL (dB)')
        ax.set_title(f'{tag.upper()} — OASPL 预测散点图')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{tag}_oaspl_scatter.png'))
        plt.close(fig)

    # ----- 训练集 vs 验证集对比 -----
    def _plot_train_vs_val(self, pred_train, target_train, pred_val, target_val):
        """训练集和验证集频率误差对比"""
        plot_dir = os.path.join(self.save_dir, 'plots')

        err_train = (pred_train - target_train).detach().cpu().numpy()
        err_val = (pred_val - target_val).detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Mean error 对比
        ax1.plot(FREQ_AXIS, err_train.mean(axis=0), 'b-', linewidth=1, label='Train Mean Error')
        ax1.plot(FREQ_AXIS, err_val.mean(axis=0), 'r-', linewidth=1, label='Val Mean Error')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Mean Error (dB)')
        ax1.set_title('训练集 vs 验证集 — 逐频率平均误差对比')
        ax1.legend()

        # RMSE 对比
        rmse_train = np.sqrt((err_train ** 2).mean(axis=0))
        rmse_val = np.sqrt((err_val ** 2).mean(axis=0))
        ax2.plot(FREQ_AXIS, rmse_train, 'b-', linewidth=1, label='Train RMSE')
        ax2.plot(FREQ_AXIS, rmse_val, 'r-', linewidth=1, label='Val RMSE')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('RMSE (dB)')
        ax2.set_title('训练集 vs 验证集 — 逐频率 RMSE 对比')
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'train_vs_val_freq_error.png'))
        plt.close(fig)

    # ----- 工况热力图 -----
    def _plot_condition_heatmap(self):
        """绘制 Mode × Type 的 MSE 热力图"""
        plot_dir = os.path.join(self.save_dir, 'plots')

        modes_list = sorted(self.mode_names.keys())
        types_list = sorted(self.type_names.keys())
        n_modes, n_types = len(modes_list), len(types_list)

        mse_mat = np.full((n_modes, n_types), np.nan)
        cos_mat = np.full((n_modes, n_types), np.nan)

        for i, md in enumerate(modes_list):
            for j, tp in enumerate(types_list):
                key = (md, tp)
                if key in self.val_cond:
                    mse_mat[i, j] = self.val_cond[key]['MSE_dB']
                    cos_mat[i, j] = self.val_cond[key]['Cosine_Sim']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        im1 = ax1.imshow(mse_mat, aspect='auto', cmap='YlOrRd', origin='upper')
        ax1.set_xticks(range(n_types))
        ax1.set_xticklabels([self.type_names[t][:6] for t in types_list], rotation=45, ha='right', fontsize=8)
        ax1.set_yticks(range(n_modes))
        ax1.set_yticklabels([self.mode_names[m] for m in modes_list], fontsize=8)
        ax1.set_title('验证集 MSE(dB) 热力图')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(cos_mat, aspect='auto', cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
        ax2.set_xticks(range(n_types))
        ax2.set_xticklabels([self.type_names[t][:6] for t in types_list], rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(n_modes))
        ax2.set_yticklabels([self.mode_names[m] for m in modes_list], fontsize=8)
        ax2.set_title('验证集 Cosine 相似度 热力图')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        fig.suptitle('Mode × Type 工况评估', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'condition_heatmap.png'))
        plt.close(fig)

    # ----- 最佳/最差样本 -----
    def _plot_best_worst(self, pred, target, modes, types, top_k=5):
        """展示最佳和最差的预测样本"""
        plot_dir = os.path.join(self.save_dir, 'plots')
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        per_mse = ((pred_np - target_np) ** 2).mean(axis=1)
        best_idx = np.argsort(per_mse)[:top_k]
        worst_idx = np.argsort(per_mse)[-top_k:][::-1]

        fig, axes = plt.subplots(2, top_k, figsize=(top_k * 3.5, 8))
        for col, idx in enumerate(best_idx):
            ax = axes[0, col]
            ax.plot(FREQ_AXIS, target_np[idx], 'b-', alpha=0.7, linewidth=1, label='True')
            ax.plot(FREQ_AXIS, pred_np[idx], 'r--', alpha=0.7, linewidth=1, label='Pred')
            ax.set_title(f'Best #{col+1}\nMSE={per_mse[idx]:.2f}')
            ax.set_xlabel('Hz')
            if col == 0: ax.set_ylabel('SPL (dB)')
            ax.legend(fontsize=6)

        for col, idx in enumerate(worst_idx):
            ax = axes[1, col]
            ax.plot(FREQ_AXIS, target_np[idx], 'b-', alpha=0.7, linewidth=1, label='True')
            ax.plot(FREQ_AXIS, pred_np[idx], 'r--', alpha=0.7, linewidth=1, label='Pred')
            ax.set_title(f'Worst #{col+1}\nMSE={per_mse[idx]:.2f}')
            ax.set_xlabel('Hz')
            if col == 0: ax.set_ylabel('SPL (dB)')
            ax.legend(fontsize=6)

        fig.suptitle('验证集 — 最佳/最差预测样本', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, 'best_worst_samples.png'))
        plt.close(fig)

    # ----- 保存 CSV -----
    def _save_detailed_csv(self):
        """保存详细的逐频率误差统计到 CSV"""
        for tag, metrics in [('train', self.train_metrics), ('val', self.val_metrics)]:
            df = pd.DataFrame({
                'Frequency_Hz': FREQ_AXIS,
                'Mean_Error_dB': metrics['freq_mean_error'],
                'Std_Error_dB': metrics['freq_std_error'],
                'RMSE_dB': metrics['freq_rmse'],
            })
            csv_path = os.path.join(self.save_dir, f'freq_error_stats_{tag}.csv')
            df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"  逐频率统计已保存: {csv_path}")

        # 保存汇总指标
        summary_rows = []
        for tag, m in [('train', self.train_metrics), ('val', self.val_metrics)]:
            row = {'Dataset': tag}
            for k, v in m.items():
                if not isinstance(v, np.ndarray):
                    row[k] = round(v, 6) if isinstance(v, float) else v
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(self.save_dir, 'metrics_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"  汇总指标已保存: {summary_csv}")

        # 保存按工况结果
        cond_rows = []
        for (md, tp), v in sorted(self.val_cond.items()):
            cond_rows.append({
                'Mode': self.mode_names.get(md, f"Mode{md}"),
                'Type': self.type_names.get(tp, f"Type{tp}"),
                'Count': v['count'],
                'MSE_dB': v['MSE_dB'],
                'Cosine_Sim': v['Cosine_Sim'],
            })
        cond_df = pd.DataFrame(cond_rows)
        cond_csv = os.path.join(self.save_dir, 'condition_breakdown.csv')
        cond_df.to_csv(cond_csv, index=False)
        print(f"  工况分组结果已保存: {cond_csv}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PIMBCN V4 模型全面评估')
    parser.add_argument('--model', type=str, default='auto',
                        help='模型权重文件路径 (默认自动查找最新V4模型)')
    parser.add_argument('--csv_dir', type=str, default='F:\\lyh\\paddlespeech\\csvdata333',
                        help='CSV 数据目录')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='结果保存目录 (默认自动生成)')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: auto / cuda / cpu')
    args = parser.parse_args()

    # 自动查找 V4 模型
    if args.model == 'auto':
        runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
        candidates = []
        if os.path.isdir(runs_dir):
            for d in os.listdir(runs_dir):
                if 'v4' in d.lower():
                    pth = os.path.join(runs_dir, d, 'models', 'best_model.pth')
                    if os.path.exists(pth):
                        candidates.append((d, pth))
        if candidates:
            # 取最新（按目录名时间戳排序）
            candidates.sort(key=lambda x: x[0], reverse=True)
            args.model = candidates[0][1]
            print(f"自动选择 V4 模型: {candidates[0][0]}")
        else:
            # 回退到默认
            args.model = 'best_model.pth'
            print("警告: 未找到 V4 模型, 使用默认路径 best_model.pth")

    # 自动生成保存目录
    if args.save_dir is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"evaluation_results_{timestamp}"

    # 设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    evaluator = ModelEvaluator(
        model_path=args.model,
        csv_dir=args.csv_dir,
        save_dir=args.save_dir,
        device=device,
    )
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
