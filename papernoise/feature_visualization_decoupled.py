"""
MBS-NN 特征空间可视化脚本 - MSSP 论文专用
用途：展示型号子分支解耦后的特征空间领域偏移
核心技术：从输出层权重提取等效型号子分支 + UMAP 降维 + 极致无边界美学
"""

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import umap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入数据集和原始模型
from noisedata import SPLdata
from soundmodel import SPLModel

# ==================== 配置参数 ====================
class Config:
    data_dir = "../csvdata333"          # 数据集路径
    model_path = "runs/spl/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260201_203236/checkpoint/best_model_epoch_95.pth"  
    batch_size = 32
    random_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "feature_visualization_decoupled"  # 输出目录
    
    # UMAP 关键超参数设置（针对领域偏移可视化优化）
    umap_n_neighbors = 20     # 调小此值可强制解耦局部流形，强化簇分离度
    umap_min_dist = 0.01      # 调小此值可压缩簇内距离，使簇更加紧凑
    umap_n_components = 2     # 2维可视化
    umap_metric = 'cosine'    # 使用余弦距离度量特征相似性

# ==================== 特征提取函数（从输出层权重提取等效型号子分支） ====================
def extract_decoupled_features(model, dataloader, device):
    """
    提取经过型号子分支解耦后的特征
    
    由于原始模型没有显式的型号子分支，我们从输出层权重中提取等效的型号子分支变换：
    - 输出层权重: (mode_nc * type_nc * num_bins) x nc
    - 我们将其 reshape 为: (mode_nc, type_nc, num_bins, nc)
    - 对于每个型号 type_idx，提取对应的子分支权重: (mode_nc * num_bins) x nc
    - 将主干特征通过该子分支变换，得到"型号解耦后的特征"
    
    参数:
        model: 原始SPLModel模型
        dataloader: 数据加载器
        device: 设备
    
    返回:
        features: 解耦后的特征 (n_samples, mode_nc * num_bins)
        type_labels: 型号标签 (n_samples,)
        mode_labels: 模式标签 (n_samples,)
    """
    model.eval()
    features = []
    mode_labels = []
    type_labels = []
    
    # 获取输出层权重并 reshape 为等效的型号子分支形式
    out_weight = model.out.weight.data  # shape: (mode_nc * type_nc * num_bins, nc)
    out_bias = model.out.bias.data if model.out.bias is not None else None
    
    # 从 hidden1 层获取隐藏层维度 nc（因为原始模型没有保存 nc 属性）
    nc = model.hidden1.out_features
    
    # Reshape 权重为 (mode_nc, type_nc, num_bins, nc)
    weight_reshaped = out_weight.view(model.mode_nc, model.type_nc, model.num_bins, nc)
    
    # 预计算每个型号的子分支权重和偏置
    type_branch_weights = []
    type_branch_biases = []
    
    for type_idx in range(model.type_nc):
        # 提取该型号的子分支权重: (mode_nc * num_bins, nc)
        type_weight = weight_reshaped[:, type_idx, :, :].view(model.mode_nc * model.num_bins, nc)
        type_branch_weights.append(type_weight)
        
        # 提取该型号的子分支偏置
        if out_bias is not None:
            bias_reshaped = out_bias.view(model.mode_nc, model.type_nc, model.num_bins)
            type_bias = bias_reshaped[:, type_idx, :].view(model.mode_nc * model.num_bins)
            type_branch_biases.append(type_bias)
        else:
            type_branch_biases.append(torch.zeros(model.mode_nc * model.num_bins, device=device))
    
    # 使用 hook 提取主干特征（hidden2层之后）
    backbone_features_cache = []
    
    def hook_fn(module, input, output):
        backbone_features_cache.append(output.detach().cpu().numpy())
    
    # 在 hidden2 层注册 hook
    hook_handle = model.hidden2.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, targets, type_ids, mode_ids in dataloader:
            inputs = inputs.to(device)
            type_ids = type_ids.to(device)
            
            # 前向传播，触发 hook 捕获主干特征
            _ = model(inputs)
            
            # 获取主干特征
            backbone_feat = torch.tensor(backbone_features_cache[-1], device=device)
            
            # 通过 dropout3
            backbone_feat = model.dropout3(backbone_feat)
            
            # 应用型号子分支变换
            batch_size = inputs.size(0)
            decoupled_features = torch.zeros(batch_size, model.mode_nc * model.num_bins, device=device)
            
            for type_idx in range(model.type_nc):
                mask = (type_ids == type_idx)
                if mask.any():
                    # 应用该型号的子分支变换: y = x @ W^T + b
                    feat_subset = backbone_feat[mask]
                    weight = type_branch_weights[type_idx].to(device)
                    bias = type_branch_biases[type_idx].to(device)
                    decoupled_features[mask] = feat_subset @ weight.T + bias
            
            # 收集特征和标签
            features.append(decoupled_features.detach().cpu().numpy())
            mode_labels.append(mode_ids.cpu().numpy())
            type_labels.append(type_ids.cpu().numpy())
    
    # 移除 hook
    hook_handle.remove()
    
    # 拼接并强制拉平标签
    features = np.concatenate(features, axis=0)
    mode_labels = np.concatenate(mode_labels, axis=0).flatten()
    type_labels = np.concatenate(type_labels, axis=0).flatten()
    
    print(f"特征提取完成: {features.shape[0]} 个样本, 特征维度: {features.shape[1]}")
    print(f"型号标签范围: {np.min(type_labels)} ~ {np.max(type_labels)}")
    return features, mode_labels, type_labels

# ==================== UMAP 降维函数 ====================
def apply_umap(features):
    """
    使用 UMAP 进行高效降维
    
    UMAP 超参数设计原理（针对解耦特征的可视化优化）：
    - n_neighbors: 局部邻域大小，较小的值（10-20）会强化局部结构，使簇更加分离
    - min_dist: 嵌入空间中点的最小距离，较小的值（0.01-0.1）会让簇更紧凑
    - metric: 使用余弦距离度量特征空间中的相似性，适合高维数据
    
    参数:
        features: 高维特征矩阵 (n_samples, n_features)
    
    返回:
        umap_results: 降维后的 2D 特征 (n_samples, 2)
    """
    print(f"\n开始 UMAP 降维...")
    print(f"参数设置: n_neighbors={Config.umap_n_neighbors}, min_dist={Config.umap_min_dist}")
    print(f"设计原理: 较小的 n_neighbors 和 min_dist 强制解耦局部流形，强化簇群分离度")
    
    umap_reducer = umap.UMAP(
        n_neighbors=Config.umap_n_neighbors,
        min_dist=Config.umap_min_dist,
        n_components=Config.umap_n_components,
        metric=Config.umap_metric,
        random_state=Config.random_seed,
        verbose=True
    )
    
    umap_results = umap_reducer.fit_transform(features)
    print(f"UMAP 降维完成: {umap_results.shape}")
    return umap_results

# ==================== 极致无边界可视化函数 ====================
def plot_embedding(umap_results, labels, save_prefix='clusters'):
    """
    绘制极致学术风格的 2D 特征空间散点图
    
    视觉规范：
    1. 无边界美学：彻底移除坐标系、刻度、边框
    2. 学术极简配色：13种冷色系/中性色系，适用于13种空调型号
    3. 散点质感：透明度 + 白色描边 + 适当点大小
    4. 超大留白：pad_inches=2.0 确保宽广呼吸空间
    """
    # 学术极简配色方案（包含13种颜色，适用于13种空调型号）
    # 配色原则：冷色系为主，兼顾对比度和学术严谨性
    colors = [
        '#000080',  # 纯正藏青 (Navy) - 主色1
        '#20b2aa',  # 蓝绿 (Teal) - 主色2
        '#808080',  # 高级灰 (Gray) - 主色3
        '#4682b4',  # 钢蓝 (Steel Blue) - 辅助色1
        '#008080',  # 深青 (Dark Teal) - 辅助色2
        '#a9a9a9',  # 暗灰 (Dark Gray) - 辅助色3
        '#191970',  # 午夜蓝 (Midnight Blue) - 辅助色4
        '#48d1cc',  # 中绿宝石 (Medium Turquoise) - 辅助色5
        '#696969',  # 昏灰 (Dim Gray) - 辅助色6
        '#5f9ea0',  # 军校蓝 (Cadet Blue) - 辅助色7
        '#00ced1',  # 暗宝石绿 (Dark Turquoise) - 辅助色8
        '#778899',  # 浅石板灰 (Light Slate Gray) - 辅助色9
        '#1e3a5f',  # 深蓝灰 (Deep Blue Gray) - 辅助色10
    ]
    
    # 创建正方形画布（适合学术论文排版）
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600)
    
    # 获取唯一标签并统计类别数
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    print(f"\n绘制 {n_classes} 个类别的特征分布...")
    
    # 绘制每个类别的散点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            umap_results[mask, 0],
            umap_results[mask, 1],
            c=colors[i % len(colors)],  # 循环使用配色
            s=20,                       # 适当点大小，增强存在感
            alpha=0.85,                 # 较高不透明度，让色带更扎实
            edgecolors='white',         # 白色描边
            linewidths=0.1,            # 极细描边，提供微观颗粒感
            marker='o',                 # 圆形标记
            zorder=2                    # 确保散点在最上层
        )
    
    # ==================== 极致无边界美学处理 ====================
    # 1. 彻底关闭所有坐标轴元素
    plt.axis('off')
    
    # 2. 双重保险：确保所有边框和刻度都被隐藏
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # 3. 移除所有边框（spines）
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 4. 设置纯白背景（确保数据悬浮感）
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    # ==================== 超大留白输出 ====================
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 保存为高分辨率 PNG（600dpi）
    png_path = os.path.join(Config.output_dir, f'{save_prefix}_umap.png')
    plt.savefig(
        png_path,
        format='png',
        dpi=600,
        transparent=False,           # 纯白背景
        bbox_inches='tight',         # 紧凑边界
        pad_inches=2.0               # 超大留白，留出后期添加箭头和文字的空间
    )
    
    # 保存为 SVG 矢量图（无损缩放）
    svg_path = os.path.join(Config.output_dir, f'{save_prefix}_umap.svg')
    plt.savefig(
        svg_path,
        format='svg',
        transparent=False,
        bbox_inches='tight',
        pad_inches=2.0
    )
    
    plt.close()
    print(f"可视化已保存: {png_path}, {svg_path}")

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("MBS-NN 特征空间可视化脚本 - 型号子分支解耦版")
    print("用途：展示型号子分支解耦后的领域偏移")
    print("核心技术：从输出层权重提取等效型号子分支")
    print("=" * 70)
    
    # 1. 设置随机种子
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    
    # 2. 加载数据集
    print("\n[步骤1] 加载数据集...")
    dataset = SPLdata(
        directory_path=Config.data_dir,
        val_split=0.0,  # 使用全部数据
        random_seed=Config.random_seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if Config.device == "cuda" else False
    )
    
    # 3. 初始化原始模型
    print("\n[步骤2] 初始化原始SPLModel...")
    model = SPLModel(nc=800).to(Config.device)
    
    # 4. 加载预训练权重
    if Config.model_path and os.path.exists(Config.model_path):
        print(f"加载预训练模型: {Config.model_path}")
        checkpoint = torch.load(Config.model_path, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("预训练模型加载完成")
    
    # 5. 提取型号子分支解耦后的特征
    print("\n[步骤3] 提取型号子分支解耦后的特征...")
    features, mode_labels, type_labels = extract_decoupled_features(model, dataloader, Config.device)
    
    # 6. 应用 UMAP 降维
    print("\n[步骤4] 应用 UMAP 降维...")
    umap_results = apply_umap(features)
    
    # 7. 绘制型号分类可视化（核心：展示13种型号的解耦效果）
    print("\n[步骤5] 绘制型号分类特征散点图...")
    plot_embedding(umap_results, type_labels, save_prefix='type_clusters_decoupled')
    
    # 8. 绘制模式分类可视化（辅助：展示模式分布）
    print("\n[步骤6] 绘制模式分类特征散点图...")
    plot_embedding(umap_results, mode_labels, save_prefix='mode_clusters_decoupled')
    
    print("\n" + "=" * 70)
    print("可视化完成！结果已保存至: " + Config.output_dir)
    print("输出格式: PNG (600dpi) + SVG 矢量图")
    print("=" * 70)

if __name__ == "__main__":
    main()