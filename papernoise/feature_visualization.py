"""
MBS-NN 特征空间可视化脚本 - MSSP 论文专用
用途：展示不同工作模式/空调型号之间的领域偏移（Domain Shift）
核心技术：UMAP 降维 + 极致无边界美学设计
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

# 导入数据集和模型 (请确保这里的导入路径与你的项目一致)
from noisedata import SPLdata
from soundmodel import SPLModel

# ==================== 配置参数 ====================
class Config:
    data_dir = "../csvdata333"          # 数据集路径
    model_path = "runs/spl/exp_nc800_epochs100_batchsize32_lr0.0001_wd1e-0520260201_203236/checkpoint/best_model_epoch_95.pth"                    # 预训练模型路径（可选）
    batch_size = 32
    random_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "feature_visualization"  # 输出目录
    
    # UMAP 关键超参数设置（针对领域偏移可视化优化）
    umap_n_neighbors = 25    # 调小此值可强制解耦局部流形，强化簇分离度
    umap_min_dist = 0.01      # 调小此值可压缩簇内距离，使簇更加紧凑
    umap_n_components = 2     # 2维可视化
    umap_metric = 'cosine'    # 使用余弦距离度量特征相似性

# ==================== 特征提取函数 ====================
def extract_features(model, dataloader, device):
    """
    使用 register_forward_hook 提取共享主干网络最后一层的特征
    """
    model.eval()
    features = []
    mode_labels = []
    type_labels = []
    
    feature_cache = []
    def hook_fn(module, input, output):
        feature_cache.append(output.detach().cpu().numpy())
    
    # 注册 hook 到 hidden2 层（共享主干网络的最后一层）
    hook_handle = model.hidden2.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, targets, type_ids, mode_ids in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
            features.append(feature_cache[-1])
            mode_labels.append(mode_ids.cpu().numpy())
            type_labels.append(type_ids.cpu().numpy())
    
    hook_handle.remove()
    
    # 拼接并强制拉平标签，这是解决之前颜色映射错乱的关键！
    features = np.concatenate(features, axis=0)
    mode_labels = np.concatenate(mode_labels, axis=0).flatten()
    type_labels = np.concatenate(type_labels, axis=0).flatten()
    
    print(f"特征提取完成: {features.shape[0]} 个样本, 特征维度: {features.shape[1]}")
    return features, mode_labels, type_labels

# ==================== UMAP 降维函数 ====================
def apply_umap(features):
    print(f"\n开始 UMAP 降维...")
    print(f"参数设置: n_neighbors={Config.umap_n_neighbors}, min_dist={Config.umap_min_dist}")
    
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
    # 经过提纯的学术极简配色方案，确保对比度
    colors = [
        '#000080',  # 纯正藏青 (Navy)
        '#20b2aa',  # 蓝绿 (Teal)
        '#808080',  # 高级灰 (Gray)
        '#4682b4',  # 钢蓝 (Steel Blue)
        '#008080',  # 深青 (Dark Teal)
        '#a9a9a9',  # 暗灰 (Dark Gray)
        '#191970',  # 午夜蓝 (Midnight Blue)
        '#48d1cc',  # 中绿宝石 (Medium Turquoise)
        '#696969',  # 昏灰 (Dim Gray)
        '#5f9ea0',  # 军校蓝 (Cadet Blue)
        '#00ced1',  # 暗宝石绿 (Dark Turquoise)
        '#778899',  # 浅石板灰 (Light Slate Gray)
        '#0000cd',  # 中蓝 (Medium Blue)
    ]
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    print(f"\n绘制 {n_classes} 个类别的特征分布...")
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            umap_results[mask, 0],
            umap_results[mask, 1],
            c=colors[i % len(colors)],
            s=20,                  # 适当增大点面积，增强存在感
            alpha=0.85,            # 提高不透明度，让色带更扎实
            edgecolors='white',
            linewidths=0.1,       # 极细的白边，只提供微观颗粒感，不撕裂整体形状
            marker='o',
            zorder=2
        )
    
    # 彻底关闭所有坐标轴元素
    plt.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 设置纯白背景
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 保存为高分辨率 PNG
    png_path = os.path.join(Config.output_dir, f'{save_prefix}_umap.png')
    plt.savefig(
        png_path,
        format='png',
        dpi=600,
        transparent=False,
        bbox_inches='tight',
        pad_inches=2.0  # 进一步加大留白，留出后期添加箭头和文字的空间
    )
    
    # 保存为 SVG 矢量图
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
    print("MBS-NN 特征空间可视化脚本 - 优化版")
    print("=" * 70)
    
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)
    
    dataset = SPLdata(
        directory_path=Config.data_dir,
        val_split=0,
        random_seed=Config.random_seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if Config.device == "cuda" else False
    )
    
    model = SPLModel(nc=800).to(Config.device)
    
    if Config.model_path and os.path.exists(Config.model_path):
        checkpoint = torch.load(Config.model_path, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    features, mode_labels, type_labels = extract_features(model, dataloader, Config.device)
    umap_results = apply_umap(features)
    
    plot_embedding(umap_results, mode_labels, save_prefix='mode_clusters')
    plot_embedding(umap_results, type_labels, save_prefix='type_clusters')

if __name__ == "__main__":
    main()