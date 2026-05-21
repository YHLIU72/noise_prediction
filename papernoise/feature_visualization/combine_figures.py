"""
MSSP 期刊双栏通栏宽图拼接脚本
用途：将两张 UMAP 降维图拼接为一张符合期刊规范的对比图
核心功能：精确尺寸控制 + 学术标签 + Domain Shift 矢量箭头
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ==================== 配置参数 ====================
class Config:
    # 图片路径
    left_img_path = r'E:\lyh\paddlespeech\papernoise\feature_visualization\type_clusters_umap.png'
    right_img_path = r'E:\lyh\paddlespeech\papernoise\feature_visualization_decoupled\type_clusters_decoupled_umap.png'
    
    # 输出配置
    output_dir = r'E:\lyh\paddlespeech\papernoise\feature_visualization'
    output_name = 'combined_domain_shift_framework'
    
    # 画布尺寸配置（MSSP 双栏通栏宽）
    total_width_inch = 7.48    # 190 mm = 7.48 英寸
    total_height_inch = 3.74   # 高度为宽度的一半（1x2布局）
    dpi = 600                  # 高分辨率输出
    
    # 字体配置（期刊规范）
    font_family = 'Times New Roman'
    fallback_font = 'Arial'     # 备用字体
    label_fontsize = 10         # 子图标签字号 (pt)
    arrow_label_fontsize = 8    # 箭头标签字号 (pt)
    
    # 箭头配置
    arrow_color = '#808080'     # 高级灰
    arrow_linewidth = 2.0       # 箭头线宽
    arrow_head_width = 0.02     # 箭头头部宽度（相对坐标）
    arrow_head_length = 0.03    # 箭头头部长度（相对坐标）

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("MSSP 双栏通栏宽图拼接脚本")
    print("用途：展示解耦前后的特征空间对比")
    print("=" * 70)
    
    # 1. 读取两张输入图片
    print("\n[步骤1] 读取输入图片...")
    try:
        left_img = Image.open(Config.left_img_path).convert('RGB')
        right_img = Image.open(Config.right_img_path).convert('RGB')
        print(f"左图尺寸: {left_img.size}")
        print(f"右图尺寸: {right_img.size}")
    except Exception as e:
        print(f"图片读取失败: {e}")
        return
    
    # 2. 创建画布（精确尺寸控制）
    print("\n[步骤2] 创建画布...")
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=(Config.total_width_inch, Config.total_height_inch),
        dpi=Config.dpi
    )
    
    # 3. 在子图中显示图片
    print("\n[步骤3] 在子图中显示图片...")
    ax1.imshow(left_img)
    ax2.imshow(right_img)
    
    # 4. 彻底关闭坐标轴（无界悬浮感）
    print("\n[步骤4] 关闭坐标轴...")
    ax1.axis('off')
    ax2.axis('off')
    
    # 5. 添加子图标签 (a) 和 (b)
    print("\n[步骤5] 添加子图标签...")
    # 定义标签样式
    label_style = {
        'fontfamily': Config.font_family,
        'fontsize': Config.label_fontsize,
        'fontweight': 'bold',
        'color': '#000000',  # 黑色标签
        'ha': 'left',         # 左对齐
        'va': 'top'           # 顶部对齐
    }
    
    # 左图标签 (a) - 使用相对坐标定位
    ax1.annotate(
        '(a)',
        xy=(0.03, 0.97),  # 相对坐标：左3%，上3%
        xycoords='axes fraction',
        **label_style
    )
    
    # 右图标签 (b)
    ax2.annotate(
        '(b)',
        xy=(0.03, 0.97),
        xycoords='axes fraction',
        **label_style
    )
    
    # 6. 在右图添加 Domain Shift 矢量箭头和标记
    print("\n[步骤6] 添加 Domain Shift 箭头和标记...")
    
    # 箭头配置（使用相对坐标）
    # 箭头从左下方指向右上方，展示领域偏移方向
    arrow_start = (0.35, 0.35)  # 起点（相对坐标）
    arrow_end = (0.65, 0.65)    # 终点（相对坐标）
    
    # 绘制单向粗箭头
    ax2.annotate(
        '',  # 空文本，只绘制箭头
        xy=arrow_end,
        xycoords='axes fraction',
        xytext=arrow_start,
        textcoords='axes fraction',
        arrowprops=dict(
            facecolor=Config.arrow_color,
            edgecolor=Config.arrow_color,
            linewidth=Config.arrow_linewidth,
            headwidth=Config.arrow_head_width * 100,  # 转换为绝对单位
            headlength=Config.arrow_head_length * 100,
            width=0.005,  # 箭身宽度
            shrink=0.05   # 箭头与起点/终点的距离
        )
    )
    
    # 添加 $\Delta_{shift}$ 标记（斜体）
    ax2.annotate(
        r'$\Delta_{shift}$',
        xy=((arrow_start[0] + arrow_end[0]) / 2, arrow_end[1] + 0.08),
        xycoords='axes fraction',
        fontfamily=Config.font_family,
        fontsize=Config.arrow_label_fontsize,
        fontstyle='italic',
        color=Config.arrow_color,
        ha='center',
        va='bottom'
    )
    
    # 7. 调整布局并保存
    print("\n[步骤7] 调整布局并保存...")
    
    # 确保子图之间没有空隙，保持紧凑对齐
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0.02  # 子图之间的微小间距
    )
    
    # 创建输出目录
    import os
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 保存为高分辨率 PNG
    png_path = os.path.join(Config.output_dir, f'{Config.output_name}.png')
    plt.savefig(
        png_path,
        dpi=Config.dpi,
        transparent=False,
        bbox_inches='tight',
        pad_inches=0.2
    )
    print(f"PNG 已保存: {png_path}")
    
    # 保存为 SVG 矢量图
    svg_path = os.path.join(Config.output_dir, f'{Config.output_name}.svg')
    plt.savefig(
        svg_path,
        format='svg',
        transparent=False,
        bbox_inches='tight',
        pad_inches=0.2
    )
    print(f"SVG 已保存: {svg_path}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("拼接完成！")
    print(f"输出文件: {png_path}")
    print(f"输出文件: {svg_path}")
    print("画布尺寸: {w} x {h} 英寸 ({dpi} dpi)".format(
        w=Config.total_width_inch,
        h=Config.total_height_inch,
        dpi=Config.dpi
    ))
    print("=" * 70)

if __name__ == "__main__":
    main()
