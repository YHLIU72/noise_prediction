"""
V13 PIMBCN 模型总架构图 — 中文学术风格
比例: 3:4 (宽:高)
"""
import os, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Arc, ConnectionPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 9,
    'axes.unicode_minus': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
})

out_dir = r"f:\lyh\paddlespeech\papernoise\physic\plot_thesis"
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# Architecture Diagram (3:4 ratio)
# ================================================================
fig_width = 9   # inches
fig_height = 12  # 3:4 ratio
fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color scheme
C_INPUT = '#E3F2FD'      # light blue - input
C_ENCODER = '#BBDEFB'    # blue - encoder  
C_DECODER = '#C8E6C9'    # green - decoder
C_HEAD = '#FFF9C4'       # yellow - head
C_LOSS = '#FFCCBC'       # orange - loss
C_EMBED = '#E1BEE7'      # purple - embeddings
C_DATA = '#B2EBF2'       # cyan - data preprocessing
C_ARROW = '#546E7A'      # dark gray - arrows
C_BORDER = '#37474F'     # border
C_TEXT = '#212121'       # text
C_HIGHLIGHT = '#FF6F00'  # highlight

def draw_box(ax, x, y, w, h, color, text='', fontsize=8, bold=False, text_color=C_TEXT, 
             border_color=None, linewidth=1.5, alpha=0.85, sub_text=''):
    """Draw a rounded box with text"""
    if border_color is None:
        border_color = color
    # Use FancyBboxPatch for rounded corners
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", 
                          facecolor=color, edgecolor=border_color, 
                          linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    if text:
        if sub_text:
            ax.text(x + w/2, y + h*0.65, text, ha='center', va='center', 
                    fontsize=fontsize, fontweight='bold' if bold else 'normal', color=text_color)
            ax.text(x + w/2, y + h*0.3, sub_text, ha='center', va='center', 
                    fontsize=fontsize-1, color=text_color, style='italic')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                    fontsize=fontsize, fontweight='bold' if bold else 'normal', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style='->'):
    """Draw an arrow"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_v_arrow(ax, x, y1, y2, color=C_ARROW, lw=1.2):
    """Vertical downward arrow"""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def draw_brace_label(ax, x, y1, y2, text, color=C_TEXT, fontsize=7):
    """Draw a brace and label on the right side"""
    ax.plot([x, x], [y1, y2], 'k-', lw=0.8)
    ax.plot([x, x+1], [y1, y1], 'k-', lw=0.5)
    ax.plot([x, x+1], [y2, y2], 'k-', lw=0.5)
    ax.text(x+1.5, (y1+y2)/2, text, va='center', fontsize=fontsize, color=color, rotation=0)

# ===================== Title =====================
ax.text(50, 98, 'PIMBCN 模型总体架构 (V13)', ha='center', va='center', 
        fontsize=16, fontweight='bold', color=C_TEXT)
ax.text(50, 95.5, 'Physics-Informed Multi-Branch Convolutional Network — 对数频率重采样版', 
        ha='center', va='center', fontsize=9, color='#757575', style='italic')

# ===================== Section Labels =====================
sections = [
    (6, 90, '(a) 数据预处理', C_DATA),
    (6, 72, '(b) 输入与嵌入层', C_INPUT),
    (6, 58, '(c) 傅里叶特征编码器', C_ENCODER),
    (6, 42, '(d) 共享解码器主体', C_DECODER),
    (6, 22, '(e) 共享任务头', C_HEAD),
    (6, 7, '(f) 损失函数', C_LOSS),
]
for x, y, text, color in sections:
    ax.text(x, y, text, fontsize=8, fontweight='bold', color=color, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

# ===================== (a) Data Preprocessing =====================
draw_box(ax, 14, 82, 72, 8, C_DATA, '数据预处理: 对数频率重采样', fontsize=9, bold=True, 
         sub_text='np.interp(log_freqs, linear_freqs, spectrum) · 低频采样密度提升 18×')
# Input CSV
draw_box(ax, 14, 84, 18, 5, '#E0F7FA', 'CSV 原始数据\n(1079 条)', fontsize=7)
draw_box(ax, 36, 84, 22, 5, '#E0F7FA', '频谱解析 2501 点\n→ 截取 20-5000 Hz', fontsize=7)
draw_box(ax, 62, 84, 24, 5, '#B2DFDB', '对数插值重采样\n1246 频点 · logspace', fontsize=7, bold=True)
draw_arrow(ax, 32, 86.5, 36, 86.5)
draw_arrow(ax, 58, 86.5, 62, 86.5)

# ===================== (b) Input Layer =====================
# Input params
draw_box(ax, 14, 66, 20, 7, C_INPUT, '物理工况参数\n(3维): 转速/流速/压比', fontsize=7.5)
draw_box(ax, 40, 66, 18, 7, C_EMBED, 'Mode 嵌入\n(4类 → 16维)', fontsize=7.5)
draw_box(ax, 64, 66, 18, 7, C_EMBED, 'Type 嵌入\n(13类 → 16维)', fontsize=7.5)

# Label
ax.text(14, 74, '输入向量 [B, 3]', fontsize=6, color='#757575')
ax.text(40, 74, 'mode_idx [B]', fontsize=6, color='#757575')
ax.text(64, 74, 'type_idx [B]', fontsize=6, color='#757575')

# ===================== (c) Fourier Feature Encoder =====================
draw_box(ax, 14, 54, 72, 7, C_ENCODER, '傅里叶特征编码器 (Fourier Feature Embedding)', fontsize=9, bold=True)

# Encoder details
draw_box(ax, 16, 49, 14, 4, '#B3E5FC', 'Fourier映射\nB∈ℝ³ˣ³²', fontsize=6)
draw_box(ax, 32, 49, 10, 4, '#B3E5FC', '64→128\nLN+GELU\nDrop 0.5', fontsize=5.5)
draw_box(ax, 44, 49, 10, 4, '#B3E5FC', '128→256\nLN+GELU\nDrop 0.5', fontsize=5.5)
draw_box(ax, 56, 49, 10, 4, '#B3E5FC', '256→256\nLN+GELU\nDrop 0.3', fontsize=5.5)
draw_box(ax, 68, 49, 16, 4, '#81D4FA', '输出\n[B, 256]', fontsize=6, bold=True)

draw_arrow(ax, 30, 51.5, 32, 51.5, lw=0.8)
draw_arrow(ax, 42, 51.5, 44, 51.5, lw=0.8)
draw_arrow(ax, 54, 51.5, 56, 51.5, lw=0.8)
draw_arrow(ax, 66, 51.5, 68, 51.5, lw=0.8)

# Connection: input -> encoder
draw_v_arrow(ax, 24, 66, 61, C_ARROW, 1.0)
draw_v_arrow(ax, 49, 66, 61, C_ARROW, 1.0)
draw_v_arrow(ax, 73, 66, 61, C_ARROW, 1.0)

# Cat label
ax.text(50, 62.5, '拼接 (Concat) → [B, 256+16+16=288]', ha='center', fontsize=7, 
        color=C_HIGHLIGHT, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF9800', alpha=0.8))

# ===================== (d) Shared Decoder Body =====================
draw_box(ax, 14, 32, 72, 10, C_DECODER, '共享解码器主体 (Shared Decoder Body)', fontsize=9, bold=True)

# fc_expand
draw_box(ax, 16, 28, 13, 3, '#A5D6A7', 'FC Expand\n288→1024\nDrop 0.5', fontsize=5.5)
draw_box(ax, 16, 25.5, 13, 2, '#C8E6C9', 'Reshape\n[B, 64, 16]', fontsize=5.5)

# Deconv chain
deconv_configs = [
    ('TConv 64→64', '16→32'),
    ('TConv 64→32', '32→64'),
    ('TConv 32→32', '64→128'),
    ('TConv 32→16', '128→256'),
    ('TConv 16→8', '256→512'),
    ('TConv 8→8', '512→1024'),
]
for i, (label, seq) in enumerate(deconv_configs):
    x = 31 + i * 7
    draw_box(ax, x, 27.5, 6.2, 3.5, '#A5D6A7', f'{label}\n{seq}', fontsize=4.8)
    if i > 0:
        draw_arrow(ax, x-0.5, 29.25, x, 29.25, lw=0.6)

draw_arrow(ax, 29, 29.25, 31, 29.25, lw=0.8)

# SE Attention
draw_box(ax, 76, 27.5, 8, 3.5, '#81C784', 'SE 通道\n注意力\nreduction=2', fontsize=5.5, bold=True)
draw_arrow(ax, 73.5, 29.25, 76, 29.25, lw=0.8)

# Output label
draw_box(ax, 30, 33.5, 40, 3, '#388E3C', '共享声学特征图 [B, 8, 1024]', fontsize=7, bold=True, 
         text_color='white')
draw_arrow(ax, 50, 36.5, 50, 33.5, lw=1.0)  # up arrow from deconv to feature

# ===================== (e) Shared Task Head =====================
draw_box(ax, 14, 14, 72, 10, C_HEAD, '共享任务头 (Shared Task Head)', fontsize=9, bold=True)

# refine
draw_box(ax, 16, 11, 14, 5, '#FFF176', 'Refine 精炼\nConv1d 8→24→8\n残差连接', fontsize=6)
# freq_proj
draw_box(ax, 33, 11, 14, 5, '#FFEE58', '频率投影\nLinear 1024→512\n→1246', fontsize=6)
# Dual path
draw_box(ax, 50, 13.5, 12, 4.5, '#FFF59D', '宽带路径\nConv1d 8→6→1\nk=7,5', fontsize=5.5)
draw_box(ax, 64, 13.5, 12, 4.5, '#FFF59D', '调性峰值路径\nConv1d 8→6→1\nk=1', fontsize=5.5)
# Merge
draw_box(ax, 78, 17, 8, 3, '#F9A825', '相加\n融合', fontsize=6, bold=True, text_color='white')

draw_arrow(ax, 30, 13.5, 33, 13.5, lw=0.8)
draw_arrow(ax, 47, 15.8, 50, 15.8, lw=0.8)
draw_arrow(ax, 47, 13.5, 50, 13.5, lw=0.8)  # freq_proj split to both paths
draw_arrow(ax, 62, 15.8, 64, 15.8, lw=0.8)
draw_arrow(ax, 76, 18, 78, 18, lw=1.0)  # merge

# Split arrow from freq_proj to dual paths
ax.annotate('', xy=(50, 14.5), xytext=(47, 14.5),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.6))
ax.annotate('', xy=(64, 14.5), xytext=(47, 14.5),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.6))

# Output
draw_box(ax, 30, 19, 38, 2.5, '#F57F17', '输出频谱 [B, 1246]  20~5000 Hz 对数采样', 
         fontsize=7, bold=True, text_color='white')

draw_v_arrow(ax, 44, 22.5, 19, C_ARROW, 1.0)

# ===================== (f) Loss Function =====================
draw_box(ax, 14, 1.5, 72, 7, C_LOSS, '多分辨率物理损失函数 (Multi-Scale Physics-Informed Loss)', fontsize=9, bold=True)

# Loss components
draw_box(ax, 16, 3.5, 18, 4.2, '#FFCC80', '多分辨率 MSE\n(权重=5.0)\n原尺度 + 2×↓ + 4×↓', fontsize=6)
draw_box(ax, 37, 3.5, 16, 4.2, '#FFAB91', '线性域峰值损失\nLinear Peak MSE\n(权重=2.0)', fontsize=6)
draw_box(ax, 56, 3.5, 18, 4.2, '#FF8A65', 'Sobolev 梯度损失\n频谱平滑约束\n(权重=3.0)', fontsize=6)
draw_box(ax, 77, 3.5, 7, 4.2, '#E64A19', 'Σ\n总损失', fontsize=7, bold=True, text_color='white')

draw_arrow(ax, 34, 5.6, 37, 5.6, lw=0.8)
draw_arrow(ax, 53, 5.6, 56, 5.6, lw=0.8)
draw_arrow(ax, 74, 5.6, 77, 5.6, lw=1.0)

# Loss → model (backprop)
ax.annotate('反向传播', xy=(40, 11.5), xytext=(40, 8.5),
            arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5, connectionstyle='arc3,rad=0.5'),
            ha='center', fontsize=7, color='#D84315', fontweight='bold')

# ===================== Dimensions and flow arrows (right side) =====================
dims = [
    (90, 89, '[B, 3]'),
    (90, 70, '[B, 256]'),
    (90, 62, '[B, 288]'),
    (90, 38, '[B, 8, 1024]'),
    (90, 21, '[B, 1246]'),
]
for x, y, text in dims:
    ax.text(x, y, text, fontsize=7, color='#546E7A', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECEFF1', edgecolor='#90A4AE', alpha=0.9))

# ===================== Key annotations =====================
# Highlight the 1-head architecture
ax.text(50, 44, '★ 核心创新: 仅使用 1 个共享任务头 (52 → 1 Head)', 
        ha='center', fontsize=8, color=C_HIGHLIGHT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor=C_HIGHLIGHT, alpha=0.9))

# Data efficiency note
ax.text(50, 40.5, '参数量: 1.64M · 每工况训练样本: 865 条 · 数据/参数比: 7.4 条/万参数', 
        ha='center', fontsize=7, color='#616161',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAFAFA', edgecolor='#BDBDBD', alpha=0.8))

# Diagonal arrows between sections
# encoder → decoder
draw_v_arrow(ax, 50, 49, 42, C_ARROW, 1.2)
# decoder → head
draw_v_arrow(ax, 50, 32, 24, C_ARROW, 1.2)

# ===================== Legend =====================
legend_y = 91.5
legend_items = [
    (C_INPUT, '输入层'), (C_ENCODER, '编码器层'), (C_DECODER, '解码器层'),
    (C_HEAD, '任务头'), (C_LOSS, '损失函数'), (C_EMBED, '嵌入层'), (C_DATA, '数据预处理'),
]
for i, (color, label) in enumerate(legend_items):
    x = 15 + i * 10
    ax.add_patch(Rectangle((x, legend_y), 8, 1.5, facecolor=color, edgecolor='#546E7A', linewidth=0.5))
    ax.text(x + 4, legend_y + 0.75, label, ha='center', va='center', fontsize=5.5)

# ===================== Bottom info =====================
ax.text(50, 0.3, '图: PIMBCN V13 模型总体架构 | 研究生毕业论文插图 | 3:4 比例', 
        ha='center', fontsize=6, color='#9E9E9E')

plt.savefig(os.path.join(out_dir, 'fig_architecture_v13.png'), dpi=300, facecolor='white')
plt.close()
print(f"架构图已保存至: {out_dir}/fig_architecture_v13.png")
print("架构图绘制完成!")
