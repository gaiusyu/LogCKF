import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# --- Matplotlib 全局配置（论文风格） ---
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'pdf.fonttype': 42, 
    'ps.fonttype': 42,
})

# --- Data ---
settings = [
    'w/o Contrastive\nKnowledge Refinement\n(Same Type)',
    'w/o Contrastive\nKnowledge Refinement\n(Cross Type)',
    'LogCKF\n(Full Model)'
]
dataset_labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']
data = {
    'Dataset 1': [0.887, 0.956, 0.982],
    'Dataset 2': [0.889, 0.901, 0.908],
    'Dataset 3': [0.395, 0.518, 0.597],
}

# --- Plotting ---
n_settings = len(settings)
n_datasets = len(dataset_labels)
index = np.arange(n_datasets)
bar_width = 0.25

fig, ax = plt.subplots(figsize=(6.0, 3.8))

colors = ['#a6bddb', '#74c476', '#fdae6b']

# 绘制柱状图
for i, setting in enumerate(settings):
    scores = [data[ds][i] for ds in dataset_labels]
    position = index - bar_width + (i * bar_width)
    bars = ax.bar(position, scores, bar_width,
                  label=setting, color=colors[i],
                  edgecolor='black', linewidth=0.7)
    
    # 数据标签稍微上移
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, color='black')

# --- 坐标轴与标题 ---
ax.set_ylabel('Weighted F1-Score', fontweight='bold')
ax.set_xlabel('Dataset', fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(dataset_labels)
ax.set_ylim(0.38, 1.05)
ax.set_title('Ablation Study of LogCKF Components', pad=10, fontweight='bold')

# --- 网格与边框 ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
ax.xaxis.grid(False)

# --- 图例 ---
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),  
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.4,
    title_fontsize=10
)

# --- 布局与保存 ---
fig.tight_layout()
plt.subplots_adjust(bottom=0.35) 
plt.savefig('ablation_study_chart_topconf.pdf', dpi=600, bbox_inches='tight')
plt.savefig('ablation_study_chart_topconf.png', dpi=600, bbox_inches='tight')
plt.show()
