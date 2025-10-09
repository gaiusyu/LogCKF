import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

data = """sessions_processed,branch,weighted_precision,weighted_recall,weighted_f1,macro_precision,macro_recall,macro_f1,interventions_in_batch
200,standard,0.8463,0.7900,0.8046,0.7127,0.7938,0.7134,0
200,hitl,0.8465,0.8163,0.8211,0.8595,0.8472,0.8434,2
400,standard,0.8487,0.7857,0.7983,0.7287,0.7947,0.7183,0
400,hitl,0.8628,0.8299,0.8349,0.8132,0.8550,0.8237,0
600,standard,0.8337,0.7823,0.7887,0.7319,0.7790,0.7177,0
600,hitl,0.8519,0.8282,0.8320,0.8138,0.8186,0.8087,1
800,standard,0.8293,0.7792,0.7882,0.7194,0.7704,0.7120,0
800,hitl,0.8536,0.8338,0.8373,0.8032,0.8132,0.8025,0
1000,standard,0.8297,0.7739,0.7849,0.7133,0.7731,0.7066,0
1000,hitl,0.8540,0.8320,0.8358,0.8024,0.8243,0.8073,0
1200,standard,0.8417,0.7864,0.7983,0.7195,0.7822,0.7145,0
1200,hitl,0.8619,0.8433,0.8469,0.8026,0.8334,0.8131,0
1400,standard,0.8339,0.7812,0.7919,0.7182,0.7760,0.7129,0
1400,hitl,0.8592,0.8384,0.8424,0.8009,0.8295,0.8097,0
1600,standard,0.8347,0.7807,0.7917,0.7210,0.7711,0.7128,0
1600,hitl,0.8620,0.8459,0.8487,0.8124,0.8309,0.8173,1
1800,standard,0.8319,0.7804,0.7906,0.7187,0.7704,0.7119,0
1800,hitl,0.8649,0.8473,0.8504,0.8166,0.8353,0.8210,0
2000,standard,0.8251,0.7680,0.7798,0.7068,0.7565,0.6965,0
2000,hitl,0.8615,0.8433,0.8465,0.8082,0.8337,0.8159,0
"""
df = pd.read_csv(StringIO(data))

std = df[df['branch'] == 'standard']
hitl = df[df['branch'] == 'hitl']

def plot_metric_with_feedback(metric, ylabel, filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(6, 4), dpi=300)

    std_color = '#1f77b4'
    hitl_color = '#d62728'

    # --- 基本曲线 ---
    plt.plot(std['sessions_processed'], std[metric], 'o-', color=std_color, linewidth=2, label='Standard')
    plt.plot(hitl['sessions_processed'], hitl[metric], 'o--', color=hitl_color, linewidth=2, label='HITL')

    # --- 标记 Human Feedback 点 ---
    interventions = hitl[hitl['interventions_in_batch'] > 0]
    plt.scatter(interventions['sessions_processed'], interventions[metric],
                color='gold', edgecolors='black', s=80, zorder=5, marker='*', label='Human Feedback')

    # --- 可选注释 ---
    for _, row in interventions.iterrows():
        plt.text(row['sessions_processed'] + 30, row[metric] + 0.002,
                 f"+{row['interventions_in_batch']}", fontsize=8, color='black')

    # --- 轴与样式 ---
    plt.xlabel('Sessions Processed', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Weighted {ylabel} vs. Sessions (with Human Interventions)', fontsize=13, weight='semibold', pad=10)
    plt.legend(fontsize=9, frameon=True, loc='lower right')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0.75, 0.88)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# 分别绘制三张图
plot_metric_with_feedback('weighted_precision', 'Precision', 'weighted_precision_feedback.pdf')
plot_metric_with_feedback('weighted_recall', 'Recall', 'weighted_recall_feedback.pdf')
plot_metric_with_feedback('weighted_f1', 'F1 Score', 'weighted_f1_feedback.pdf')
