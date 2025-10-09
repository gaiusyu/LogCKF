import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys

# 设置 Matplotlib 支持中文显示
# (请确保您的系统已安装支持中文的字体，如'SimHei', 'Microsoft YaHei'等)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    print("警告: 未找到中文字体'SimHei'，图片中的中文可能无法正常显示。")
    print("您可以尝试安装'SimHei'字体或在此处修改为系统已有的中文字体，如 'Microsoft YaHei'。")


def calculate_metrics_from_csv(input_file: str, true_col: str, pred_col: str, output_png: str):
    """
    从 CSV 文件中读取实验结果，计算并报告详细的分类指标。

    Args:
        input_file (str): 包含实验结果的 CSV 文件路径。
        true_col (str): CSV 文件中真实标签所在的列名。
        pred_col (str): CSV 文件中预测标签所在的列名。
        output_png (str): 生成的混淆矩阵图片的保存路径。
    """
    # --- 1. 数据加载与验证 ---
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在。")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"错误: 无法读取 CSV 文件。原因: {e}")
        sys.exit(1)

    if true_col not in df.columns or pred_col not in df.columns:
        print(f"错误: 列名 '{true_col}' 或 '{pred_col}' 在文件中未找到。")
        print(f"文件中的可用列为: {df.columns.tolist()}")
        sys.exit(1)
        
    # 移除包含空值的行，确保数据清洁
    df.dropna(subset=[true_col, pred_col], inplace=True)
    
    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    if len(y_true) == 0:
        print("错误: 清理后没有有效的数据行可供分析。")
        sys.exit(1)

    # 获取所有出现过的标签，并排序以保证一致性
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    print("\n" + "="*80)
    print(" " * 25 + "分类性能评估报告")
    print("="*80)

    # --- 2. 计算每个类别的详细指标 ---
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    print("\n--- 各类别详细指标 ---\n")
    header = f"{'类别':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'样本数 (Support)':<15}"
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        print(f"{label:<30} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {s[i]:<15}")
    print("-" * len(header))

    # --- 3. 计算总体平均指标 ---
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- 总体平均指标 ---\n")
    print(f"{'平均方法':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    print(f"{'Micro Average':<18} | {micro_p:<10.4f} | {micro_r:<10.4f} | {micro_f1:<10.4f}")
    print(f"{'Macro Average':<18} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f}")
    print(f"{'Weighted Average':<18} | {weighted_p:<10.4f} | {weighted_r:<10.4f} | {weighted_f1:<10.4f}")
    print("\n" + "="*80)

    # --- 4. 生成并保存混淆矩阵 ---
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # 动态调整图片大小，防止标签过多时重叠
        fig_height = max(6, len(labels) * 0.6)
        fig_width = max(8, len(labels) * 0.8)
        
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('真实标签 (True Label)', fontsize=12)
        plt.xlabel('预测标签 (Predicted Label)', fontsize=12)
        plt.title('混淆矩阵 (Confusion Matrix)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_png)
        print(f"\n✅ 混淆矩阵已保存至: '{output_png}'")
    except Exception as e:
        print(f"\n❌ 生成混淆矩阵失败。原因: {e}")


if __name__ == "__main__":
    # 使用 argparse 来处理命令行参数，使脚本更灵活
    parser = argparse.ArgumentParser(
        description="从CSV文件计算并报告多分类任务的性能指标。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    
    parser.add_argument(
        'input_file', 
        type=str, 
        help="必须参数。包含实验结果的CSV文件路径。\n例如: 'my_results.csv'"
    )
    # ---  这里是修正的地方 ---
    parser.add_argument(
        '--true-col', 
        type=str, 
        default='true_label', 
        help="可选参数。CSV中真实标签的列名 (默认为 'true_label')。"
    )
    parser.add_argument(
        '--pred-col', 
        type=str, 
        default='predicted_label', 
        help="可选参数。CSV中预测标签的列名 (默认为 'predicted_label')。"
    )
    parser.add_argument(
        '--output-png', 
        type=str, 
        default='confusion_matrix.png', 
        help="可选参数。混淆矩阵图片的输出文件名 (默认为 'confusion_matrix.png')。"
    )

    # 如果没有提供参数，则打印帮助信息
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    calculate_metrics_from_csv(args.input_file, args.true_col, args.pred_col, args.output_png)