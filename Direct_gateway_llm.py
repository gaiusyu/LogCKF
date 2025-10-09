import pandas as pd
import json
import os
import re
import time
import openai
import numpy as np
import sys
import random
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- 1. Global Configurations ---
# --- LLM and Environment Settings ---
INTERNAL_LLM_BASE_URL = "*******"
INTERNAL_LLM_API_KEY = "dummy-key"
USER_ALIAS = "****"
NAMESPACE = "sdk_test"

# --- Dataset and File Paths for the new dataset ---
DATA_DIR = "./dignosis_dataset/"
LABEL_FILE = os.path.join(DATA_DIR, "label.txt")
LOG_FILES_DIR = os.path.join(DATA_DIR, "log_analysis_runs")

# --- Fault Category Definitions ---
FAULT_CATEGORY_DESCRIPTIONS = {
    "Resource not found": "The request failed because a required resource, such as a database, table, or API endpoint, could not be located.",
    "InternalServerError": "The request failed due to a bug or unhandled exception within the backend service's code, often indicated by a programming error traceback in the logs.",
    "Rate limiting": "The request was actively terminated by the system for exceeding resource usage limits, such as a query execution timeout or an API request frequency cap.",
    "Infrastructure Configuration Error": "The request failed because the backend service could not communicate with a dependent infrastructure component, such as a downstream service, database, or cache, due to a timeout, network issue, or misconfiguration.",
    "Normal": "No fault detected; logs indicate normal system operation.",
}
FAULT_CATEGORIES = list(FAULT_CATEGORY_DESCRIPTIONS.keys())
FALLBACK_CATEGORY = "Normal" # Used if LLM fails or gives an invalid answer

# --- Framework Settings ---
MAX_WORKERS = 16
TASK_LIMIT = 5000 # Default task limit, can be overridden by args
MAX_LOG_LINES_IN_PROMPT = 500
RANDOM_SEED = 42 # For reproducible sampling

# --- 2. LLM and Utility Functions ---
def call_llm(prompt: str, system_prompt: str, session_id: str, model_name: str) -> str:
    """A simplified LLM call function that expects a plain text response."""
    try:
        client = OpenAI(base_url=INTERNAL_LLM_BASE_URL, api_key=INTERNAL_LLM_API_KEY)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize internal OpenAI client: {e}")
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                extra_headers={"x-extra-session-id": session_id, "x-extra-user-alias": USER_ALIAS, "x-extra-namespace": NAMESPACE},
                model=model_name,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            tqdm.write(f"LLM call failed for session {session_id}, retrying... Error: {e}")
            time.sleep(5)
    raise Exception(f"LLM call failed after 3 retries for session {session_id}")

def parse_llm_output(response_text: str) -> str:
    """Parses the LLM's free-text response to find the best-matching fault category."""
    # Prioritize exact matches first
    for category in FAULT_CATEGORIES:
        # Use word boundaries to avoid partial matches (e.g., "Error" in "InternalServerError")
        if re.search(r'\b' + re.escape(category) + r'\b', response_text, re.IGNORECASE):
            return category
    
    # If no exact match, check for case-insensitive keywords as a fallback
    for category in FAULT_CATEGORIES:
        if category.lower() in response_text.lower():
            return category
            
    return FALLBACK_CATEGORY

def calculate_and_save_metrics(y_true: List[str], y_pred: List[str], output_dir: str):
    """Calculates and saves all metrics to a file and generates a confusion matrix."""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')

    report_lines = []
    report_lines.append("### Simple LLM Benchmark (New Dataset) Evaluation Report ###\n")

    if not y_true or not y_pred:
        report_lines.append("No valid results to evaluate.")
        with open(report_path, 'w', encoding='utf-8') as f: f.write("\n".join(report_lines))
        print("\n".join(report_lines))
        return
    
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    report_lines.append("\n--- 各类别详细指标 ---\n")
    header = f"{'类别':<35} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'样本数 (Support)':<15}"
    report_lines.append(header); report_lines.append("-" * len(header))
    for i, label in enumerate(labels):
        report_lines.append(f"{label:<35} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {s[i]:<15}")
    report_lines.append("-" * len(header))
    
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    report_lines.append("\n--- 总体平均指标 ---\n")
    report_lines.append(f"{'平均方法':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    report_lines.append("-" * 55)
    report_lines.append(f"{'Micro Average':<18} | {micro_p:<10.4f} | {micro_r:<10.4f} | {micro_f1:<10.4f}")
    report_lines.append(f"{'Macro Average':<18} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f}")
    report_lines.append(f"{'Weighted Average':<18} | {weighted_p:<10.4f} | {weighted_r:<10.4f} | {weighted_f1:<10.4f}")

    final_report = "\n".join(report_lines)
    print("\n" + final_report)
    with open(report_path, 'w', encoding='utf-8') as f: f.write(final_report)
    print(f"\n--- 完整评估报告已保存至: {report_path} ---")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('真实标签'); plt.xlabel('预测标签'); plt.title('混淆矩阵 - 简单LLM基准测试 (新数据集)')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(cm_path)
    print(f"--- 混淆矩阵已保存至: {cm_path} ---")

# --- 3. Main Diagnosis Class ---
class SimpleBenchmarkForNewDataset:
    def __init__(self, task_limit: int, model_name: str):
        self.task_limit = task_limit
        self.model_name = model_name
        self.tasks_df = None
        self.output_dir = None

    def _setup_output_dir(self):
        safe_model_name = re.sub(r'[^\w\-_\.]', '_', self.model_name)
        self.output_dir = f"simple_benchmark_{safe_model_name}_{self.task_limit}_tasks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"--- 结果将保存在: {self.output_dir} ---")

    def _prepare_data(self):
        print("--- [数据准备] 开始加载标签数据...")
        if not os.path.exists(LABEL_FILE):
            raise FileNotFoundError(f"Label file not found at: {LABEL_FILE}")
            
        tasks = []
        with open(LABEL_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    session_id, label = line.split(',', 1)
                    if label in FAULT_CATEGORIES:
                        tasks.append({'session_id': session_id, 'label_str': label})

        all_tasks_df = pd.DataFrame(tasks)
        print(f"--- 共加载 {len(all_tasks_df)} 个有效故障事件。---")

        if len(all_tasks_df) > self.task_limit:
            print(f"--- 任务总数超过上限 {self.task_limit}，将随机抽取 {self.task_limit} 个任务进行处理... ---")
            self.tasks_df = all_tasks_df.sample(n=self.task_limit, random_state=RANDOM_SEED)
        else:
            self.tasks_df = all_tasks_df
        
        print(f"--- 本次运行将处理 {len(self.tasks_df)} 个任务。---")

    def _get_related_logs_text(self, session_id: str) -> str:
        log_file_path = os.path.join(LOG_FILES_DIR, f"{session_id}.log")
        if not os.path.exists(log_file_path):
            tqdm.write(f"Warning: Log file not found for session {session_id}")
            return None
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # If logs are too long, take the last N lines which are often more relevant
                if len(lines) > MAX_LOG_LINES_IN_PROMPT:
                    lines = lines[-MAX_LOG_LINES_IN_PROMPT:]
                return "".join(lines)
        except Exception as e:
            tqdm.write(f"Error reading log file {log_file_path}: {e}")
            return None

    def run_simple_diagnosis(self, session_id: str, true_label: str) -> Tuple[str, str, str]:
        raw_log_text = self._get_related_logs_text(session_id)
        if raw_log_text is None:
            return session_id, true_label, "Error_NoLogs"
        
        code_block = "```"
        system_prompt = "You are an expert AI system diagnoser. Analyze the provided logs to identify the single most likely fault category from the given list. Respond with only the name of the fault category."
        description_block = "\n".join([f"- **{name}**: {desc}" for name, desc in FAULT_CATEGORY_DESCRIPTIONS.items()])
        
        prompt = f"""### Task: Diagnose Fault from Logs
Analyze the following system logs and determine the most appropriate fault category.

### Available Fault Categories:
{description_block}

### System Logs:
{code_block}
{raw_log_text}
{code_block}

### Your Answer:
Based on the logs, what is the single most likely fault category? Choose one from the list: {', '.join(FAULT_CATEGORIES)}.
"""
        try:
            llm_response = call_llm(prompt, system_prompt, session_id=f"simple-{session_id}", model_name=self.model_name)
            predicted_label = parse_llm_output(llm_response)
            return session_id, true_label, predicted_label
        except Exception as e:
            tqdm.write(f"\n!!! Critical Error on task {session_id}: {e}")
            return session_id, true_label, "Error_LLM"

    def run_analysis(self):
        self._setup_output_dir()
        self._prepare_data()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.run_simple_diagnosis, r['session_id'], r['label_str']): r['session_id']
                for _, r in self.tasks_df.iterrows()
            }
            
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc=f"Running Benchmark ({self.model_name}, {self.task_limit} tasks)")
            for future in progress_bar:
                try: 
                    results.append(future.result())
                except Exception as e: 
                    tqdm.write(f"\nA task failed to execute: {e}")
        
        detailed_df = pd.DataFrame(results, columns=['session_id', 'true_label', 'predicted_label'])
        detailed_df.to_csv(os.path.join(self.output_dir, 'simple_benchmark_predictions.csv'), index=False)
        
        valid_results = [r for r in results if "Error" not in r[2]]
        y_true = [r[1] for r in valid_results]
        y_pred = [r[2] for r in valid_results]
        
        calculate_and_save_metrics(y_true, y_pred, self.output_dir)

# --- 4. Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行一个简单的LLM基准测试，用于新数据集的日志故障诊断。")
    parser.add_argument(
        '--task-limit', 
        type=int, 
        default=TASK_LIMIT, 
        help=f"要处理的最大任务数 (默认为: {TASK_LIMIT})。"
    )
    parser.add_argument(
        '--model-name', 
        type=str, 
        default="gpt-4.1-2025-04-14", 
        help="要使用的LLM模型名称 (默认为: 'gpt-4.1-2025-04-14')。"
    )
    args = parser.parse_args()

    if not all(os.path.exists(p) for p in [LABEL_FILE, LOG_FILES_DIR]):
        print(f"错误: 标签文件或日志目录未找到。请检查路径。")
    else:
        benchmark = SimpleBenchmarkForNewDataset(task_limit=args.task_limit, model_name=args.model_name)
        benchmark.run_analysis()