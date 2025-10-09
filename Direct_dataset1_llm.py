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
import threading 
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from collections import Counter 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- 1. Global Configurations ---
# --- LLM and Environment Settings ---
INTERNAL_LLM_BASE_URL = "https://sre-data-mlops-serve.byted.org/llm_service/v1"
INTERNAL_LLM_API_KEY = "dummy-key"
USER_ALIAS = "yusiyu.6@bytedance.com"
NAMESPACE = "sdk_test"

# --- Dataset and File Paths ---
DATA_DIR = "./"
LOG_DATA_FILE = os.path.join(DATA_DIR, "preliminary_sel_log_dataset.csv")
LABEL_FILE_1 = os.path.join(DATA_DIR, "preliminary_train_label_dataset.csv")
LABEL_FILE_2 = os.path.join(DATA_DIR, "preliminary_train_label_dataset_s.csv")

# --- Fault Category Definitions ---
FAULT_CATEGORY_DESCRIPTIONS = {
    "CPU Fault": "Faults related to CPU performance, utilization, or errors. May involve high load, core failures, or specific CPU error messages.",
    "Memory Fault": "Faults related to memory allocation, corruption, or exhaustion. Often indicated by 'out of memory' errors, segmentation faults, or ECC errors.",
    "Other Fault": "A general category for any other type of hardware or system fault not related to CPU or Memory, such as disk, network, or power supply issues."
}
RAW_LABEL_TO_CATEGORY = {0: "CPU Fault", 1: "CPU Fault", 2: "Memory Fault", 3: "Other Fault"}
FAULT_CATEGORIES = list(FAULT_CATEGORY_DESCRIPTIONS.keys())
LAST_RESORT_FAULT_CATEGORY = "Other Fault"

# --- Framework Settings ---
MAX_WORKERS = 16
TASK_LIMIT = 5000
MAX_LOG_LINES_IN_PROMPT = 500
RANDOM_SEED = 42 # For reproducible sampling
ENABLE_MEMORY = True 

# --- 2. LogParser 和 MemoryStore ---
class LogParser:
    """
    一个简单的日志解析器，用于从原始日志中提取模板。
    """
    def process_logs(self, log_df: pd.DataFrame) -> frozenset:
        """
        处理日志DataFrame，返回一个包含所有模板的、不可变的frozenset。
        frozenset可以作为字典的键。
        """
        templates = set()
        for _, row in log_df.iterrows():
            log_message = str(row.get('msg', ''))
            if not log_message: continue
            
            # 简单的模板化规则：将所有数字替换为 <*>
            template_text = re.sub(r'\d+', '<*>', log_message)
            templates.add(template_text)
            
        return frozenset(templates)

class MemoryStore:
    """
    一个线程安全的、基于模板哈希的记忆存储。
    """
    def __init__(self):
        # 使用字典来存储，key是模板的frozenset，value是预测的标签
        self.store = {}
        self.lock = threading.Lock()

    def add(self, template_set: frozenset, predicted_label: str):
        if not template_set: return
        with self.lock:
            # 只有当这个模板集合不存在时才添加
            if template_set not in self.store:
                self.store[template_set] = predicted_label

    def search(self, template_set: frozenset) -> str:
        if not template_set: return None
        # 这是一个原子操作，通常不需要锁，但为了保持风格一致性加上
        with self.lock:
            return self.store.get(template_set)

    def __len__(self):
        with self.lock:
            return len(self.store)

# --- 3. LLM and Utility Functions ---
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
    """Parses the LLM's free-text response to find a valid fault category."""
    for category in FAULT_CATEGORIES:
        if re.search(r'\b' + re.escape(category) + r'\b', response_text, re.IGNORECASE):
            return category
    return LAST_RESORT_FAULT_CATEGORY

# --- 4. 评估函数 ---
def calculate_and_save_metrics(y_true: List[str], y_pred: List[str], from_memory_flags: List[bool], output_dir: str):
    """Calculates and saves all metrics, including memory hit rate."""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')

    report_lines = []
    report_lines.append("### Simple LLM Benchmark (with Memory) Evaluation Report ###\n")

    if not y_true or not y_pred:
        report_lines.append("No valid results to evaluate.")
        with open(report_path, 'w') as f: f.write("\n".join(report_lines))
        print("\n".join(report_lines))
        return

    # --- 计算并记录 Memory Hit Rate ---
    memory_hits = sum(from_memory_flags)
    total_tasks = len(from_memory_flags)
    hit_rate = (memory_hits / total_tasks) * 100 if total_tasks > 0 else 0
    report_lines.append(f"--- Memory Usage ---")
    report_lines.append(f"Memory Hits: {memory_hits} / {total_tasks} ({hit_rate:.2f}%)\n")
    
    # --- 保持原有的指标计算 ---
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    report_lines.append("\n--- 各类别详细指标 ---\n")
    header = f"{'类别':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'样本数 (Support)':<15}"
    report_lines.append(header); report_lines.append("-" * len(header))
    for i, label in enumerate(labels):
        report_lines.append(f"{label:<20} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {s[i]:<15}")
    report_lines.append("-" * len(header))
    
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
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
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('真实标签'); plt.xlabel('预测标签'); plt.title('混淆矩阵 - 简单LLM基准测试 (带记忆)')
    plt.tight_layout(); plt.savefig(cm_path)
    print(f"--- 混淆矩阵已保存至: {cm_path} ---")

# --- 5.  Main Diagnosis Class ---
class SimpleLLMBenchmark:
    def __init__(self, task_limit: int, model_name: str):
        self.task_limit = task_limit
        self.model_name = model_name
        self.tasks_df = None
        self.log_df = None
        self.output_dir = None
        # --- 初始化 parser 和 memory store ---
        self.log_parser = LogParser()
        self.memory_store = MemoryStore()

    def _setup_output_dir(self):
        safe_model_name = re.sub(r'[^\w\-_\.]', '_', self.model_name)
        self.output_dir = f"Ali_simple_mem_{safe_model_name}_{self.task_limit}_tasks_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"--- 结果将保存在: {self.output_dir} ---")

    def _prepare_data(self):
        print("--- [数据准备] 开始加载日志和标签数据...")
        self.log_df = pd.read_csv(LOG_DATA_FILE)
        self.log_df['time_unix'] = pd.to_datetime(self.log_df['time'], errors='coerce').astype(np.int64) // 10**9
        self.log_df.dropna(subset=['time_unix'], inplace=True)
        self.log_df.set_index('sn', inplace=True)
        
        labels_df = pd.concat([pd.read_csv(f) for f in [LABEL_FILE_1, LABEL_FILE_2] if os.path.exists(f)], ignore_index=True).drop_duplicates()
        labels_df['label_str'] = labels_df['label'].map(RAW_LABEL_TO_CATEGORY)
        labels_df['fault_time_unix'] = pd.to_datetime(labels_df['fault_time'], errors='coerce').astype(np.int64) // 10**9
        labels_df.dropna(subset=['label_str', 'fault_time_unix'], inplace=True)
        
        all_tasks_df = labels_df[['sn', 'fault_time_unix', 'label_str']]
        print(f"--- 共加载 {len(all_tasks_df)} 个有效故障事件。---")

        # --- 这里改用随机抽样，而不是.head()，以保证评估的公平性 ---
        if len(all_tasks_df) > self.task_limit:
            print(f"--- 任务总数超过上限 {self.task_limit}，将随机抽取 {self.task_limit} 个任务进行处理... ---")
            self.tasks_df = all_tasks_df.sample(n=self.task_limit, random_state=RANDOM_SEED)
        else:
            self.tasks_df = all_tasks_df
        
        print(f"--- 本次运行将处理 {len(self.tasks_df)} 个任务。---")

    def _get_related_logs(self, sn: str, fault_time_unix: int) -> pd.DataFrame:
        try:
            server_logs = self.log_df.loc[sn].copy()
            if isinstance(server_logs, pd.Series): server_logs = pd.DataFrame([server_logs])
        except KeyError: return None
        window = 36000
        related_logs = server_logs[(server_logs['time_unix'] >= fault_time_unix - window) & (server_logs['time_unix'] <= fault_time_unix)]
        return related_logs if not related_logs.empty else None

    # ---  记忆逻辑 ---
    def run_simple_diagnosis(self, sn: str, fault_time: int, true_label: str) -> Tuple[str, str, str, bool]:
        task_id = f"{sn}_{fault_time}"
        related_logs_df = self._get_related_logs(sn, fault_time)
        
        if related_logs_df is None or related_logs_df.empty:
            return task_id, true_label, "Error_NoLogs", False

        # 1. 解析日志，获取模板集合
        template_set = self.log_parser.process_logs(related_logs_df)

        # 2. 检查记忆库
        if ENABLE_MEMORY:
            cached_result = self.memory_store.search(template_set)
            if cached_result:
                tqdm.write(f"--- [Memory Hit] Task {task_id}. Reusing result: {cached_result} ---")
                return task_id, true_label, cached_result, True

        # 3. 如果未命中，调用LLM
        log_lines = related_logs_df.sort_values('time_unix', ascending=False).head(MAX_LOG_LINES_IN_PROMPT)
        raw_log_text = "\n".join(log_lines.apply(lambda row: f"{row['time']} {row['msg']}", axis=1))

        code_block = "```"
        system_prompt = "You are an expert AI system diagnoser. Analyze the provided logs to identify the single most likely fault category from a given list. Respond with only the name of the fault category."
        description_block = "\n".join([f"- {name}: {desc}" for name, desc in FAULT_CATEGORY_DESCRIPTIONS.items()])
        
        prompt = f"""### Task: Diagnose Fault from Logs
Analyze the following system logs and determine the most appropriate fault category.
### Available Fault Categories:
{description_block}
### System Logs:
{code_block}
{raw_log_text}
{code_block}
### Your Answer:
Based on the logs, what is the single most likely fault category? Choose one from {FAULT_CATEGORIES}.
"""
        try:
            llm_response = call_llm(prompt, system_prompt, session_id=f"simple-{task_id}", model_name=self.model_name)
            predicted_label = parse_llm_output(llm_response)
            
            # 4. 将新结果存入记忆库
            if ENABLE_MEMORY:
                self.memory_store.add(template_set, predicted_label)
                tqdm.write(f"--- [Memory Update] Task {task_id} added. Store size: {len(self.memory_store)}. ---")

            return task_id, true_label, predicted_label, False
        except Exception as e:
            tqdm.write(f"\n!!! Critical Error on task {task_id}: {e}")
            return task_id, true_label, "Error_LLM", False

    def run_analysis(self):
        self._setup_output_dir()
        self._prepare_data()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.run_simple_diagnosis, r['sn'], r['fault_time_unix'], r['label_str']): f"{r['sn']}_{r['fault_time_unix']}"
                for _, r in self.tasks_df.iterrows()
            }
            
            progress_bar = tqdm(as_completed(futures), total=len(futures), desc=f"Running Benchmark ({self.model_name}, {self.task_limit} tasks)")
            for future in progress_bar:
                try: results.append(future.result())
                except Exception as e: tqdm.write(f"\nA task failed to execute: {e}")
        
        # --- 保存的结果包含 is_from_memory ---
        detailed_df = pd.DataFrame(results, columns=['task_id', 'true_label', 'predicted_label', 'is_from_memory'])
        detailed_df.to_csv(os.path.join(self.output_dir, 'simple_benchmark_predictions.csv'), index=False)
        
        valid_results = [r for r in results if "Error" not in r[2]]
        y_true = [r[1] for r in valid_results]
        y_pred = [r[2] for r in valid_results]
        from_memory_flags = [r[3] for r in valid_results]
        
        calculate_and_save_metrics(y_true, y_pred, from_memory_flags, self.output_dir)

# --- 6. Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行一个带记忆功能的简单LLM基准测试，用于日志故障诊断。")
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

    if not all(os.path.exists(f) for f in [LOG_DATA_FILE, LABEL_FILE_1, LABEL_FILE_2]):
        print(f"错误: 一个或多个数据/标签文件未找到。请检查文件路径。")
    else:
        benchmark = SimpleLLMBenchmark(task_limit=args.task_limit, model_name=args.model_name)
        benchmark.run_analysis()