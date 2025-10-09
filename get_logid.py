import requests
import json
import sys
import re
import time
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 配置部分 ---

# API 相关配置
API_URL = "********"
BEARER_TOKEN = '********'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Domain': '*****'
}

# 并发与重试配置
MAX_WORKERS = 10          # 最大并发线程数
MAX_RETRIES = 3           # 最大重试次数 (总共会尝试 1 + 3 = 4 次)
INITIAL_BACKOFF_MIN_S = 5 # 首次退避的最小随机秒数
INITIAL_BACKOFF_MAX_S = 30# 首次退避的最大随机秒数
REQUEST_TIMEOUT = 200     # 单次API请求的超时时间 (秒)
OUTPUT_DIR = "temp"       # 日志输出目录

# --- 2. 核心函数：API调用、重试与文件保存 ---

def fetch_and_save_data(logid):
    """
    为单个 logid 获取数据，包含指数退避重试逻辑，并保存到文件。
    这是一个独立的、健壮的工作单元，适合在线程中运行。
    
    :param logid: 要查询的 logid
    :return: 一个描述操作结果的字符串
    """
    payload = {
        "logid": str(logid),
        "scan_span_in_min": 1
    }
    output_filename = os.path.join(OUTPUT_DIR, f"{logid}.json")
    initial_sleep_base = random.uniform(INITIAL_BACKOFF_MIN_S, INITIAL_BACKOFF_MAX_S)
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            print(f"LogID: {logid} | 尝试 {attempt + 1}/{MAX_RETRIES + 1} | 发送请求...")
            
            response = requests.post(
                API_URL, 
                headers=HEADERS, 
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data_to_save = response.json()
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            
            success_msg = f"成功: LogID {logid} 的数据已保存至 '{output_filename}'"
            print(success_msg)
            return success_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"错误: LogID {logid} | 尝试 {attempt + 1} 失败: {e}"
            print(error_msg)
            
            if attempt < MAX_RETRIES:
                sleep_duration = initial_sleep_base * (2 ** attempt)
                print(f"LogID: {logid} | {sleep_duration:.2f} 秒后进行下一次重试...")
                time.sleep(sleep_duration)
            else:
                final_error_msg = f"失败: LogID {logid} | 在尝试 {MAX_RETRIES + 1} 次后依然失败，放弃任务。"
                print(final_error_msg)
                return final_error_msg
                
    return f"未知状态: LogID {logid} 的任务循环异常结束。"

# --- 3. 主执行流程 ---

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取初始JSON文件
input_filename = "search_result_data.json"
try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
except FileNotFoundError:
    print(f"错误：输入文件 '{input_filename}' 未找到。脚本已终止。")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"错误：文件 '{input_filename}' 不是有效的JSON格式。脚本已终止。")
    sys.exit(1)

# 提取所有 logid
logs = data_dict.get('data', {}).get('logs', [])
if not logs:
    print("在输入文件中未找到任何日志条目 ('data.logs'为空或不存在)。")
    sys.exit(0)

regex_pattern = r'"logid":"([^"]+)"'
logid_list = []
for log in logs:
    log_messages = log.get('_msg', {}).get('value', '')
    match = re.search(regex_pattern, log_messages)
    if match:
        logid_list.append(match.group(1))

if not logid_list:
    print("已解析日志，但在 '_msg' 字段中未提取到任何 logid。")
    sys.exit(0)
    
print(f"发现 {len(logid_list)} 个 logid，开始并发处理...")

# 使用线程池并发执行所有任务
results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_logid = {executor.submit(fetch_and_save_data, logid): logid for logid in logid_list}
    
    for future in as_completed(future_to_logid):
        try:
            result_message = future.result()
            results.append(result_message)
        except Exception as exc:
            logid = future_to_logid[future]
            error_info = f"严重错误: LogID {logid} 的任务线程奔溃: {exc}"
            print(error_info)
            results.append(error_info)

# 打印最终的执行结果汇总
print("\n" + "="*20 + " 所有任务已完成 " + "="*20)
print("执行结果汇总:")
for result in results:
    print(f"- {result}")
print("="*55)