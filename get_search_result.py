import requests
import json
import sys

# --- 1. 配置部分 ---
filename = "Search_response_task.json"

# 使用 with open() 安全地打开和关闭文件
# 'r' 表示读取模式, encoding='utf-8' 是最佳实践
with open(filename, 'r', encoding='utf-8') as f:
    # 使用 json.load() 从文件中解析JSON数据
    data_dict = json.load(f)

# --- 现在 data_dict 就是一个标准的 Python 字典了 ---

# 1. 获取顶层的键，比如 'status'
status = data_dict['data']

# API 端点
API_URL = "*******"  # <-- 替换成你的真实 URL

# 请求头 (对应 curl 的 -H 参数)
# !!! 重要：请将 'your_token_here' 替换为你的真实 Bearer Token !!!
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ********',
    # 如果有其他Header，比如 'Domain'，也加在这里
    'Domain': 'streamlog'
}

# 请求体/载荷 (对应 curl 的 -d @payload.json)
# 你可以直接在Python中定义这个字典，也可以从文件读取
PAYLOAD =status


# 输出文件的名称
OUTPUT_FILENAME = "search_result_data.json"


def fetch_and_save_data():
    """
    主函数：发送API请求并将结果保存到文件。
    """
    print(f"Sending POST request to: {API_URL}")
    
    # --- 2. 发送请求 (执行 curl) ---
    try:
        response = requests.post(
            API_URL, 
            headers=HEADERS, 
            json=PAYLOAD,  # requests库会自动将字典转为JSON字符串
            timeout=20     # 设置20秒超时，防止永久等待
        )
        
        # 检查HTTP响应状态码，如果不是 2xx，则会抛出异常
        response.raise_for_status()
        
        # 将返回的JSON响应解析为Python字典
        data_to_save = response.json()
        print("Request successful! Received data.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Status Code: {http_err.response.status_code}")
        print(f"Response Body: {http_err.response.text}")
        sys.exit(1) # 退出脚本，因为请求失败了
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Failed to decode response as JSON. The server might have returned non-JSON content.")
        print(f"Raw Response: {response.text}")
        sys.exit(1)
        
    # --- 3. 存储到文件 (对应 curl 的 > output.json) ---
    try:
        # 使用 'w'模式（写入），并指定 utf-8 编码以支持所有字符
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            # 使用 json.dump() 将Python字典写入文件
            # indent=4 会让JSON文件格式化，非常易于阅读
            # ensure_ascii=False 确保中文等非ASCII字符能正确显示，而不是被转义
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully saved data to '{OUTPUT_FILENAME}'")

    except IOError as e:
        print(f"Error writing to file '{OUTPUT_FILENAME}': {e}")
        sys.exit(1)


# --- 运行主函数 ---
if __name__ == "__main__":
    fetch_and_save_data()