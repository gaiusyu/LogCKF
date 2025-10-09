#!/bin/bash

# ==============================================================================
#            批量运行日志诊断模型实验的自动化脚本
# ==============================================================================

# --- 配置区 ---


PYTHON_SCRIPT="Direct_dataset1_llm.py" 

# 2. 要测试的模型列表
MODELS=(
  "gpt-5-2025-08-07"
  "gpt-4.1-2025-04-14"
  "deepseek-r1"
  "deepseek-v3"
  "doubao-pro-32b"
  "doubao-1.6-20250615"
)

# 3. 每个模型要运行的次数
NUM_RUNS=3

# 4. (可选) 为每次运行设置的任务数量
#    如果您想在运行时也控制任务数，可以取消下面这行的注释，并在命令中加入 --task-limit
# TASK_LIMIT=100

# --- 执行逻辑 ---

echo "==============================================="
echo "        启动实验批量运行程序"
echo "==============================================="
echo ""

# 获取模型总数，用于显示进度
TOTAL_MODELS=${#MODELS[@]}
MODEL_COUNT=0

# 遍历模型列表
for model in "${MODELS[@]}"; do
  MODEL_COUNT=$((MODEL_COUNT + 1))
  echo "-----------------------------------------------"
  echo ">>>>> 开始处理模型 $MODEL_COUNT / $TOTAL_MODELS: $model <<<<<"
  echo "-----------------------------------------------"

  # 为每个模型循环指定的次数
  for (( i=1; i<=NUM_RUNS; i++ )); do
    echo ""
    echo "--- 模型 [$model] - 第 $i / $NUM_RUNS 次运行 ---"
    
    # 构造并执行Python命令
    # 注意: 如果您的Python脚本需要其他参数 (如此前的 --task-limit)，
    # 可以在这里添加，例如:
    # python3 "$PYTHON_SCRIPT" --model-name "$model" --task-limit $TASK_LIMIT
    
    python3 "$PYTHON_SCRIPT" --model-name "$model"

    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
      echo "--- 模型 [$model] - 第 $i 次运行成功。---"
    else
      echo "--- !!! 警告: 模型 [$model] - 第 $i 次运行失败或出错。!!! ---"
    fi
    echo ""
  done
done

echo "==============================================="
echo "       所有实验均已完成！"
echo "==============================================="