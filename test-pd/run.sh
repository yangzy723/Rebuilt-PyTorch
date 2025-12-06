#!/bin/bash

# ================= 配置 =================
COUNTS=(16 32 64 128 256)
REPS=3
RESULT_DIR="results"
# =======================================

echo "========================================"
echo "      LLM Inference Benchmark Tool      "
echo "      (Multi-Batch & 3x Repetitions)    "
echo "========================================"

# 1. 准备环境
rm -rf $RESULT_DIR
mkdir -p $RESULT_DIR

# 2. 开始循环测试
for C in "${COUNTS[@]}"; do
    echo ""
    echo "########################################"
    echo "   Testing Request Count: $C"
    echo "########################################"

    for ((r=1; r<=REPS; r++)); do
        echo "   --- Repetition $r / $REPS ---"
        
        # 定义文件名
        FILE_PREFIX="${RESULT_DIR}/res_c${C}_r${r}"

        # ----------------------------------------------
        # 阶段 1: 串行基准测试 (Serial)
        # ----------------------------------------------
        echo "       [Phase 1] Serial Mode..."
        python prefill_client.py --count $C --output "${FILE_PREFIX}_serial_prefill.json" > /dev/null 2>&1
        sleep 1
        python decode_client.py --count $C --output "${FILE_PREFIX}_serial_decode.json" > /dev/null 2>&1
        
        # ----------------------------------------------
        # 阶段 2: 并行干扰测试 (Parallel)
        # ----------------------------------------------
        echo "       [Phase 2] Parallel Mode..."
        sleep 2 # Cool down

        # 后台运行
        python prefill_client.py --count $C --output "${FILE_PREFIX}_parallel_prefill.json" > /dev/null 2>&1 &
        PID_P=$!
        
        python decode_client.py --count $C --output "${FILE_PREFIX}_parallel_decode.json" > /dev/null 2>&1 &
        PID_D=$!

        wait $PID_P $PID_D
        echo "       Done."
    done
done

# ----------------------------------------------
# 阶段 3: 生成聚合报告
# ----------------------------------------------
echo ""
echo ">>> Generating Final Analysis Report..."
python analyze_results.py