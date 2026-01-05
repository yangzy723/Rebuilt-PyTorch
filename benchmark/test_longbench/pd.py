import asyncio
import aiohttp
import pandas as pd
import time
import secrets
import random
import json
import numpy as np
import sys
from datetime import datetime, timedelta

# ================= CONFIGURATION AREA =================

# Address of the two MPS instances
URL_PREFILL = "http://127.0.0.1:30001/v1/completions" # Prefill Node
URL_DECODE  = "http://127.0.0.1:30002/v1/completions" # Decode Node

# --- LONGBENCH SIMULATION CONFIG ---
# 模拟 LongBench 各子集的长度分布特征 (Mean, StdDev)
LONGBENCH_STATS = {
    # NarrativeQA: 超长上下文 (Avg ~18k)
    "narrativeqa": {"input_mean": 18000, "input_std": 4000, "output_mean": 50, "output_std": 20},
    # Qasper: 中长上下文 (Avg ~3.5k)
    "qasper":      {"input_mean": 3500,  "input_std": 1000, "output_mean": 50, "output_std": 20},
    # HotpotQA: 多文档问答 (Avg ~10k)
    "hotpotqa":    {"input_mean": 10000, "input_std": 2500, "output_mean": 40, "output_std": 10}
}

# 选择要模拟的子集
LONGBENCH_SUBSET = "narrativeqa"
READ_LIMIT = 200      # 模拟生成的请求数量

# --- TRAFFIC GENERATION ---
TARGET_QPS = 2.0      # 每秒模拟发起的请求数

# --- CONTROL PARAMETERS ---
SAMPLE_INTERVAL = 1     # Sampling
SPEEDUP_FACTOR = 1.0    # Time Compression

MAX_REQUESTS = 1000     # Total trace rows to process

# --- SLO CONFIGURATION  ---
SLO_TTFT = 30       # Seconds (Time To First Token 阈值, 仅适用于 Prefill Worker)
SLO_TPOT = 0.2      # Seconds (Time Per Output Token 阈值, 仅适用于 Decode Worker)

# ===================================================

VOCAB = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "system", "model", "inference", "performance", "latency", "throughput", "gpu",
    "compute", "memory", "cache", "token", "context", "decode", "prefill", "batch",
    "queue", "request", "server", "client", "python", "async", "await", "test"
]

def get_random_prompt_words(token_count):
    """
    构造指定长度的 Prompt。
    """
    prefix = f"REQ_{secrets.token_hex(3)}: "
    num_words = int(token_count)
    if num_words <= 0: return prefix

    # 优化: 随机生成 100 个词，然后重复拼接
    base_chunk_size = 100
    base_words = random.choices(VOCAB, k=min(num_words, base_chunk_size))
    base_str = " ".join(base_words)

    repeats = num_words // base_chunk_size
    remainder = num_words % base_chunk_size

    final_str = (base_str + " ") * repeats
    if remainder > 0:
        final_str += " ".join(base_words[:remainder])

    return prefix + final_str[:len(final_str)-1] # 去掉最后一个空格

async def send_request(session, row, start_timestamp_ref, results_list, role):
    # role: "prefill_worker" or "decode_worker"

    target_time_offset = row['relative_timestamp'] / SPEEDUP_FACTOR

    # === Differentiate Input/Output ===
    if role == "prefill_worker":
        target_url = URL_PREFILL
        # Prefill Worker: Compute Bound
        prompt_text = get_random_prompt_words(row['ContextTokens'])
        req_max_tokens = 1
        recorded_input_len = int(row['ContextTokens'])
    else:
        target_url = URL_DECODE
        # Decode Worker: Memory Bound
        prompt_text = get_random_prompt_words(5)
        req_max_tokens = int(row['GeneratedTokens'])
        recorded_input_len = 5

    payload = {
        "model": "/data/datasets/models-hf/Llama-3.1-8B-Instruct/",
        "prompt": prompt_text,
        "max_tokens": req_max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True
    }

    # Wait for launch
    now_offset = time.time() - start_timestamp_ref
    wait_seconds = target_time_offset - now_offset
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    req_start = time.time()
    ttft = 0.0
    first_token_time = 0.0
    last_token_time = 0.0
    token_count = 0
    token_intervals = []

    try:
        async with session.post(target_url, json=payload) as response:
            if response.status != 200:
                await response.read()
                return

            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == 'data: [DONE]': continue

                if line.startswith('data: '):
                    try:
                        chunk_json = json.loads(line[6:])
                        choices = chunk_json.get("choices", [])
                        if not choices: continue
                        delta_content = choices[0].get("text", "")

                        if delta_content:
                            current_time = time.time()
                            token_count += 1

                            if first_token_time == 0.0:
                                first_token_time = current_time
                                last_token_time = current_time
                                ttft = first_token_time - req_start
                            else:
                                interval = current_time - last_token_time
                                token_intervals.append(interval)
                                last_token_time = current_time
                    except:
                        continue

            req_end = time.time()
            total_latency = req_end - req_start

            # --- SLO Check Logic ---
            ttft_violated = False
            bad_token_intervals_count = 0

            if role == "prefill_worker":
                # Prefill 关注 TTFT
                if ttft > SLO_TTFT:
                    ttft_violated = True
            elif role == "decode_worker":
                # Decode 关注 TPOT
                bad_token_intervals_count = sum(1 for t in token_intervals if t > SLO_TPOT)

            results_list.append({
                "role": role,
                "input_len": recorded_input_len,
                "output_len": token_count,
                "latency": total_latency,
                "ttft": ttft,
                "token_intervals": token_intervals,
                "status": response.status,
                "ttft_violated": ttft_violated,
                "bad_token_cnt": bad_token_intervals_count
            })

    except Exception as e:
        pass

def generate_simulated_longbench_data():
    """
    不依赖 datasets 库，直接通过统计分布生成模拟数据。
    """
    print(f"Simulating LongBench subset: '{LONGBENCH_SUBSET}' (No download needed)...")

    # 获取统计参数，如果找不到则默认使用 qasper
    stats = LONGBENCH_STATS.get(LONGBENCH_SUBSET, LONGBENCH_STATS["qasper"])

    print(f"Stats: Input Mean={stats['input_mean']}, Std={stats['input_std']}")

    data_list = []
    current_synthetic_time = pd.Timestamp.now()

    # 批量生成长度数据 (正态分布)
    input_lens = np.random.normal(stats["input_mean"], stats["input_std"], READ_LIMIT)
    input_lens = np.maximum(input_lens, 100).astype(int) # 最小 100 token

    output_lens = np.random.normal(stats["output_mean"], stats["output_std"], READ_LIMIT)
    output_lens = np.maximum(output_lens, 10).astype(int) # 最小 10 token

    # 批量生成到达间隔 (指数分布)
    inter_arrivals = np.random.exponential(1.0 / TARGET_QPS, READ_LIMIT)

    for i in range(READ_LIMIT):
        current_synthetic_time += pd.Timedelta(seconds=inter_arrivals[i])

        data_list.append({
            "TIMESTAMP": current_synthetic_time,
            "ContextTokens": input_lens[i],
            "GeneratedTokens": output_lens[i],
            "original_id": i
        })

    return pd.DataFrame(data_list)

async def main():
    print(f"\n=== Generating Simulation Data (LongBench: {LONGBENCH_SUBSET}) ===")
    df = generate_simulated_longbench_data()

    df = df.sort_values('TIMESTAMP')

    if SAMPLE_INTERVAL > 1:
        print(f"Applying sampling: picking 1 request every {SAMPLE_INTERVAL} rows.")
        df = df.iloc[::SAMPLE_INTERVAL].copy()

    if MAX_REQUESTS and MAX_REQUESTS < len(df):
        print(f"Limiting execution to first {MAX_REQUESTS} requests.")
        df = df.head(MAX_REQUESTS)

    if df.empty:
        print("Error: No data generated.")
        return

    start_time_base = df['TIMESTAMP'].iloc[0]
    df['relative_timestamp'] = (df['TIMESTAMP'] - start_time_base).dt.total_seconds()

    total_trace_rows = len(df)
    total_http_requests = total_trace_rows * 2

    avg_input_len = df['ContextTokens'].mean()

    print(f"\n=== PD Separation Simulation (MPS + Simulated LongBench) ===")
    print(f"Prefill Node : {URL_PREFILL}")
    print(f"Decode Node  : {URL_DECODE}")
    print(f"Simulated Set: {LONGBENCH_SUBSET}")
    print(f"Avg Context  : {avg_input_len:.0f} tokens")
    print(f"Target QPS   : {TARGET_QPS}")
    print(f"Total Req    : {total_http_requests} (Prefill + Decode)")

    results = []
    connector = aiohttp.TCPConnector(limit=4000)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        benchmark_start_time = time.time()
        print("Starting requests...")

        for _, row in df.iterrows():
            tasks.append(asyncio.create_task(
                send_request(session, row, benchmark_start_time, results, "prefill_worker")
            ))
            tasks.append(asyncio.create_task(
                send_request(session, row, benchmark_start_time, results, "decode_worker")
            ))

        completed_count = 0
        for future in asyncio.as_completed(tasks):
            await future
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_http_requests:
                percent = (completed_count / total_http_requests) * 100
                sys.stdout.write(f"\r[Progress]: {completed_count}/{total_http_requests} ({percent:.1f}%) requests finished.")
                sys.stdout.flush()

        benchmark_end_time = time.time()
        benchmark_duration = benchmark_end_time - benchmark_start_time
        print(f"\nAll tasks completed in {benchmark_duration:.2f} seconds.")

    # --- Statistics ---
    res_df = pd.DataFrame(results)

    # Save Results
    filename = f"result_longbench_{LONGBENCH_SUBSET}_sim.csv"
    res_df.drop(columns=['token_intervals'], errors='ignore').to_csv(filename, index=False)
    print(f"\nData saved to {filename}")

    print(f"\n" + "="*50)
    print(f"  RESULTS (LongBench: {LONGBENCH_SUBSET})")
    print(f"="*50)

    if res_df.empty:
        print("No valid responses.")
        return

    valid_df = res_df[res_df['status'] == 200]
    count = len(valid_df)

    if count == 0:
        print("No successful requests (200 OK).")
        return

    # --- System-wide Throughput ---
    total_input_tokens = valid_df['input_len'].sum()
    total_output_tokens = valid_df['output_len'].sum()

    rps = count / benchmark_duration
    prefill_tps = total_input_tokens / benchmark_duration
    decode_tps = total_output_tokens / benchmark_duration

    print(f"Total Successful Requests: {count}")
    print(f"Benchmark Duration     : {benchmark_duration:.2f} s")
    print(f"Total Input Tokens     : {total_input_tokens}")
    print(f"Total Output Tokens    : {total_output_tokens}")

    print(f"\n[System Throughput]")
    print(f"  Total Requests/s     : {rps:.2f} req/s")
    print(f"  Prefill Tokens/s     : {prefill_tps:.2f} tokens/s")
    print(f"  Decode Tokens/s      : {decode_tps:.2f} tokens/s")

    # --- Role Based Analysis ---
    prefill_df = valid_df[valid_df['role'] == 'prefill_worker']
    decode_df  = valid_df[valid_df['role'] == 'decode_worker']

    # 1. Prefill Worker Stats (Focus on TTFT / Latency)
    print(f"\n[Prefill Worker Stats] (Compute Bound -> TTFT)")
    if not prefill_df.empty:
        # TTFT Stats
        p50_ttft = prefill_df['ttft'].quantile(0.50)
        p90_ttft = prefill_df['ttft'].quantile(0.90)
        p99_ttft = prefill_df['ttft'].quantile(0.99)
        avg_ttft = prefill_df['ttft'].mean()

        # SLO Violation (Request Level)
        ttft_violation_count = prefill_df['ttft_violated'].sum()
        ttft_violation_rate = (ttft_violation_count / len(prefill_df)) * 100

        print(f"  Requests Count       : {len(prefill_df)}")
        print(f"  Avg TTFT             : {avg_ttft:.4f} s")
        print(f"  P50 TTFT             : {p50_ttft:.4f} s")
        print(f"  P90 TTFT             : {p90_ttft:.4f} s")
        print(f"  P99 TTFT             : {p99_ttft:.4f} s")
        print(f"  SLO Violation Rate   : {ttft_violation_rate:.2f}% (> {SLO_TTFT}s)")
    else:
        print("  No prefill requests recorded.")

    # 2. Decode Worker Stats (Focus on TPOT)
    print(f"\n[Decode Worker Stats] (Memory Bound -> TPOT)")
    if not decode_df.empty:
        # Collect all intervals for Global TPOT calculation
        all_intervals = []
        for intervals in decode_df['token_intervals']:
            all_intervals.extend(intervals)

        print(f"  Requests Count       : {len(decode_df)}")

        if all_intervals:
            all_intervals_array = np.array(all_intervals)
            all_intervals_ms = all_intervals_array * 1000  # Convert to ms

            avg_tpot = np.mean(all_intervals_ms)
            p50_tpot = np.percentile(all_intervals_ms, 50)
            p90_tpot = np.percentile(all_intervals_ms, 90)
            p99_tpot = np.percentile(all_intervals_ms, 99)

            # SLO Violation (Token Level)
            total_intervals_count = len(all_intervals_array)
            total_bad_intervals = np.sum(all_intervals_array > SLO_TPOT)
            tpot_violation_rate = (total_bad_intervals / total_intervals_count) * 100

            print(f"  Total Token Intervals: {total_intervals_count}")
            print(f"  Avg TPOT             : {avg_tpot:.2f} ms")
            print(f"  P50 TPOT             : {p50_tpot:.2f} ms")
            print(f"  P90 TPOT             : {p90_tpot:.2f} ms")
            print(f"  P99 TPOT             : {p99_tpot:.2f} ms")
            print(f"  SLO Violation Rate   : {tpot_violation_rate:.2f}% (> {SLO_TPOT}s)")
        else:
            print("  No output tokens generated (TPOT N/A).")
    else:
        print("  No decode requests recorded.")

if __name__ == "__main__":
    asyncio.run(main())