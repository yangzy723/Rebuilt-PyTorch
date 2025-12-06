import asyncio
import aiohttp
import time
import json
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# ================= 配置区域 =================
DEFAULT_PREFILL_URL = "http://localhost:30001/generate"
DEFAULT_DECODE_URL = "http://localhost:30002/generate"

DECODE_OUTPUT_LEN = 256
DECODE_PROMPT_LEN = 10 

@dataclass
class Metric:
    avg: float
    p99: float

@dataclass
class ExperimentResult:
    prefill_len: int
    mode: str  # "separate" or "colocate"
    ttft: Optional[Metric] = None
    tpot: Optional[Metric] = None

# 简单的词表用于生成随机 Prompt，避免 KV Cache 命中
VOCAB = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry",
    "strawberry", "tangerine", "ugli", "fruit", "watermelon", "sky", "blue", "red",
    "green", "yellow", "purple", "ocean", "river", "mountain", "forest", "tree",
    "flower", "sun", "moon", "star", "planet", "galaxy", "universe", "physics",
    "math", "code", "python", "java", "cpp", "algorithm", "data", "structure",
    "network", "server", "client", "request", "response", "latency", "throughput"
]

def generate_prompt(length):
    """
    生成指定长度（大约对应 token 数量）的随机 Prompt。
    使用随机单词组合，确保每次请求的内容不同，从而通过 Cache Miss 测出真实的 Prefill 性能。
    """
    # 随机选择 length 个单词并用空格连接
    # length 参数在这里近似代表 token 数量 (假设 1 word ≈ 1 token)
    return " ".join(random.choices(VOCAB, k=length))

async def send_request_stream(session, url, req_type, input_len, output_len, req_id) -> Tuple[float, List[float]]:
    """
    流式请求发送函数。
    返回: (ttft, tpot_list)
    - ttft: 首字延迟 (ms)
    - tpot_list: 后续每个 token 的间隔延迟列表 (ms)
    """
    prompt = generate_prompt(input_len)
    payload = {
        "text": prompt,
        "stream": True,  # 开启流式
        "sampling_params": {
            "max_new_tokens": output_len,
            "ignore_eos": True
        }
    }
    
    start_time = time.perf_counter()
    ttft = None
    tpots = []
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"[{req_type}-{req_id}] Error: Status {response.status}", file=sys.stderr)
                return None, []
            
            # 记录上一次收到数据的时间
            last_chunk_time = None
            
            # 使用 iter_any 读取流式数据
            async for _ in response.content.iter_any():
                now = time.perf_counter()
                
                if last_chunk_time is None:
                    # 收到第一个 chunk -> TTFT
                    ttft = (now - start_time) * 1000
                else:
                    # 收到后续 chunk -> TPOT (Inter-token Latency)
                    inter_token_latency = (now - last_chunk_time) * 1000
                    tpots.append(inter_token_latency)
                
                last_chunk_time = now
                
    except Exception as e:
        print(f"[{req_type}-{req_id}] Request Failed: {e}", file=sys.stderr)
        return None, []

    if ttft is None:
        return None, []

    return ttft, tpots

async def run_batch(url, req_type, count, input_len, output_len):
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request_stream(session, url, req_type, input_len, output_len, i) 
            for i in range(count)
        ]
        results = await asyncio.gather(*tasks)
    
    # results 是一个 list of (ttft, tpot_list)
    ttft_list = []
    all_tpots = []
    
    for r in results:
        if r and r[0] is not None:
            ttft_list.append(r[0])
            all_tpots.extend(r[1]) 
            
    return ttft_list, all_tpots

def calculate_metric(data_points):
    """
    计算统计指标。
    data_points: 可以是所有请求的 TTFT 列表，也可以是所有请求的所有 Token 间隔列表。
    """
    if not data_points:
        return Metric(0, 0)
    
    return Metric(
        avg=np.mean(data_points),
        p99=np.percentile(data_points, 99)
    )

async def main(args):
    # 压力测试长度设置
    prefill_lengths = [128, 256, 512, 1024, 2048, 4096]
    results_db = []

    print(f"=== Starting Benchmark (Streaming Mode | Random Prompts) ===")
    print(f"Batch Size: {args.batch_size}")
    print(f"Repetitions: {args.repeat}")
    print(f"Total Requests per Config: {args.batch_size * args.repeat}")
    print(f"URLs -> P: {args.url_prefill}, D: {args.url_decode}")

    for p_len in prefill_lengths:
        print(f"\n>>> Testing Prefill Length: {p_len} (Randomized)")
        
        # 汇总所有轮次的数据
        sep_p_ttfts = [] 
        sep_d_tpots = []
        col_p_ttfts = []
        col_d_tpots = []

        for r in range(args.repeat):
            print(f"  [Round {r+1}/{args.repeat}]")

            # -------------------------------------------------
            # 1. 单独测试 Prefill (Separate P)
            # -------------------------------------------------
            p_ttfts, _ = await run_batch(args.url_prefill, "prefill", args.batch_size, p_len, 1)
            sep_p_ttfts.extend(p_ttfts)

            # -------------------------------------------------
            # 2. 单独测试 Decode (Separate D)
            # -------------------------------------------------
            _, d_tpots = await run_batch(args.url_decode, "decode", args.batch_size, DECODE_PROMPT_LEN, DECODE_OUTPUT_LEN)
            sep_d_tpots.extend(d_tpots)

            # -------------------------------------------------
            # 3. Colocate 测试 (同时发送)
            # -------------------------------------------------
            async with aiohttp.ClientSession() as session:
                task_p = [send_request_stream(session, args.url_prefill, "prefill", p_len, 1, i) for i in range(args.batch_size)]
                task_d = [send_request_stream(session, args.url_decode, "decode", DECODE_PROMPT_LEN, DECODE_OUTPUT_LEN, i) for i in range(args.batch_size)]
                
                all_res = await asyncio.gather(*(task_p + task_d))
                
                # 拆分结果
                for res in all_res[:args.batch_size]:
                    if res and res[0] is not None:
                        col_p_ttfts.append(res[0])
                
                for res in all_res[args.batch_size:]:
                    if res and res[0] is not None:
                        col_d_tpots.extend(res[1])
            
            print(f"    Batch Done.")
            await asyncio.sleep(1)

        # --- 汇总与计算 ---
        results_db.append(ExperimentResult(
            prefill_len=p_len,
            mode="separate",
            ttft=calculate_metric(sep_p_ttfts),
            tpot=calculate_metric(sep_d_tpots)
        ))
        
        results_db.append(ExperimentResult(
            prefill_len=p_len,
            mode="colocate",
            ttft=calculate_metric(col_p_ttfts),
            tpot=calculate_metric(col_d_tpots)
        ))

    with open(args.output_json, 'w') as f:
        json_data = [asdict(r) for r in results_db]
        json.dump(json_data, f, indent=2)
    print(f"\nData saved to {args.output_json}")

    plot_results(results_db, args.output_img)

def plot_results(data: List[ExperimentResult], filename: str):
    lengths = sorted(list(set(d.prefill_len for d in data)))
    x = np.arange(len(lengths))
    width = 0.35

    # 提取数据
    sep_ttft_p99 = [next(d.ttft.p99 for d in data if d.prefill_len == l and d.mode == "separate") for l in lengths]
    col_ttft_p99 = [next(d.ttft.p99 for d in data if d.prefill_len == l and d.mode == "colocate") for l in lengths]
    
    sep_ttft_avg = [next(d.ttft.avg for d in data if d.prefill_len == l and d.mode == "separate") for l in lengths]
    col_ttft_avg = [next(d.ttft.avg for d in data if d.prefill_len == l and d.mode == "colocate") for l in lengths]

    sep_tpot_p99 = [next(d.tpot.p99 for d in data if d.prefill_len == l and d.mode == "separate") for l in lengths]
    col_tpot_p99 = [next(d.tpot.p99 for d in data if d.prefill_len == l and d.mode == "colocate") for l in lengths]

    sep_tpot_avg = [next(d.tpot.avg for d in data if d.prefill_len == l and d.mode == "separate") for l in lengths]
    col_tpot_avg = [next(d.tpot.avg for d in data if d.prefill_len == l and d.mode == "colocate") for l in lengths]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Performance (Streaming Mode | Random Prompts): Separate vs Colocate', fontsize=16)

    def plot_bars(ax, data1, data2, title, ylabel):
        ax.bar(x - width/2, data1, width, label='Separate')
        ax.bar(x + width/2, data2, width, label='Colocate')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Prefill Length')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 1. TTFT Avg
    plot_bars(axes[0, 0], sep_ttft_avg, col_ttft_avg, 'Prefill TTFT (Avg)', 'Latency (ms)')
    # 2. TTFT P99
    plot_bars(axes[0, 1], sep_ttft_p99, col_ttft_p99, 'Prefill TTFT (P99)', 'Latency (ms)')
    # 3. TPOT Avg (Inter-token)
    plot_bars(axes[1, 0], sep_tpot_avg, col_tpot_avg, 'Decode Inter-token Latency (Avg)', 'Latency (ms)')
    # 4. TPOT P99 (Inter-token)
    plot_bars(axes[1, 1], sep_tpot_p99, col_tpot_p99, 'Decode Inter-token Latency (P99)', 'Latency (ms)')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Chart saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-prefill", type=str, default=DEFAULT_PREFILL_URL, help="URL for separate prefill server")
    parser.add_argument("--url-decode", type=str, default=DEFAULT_DECODE_URL, help="URL for separate decode server")
    # 64 个用户并发
    parser.add_argument("--batch-size", type=int, default=64, help="Requests per batch")
    parser.add_argument("--repeat", type=int, default=9, help="Number of repetitions")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")
    parser.add_argument("--output-img", type=str, default="benchmark_results.png")
    
    args = parser.parse_args()
    asyncio.run(main(args))