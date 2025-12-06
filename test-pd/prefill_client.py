import asyncio
import aiohttp
import time
import json
import argparse
import sys
import random  # 新增引用

# ================= 配置 =================
# 对应原脚本端口 30001
URL = "http://localhost:30001/generate" 
PROMPT_LEN = 1024
OUTPUT_LEN = 1

# 定义一个简单的词表用于生成随机 Prompt
VOCAB = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry",
    "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xigua", "yam", "zucchini",
    "cat", "dog", "elephant", "fish", "giraffe", "horse", "iguana", "jellyfish", 
    "kangaroo", "lion", "monkey", "newt", "octopus", "penguin", "quail", "rabbit",
    "snake", "tiger", "urchin", "vulture", "whale", "xray", "yak", "zebra",
    "computation", "parallel", "graphics", "processing", "unit", "memory", "bandwidth",
    "latency", "throughput", "interference", "benchmark", "analysis", "system", "kernel"
]
# =======================================

def generate_prompt(length):
    # 随机从词表中抽取 length 个单词，并用空格连接
    # 这能模拟更真实的 Token 分布，并防止 Cache 命中
    words = random.choices(VOCAB, k=length)
    return " ".join(words)

async def send_request(session, req_id):
    # 每次调用都会生成一个新的随机 Prompt
    prompt = generate_prompt(PROMPT_LEN)
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": OUTPUT_LEN,
            "ignore_eos": True
        }
    }
    
    start = time.perf_counter()
    success = False
    try:
        async with session.post(URL, json=payload) as response:
            await response.read()
            success = True
    except Exception as e:
        print(f"[P-{req_id}] Request Failed: {e}", file=sys.stderr)
    
    latency = time.perf_counter() - start
    return latency if success else None

async def main(args):
    print(f"--- [Prefill Worker] Starting {args.count} requests to {URL} ---")
    print(f"--- [Prefill Worker] Prompt Length: {PROMPT_LEN} random words ---")
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(args.count)]
        
        start_wall = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_wall = time.perf_counter()

    valid_latencies = [r for r in results if r is not None]
    
    # 保存结果到 JSON
    output_data = {
        "type": "prefill",
        "wall_time": end_wall - start_wall,
        "latencies": valid_latencies,
        "count": len(valid_latencies)
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f)
        
    print(f"--- [Prefill Worker] Done. Avg Latency: {sum(valid_latencies)/len(valid_latencies):.4f}s ---")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128, help="Number of requests")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()
    
    asyncio.run(main(args))