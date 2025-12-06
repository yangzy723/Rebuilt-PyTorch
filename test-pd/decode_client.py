import asyncio
import aiohttp
import time
import json
import argparse
import sys
import random # 新增引用

# ================= 配置 =================
# 对应原脚本端口 30002
URL = "http://localhost:30002/generate"
PROMPT_LEN = 10
OUTPUT_LEN = 128

# 定义一个简单的词表
VOCAB = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "one", "two", "three", "four", "five", "six"
]
# =======================================

def generate_prompt(length):
    # 随机抽取 length 个单词
    words = random.choices(VOCAB, k=length)
    return " ".join(words)

async def send_request(session, req_id):
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
        print(f"[D-{req_id}] Request Failed: {e}", file=sys.stderr)
    
    latency = time.perf_counter() - start
    return latency if success else None

async def main(args):
    print(f"--- [Decode Worker] Starting {args.count} requests to {URL} ---")
    print(f"--- [Decode Worker] Prompt Length: {PROMPT_LEN} random words ---")
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(args.count)]
        
        start_wall = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_wall = time.perf_counter()

    valid_latencies = [r for r in results if r is not None]
    
    # 保存结果到 JSON
    output_data = {
        "type": "decode",
        "wall_time": end_wall - start_wall,
        "latencies": valid_latencies,
        "count": len(valid_latencies)
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f)

    print(f"--- [Decode Worker] Done. Avg Latency: {sum(valid_latencies)/len(valid_latencies):.4f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128, help="Number of requests")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()
    
    asyncio.run(main(args))