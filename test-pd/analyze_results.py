import json
import statistics
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置 (需与 run.sh 保持一致) =================
COUNTS = [16, 32, 64, 128, 256]
REPS = 3
RESULT_DIR = "results"
CSV_FILE = "experiment_summary.csv"
PLOT_FILE = "mps_benchmark_analysis.png"

# 图表字体设置 (如需中文支持可取消注释)
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False
# ==============================================================

def load_json(filepath):
    """加载 JSON 文件的辅助函数"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_avg_latency(data):
    """从结果数据中提取平均延迟"""
    if not data or 'latencies' not in data or not data['latencies']:
        return 0.0
    return statistics.mean(data['latencies'])

def collect_data():
    """遍历所有实验配置，收集并聚合数据"""
    aggregated_results = []

    print(f"{'Count':<8} | {'Speedup':<10} | {'Prefill Deg%':<15} | {'Decode Deg%':<15}")
    print("-" * 60)

    for c in COUNTS:
        # 临时列表，用于存储该 Count 下所有 Reps 的数据以计算平均值
        metrics = {
            "s_total": [], "p_total": [],
            "s_p_lat": [], "p_p_lat": [],
            "s_d_lat": [], "p_d_lat": []
        }

        valid_reps = 0
        for r in range(1, REPS + 1):
            prefix = f"{RESULT_DIR}/res_c{c}_r{r}"
            
            # 加载单次实验的 4 个文件
            s_p = load_json(f"{prefix}_serial_prefill.json") # Serial Prefill
            s_d = load_json(f"{prefix}_serial_decode.json")  # Serial Decode
            p_p = load_json(f"{prefix}_parallel_prefill.json") # Parallel Prefill
            p_d = load_json(f"{prefix}_parallel_decode.json")  # Parallel Decode

            if not all([s_p, s_d, p_p, p_d]):
                continue
            
            valid_reps += 1

            # 1. 计算总耗时 (Wall Time)
            # Serial 总耗时 = Prefill墙钟 + Decode墙钟
            s_total = s_p['wall_time'] + s_d['wall_time']
            # Parallel 总耗时 = Max(Prefill墙钟, Decode墙钟) (因为是并行)
            p_total = max(p_p['wall_time'], p_d['wall_time'])

            metrics["s_total"].append(s_total)
            metrics["p_total"].append(p_total)

            # 2. 计算延迟 (Latency)
            metrics["s_p_lat"].append(get_avg_latency(s_p))
            metrics["s_d_lat"].append(get_avg_latency(s_d))
            metrics["p_p_lat"].append(get_avg_latency(p_p))
            metrics["p_d_lat"].append(get_avg_latency(p_d))

        if valid_reps == 0:
            print(f"{c:<8} | No Data")
            continue

        # 计算该 Count 下的平均值
        avg_data = {k: statistics.mean(v) for k, v in metrics.items()}
        
        # 计算衍生指标
        speedup = avg_data["s_total"] / avg_data["p_total"] if avg_data["p_total"] > 0 else 0
        p_deg = ((avg_data["p_p_lat"] - avg_data["s_p_lat"]) / avg_data["s_p_lat"] * 100) if avg_data["s_p_lat"] > 0 else 0
        d_deg = ((avg_data["p_d_lat"] - avg_data["s_d_lat"]) / avg_data["s_d_lat"] * 100) if avg_data["s_d_lat"] > 0 else 0

        print(f"{c:<8} | {speedup:<10.2f} | {p_deg:<14.1f}% | {d_deg:<14.1f}%")

        aggregated_results.append({
            "count": c,
            "speedup": speedup,
            "s_total": avg_data["s_total"],
            "p_total": avg_data["p_total"],
            "s_p_lat": avg_data["s_p_lat"],
            "p_p_lat": avg_data["p_p_lat"],
            "s_d_lat": avg_data["s_d_lat"],
            "p_d_lat": avg_data["p_d_lat"]
        })

    return aggregated_results

def plot_visual_style(results):
    """使用 visual.py 的风格进行绘图"""
    if not results:
        print("没有数据可绘图。")
        return

    # 准备绘图数据
    counts_x = [r['count'] for r in results]
    batch_sizes = [f"{c} Req" for c in counts_x]
    x = np.arange(len(batch_sizes))

    time_serial = [r['s_total'] for r in results]
    time_parallel = [r['p_total'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    prefill_serial = [r['s_p_lat'] for r in results]
    prefill_parallel = [r['p_p_lat'] for r in results]
    
    decode_serial = [r['s_d_lat'] for r in results]
    decode_parallel = [r['p_d_lat'] for r in results]

    # 开始绘图
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle('PD Separation & MPS Benchmark (Avg of 3 Runs)', fontsize=20, weight='bold')

    # --- 子图 1: 总耗时对比与加速比 ---
    ax1 = fig.add_subplot(2, 2, 1)
    width = 0.35

    ax1.bar(x - width/2, time_serial, width, label='Serial Time (Avg)', color='#aec7e8', edgecolor='black')
    ax1.bar(x + width/2, time_parallel, width, label='Parallel Time (Avg)', color='#ffbb78', edgecolor='black')

    ax1.set_ylabel('Total Wall-Clock Time (s)', fontsize=12)
    ax1.set_title('Total Execution Time & Throughput Speedup', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, speedups, color='#d62728', marker='o', linewidth=2, label='Speedup (x)')
    for i, txt in enumerate(speedups):
        ax1_twin.annotate(f"{txt:.2f}x", (x[i], speedups[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#d62728', weight='bold')

    ax1_twin.set_ylabel('Speedup Factor', color='#d62728', fontsize=12)
    # 动态设置 Y 轴范围，让图更好看
    if speedups:
        ax1_twin.set_ylim(min(speedups)*0.9, max(speedups)*1.1)
    ax1_twin.tick_params(axis='y', labelcolor='#d62728')

    # --- 子图 2: Prefill 延迟对比 ---
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.bar(x - width/2, prefill_serial, width, label='Serial Prefill', color='#98df8a', edgecolor='black')
    ax2.bar(x + width/2, prefill_parallel, width, label='Parallel Prefill', color='#2ca02c', edgecolor='black')

    ax2.set_ylabel('Latency (s)', fontsize=12)
    ax2.set_title('Prefill Latency Comparison (Stability)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(x)):
        if prefill_serial[i] > 0:
            diff = ((prefill_parallel[i] - prefill_serial[i]) / prefill_serial[i]) * 100
            color = 'red' if diff > 1.0 else ('green' if diff < -1.0 else 'black')
            ax2.text(x[i], max(prefill_serial[i], prefill_parallel[i]), f"{diff:+.1f}%", ha='center', color=color, fontsize=10, weight='bold', va='bottom')

    # --- 子图 3: Decode 延迟对比 ---
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.bar(x - width/2, decode_serial, width, label='Serial Decode', color='#c5b0d5', edgecolor='black')
    ax3.bar(x + width/2, decode_parallel, width, label='Parallel Decode', color='#9467bd', edgecolor='black')

    ax3.set_ylabel('Latency (s)', fontsize=12)
    ax3.set_title('Decode Latency Comparison (Interference)', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(x)):
        if decode_serial[i] > 0:
            diff = ((decode_parallel[i] - decode_serial[i]) / decode_serial[i]) * 100
            ax3.text(x[i], max(decode_serial[i], decode_parallel[i]), f"{diff:+.1f}%", ha='center', color='red', fontsize=10, weight='bold', va='bottom')

    # --- 子图 4: 文本总结 ---
    ax4 = fig.add_subplot(2, 2, 2)
    ax4.axis('off')

    avg_speedup = np.mean(speedups) if speedups else 0
    
    # 计算平均 Decode 劣化
    d_degs = [((dp - ds)/ds)*100 for dp, ds in zip(decode_parallel, decode_serial) if ds > 0]
    decode_deg_avg = np.mean(d_degs) if d_degs else 0

    summary_text = (
        "Benchmark Analysis Summary:\n\n"
        "1. Throughput Speedup:\n"
        f"   Avg Speedup: {avg_speedup:.2f}x\n"
        f"   (Max: {max(speedups):.2f}x, Min: {min(speedups):.2f}x)\n"
        "   Higher is better.\n\n"
        "2. Prefill Stability:\n"
        "   Compares isolated vs colocated latency.\n"
        "   Small % diff means good isolation.\n\n"
        "3. Decode Cost:\n"
        f"   Avg Degradation: {decode_deg_avg:.1f}%\n"
        "   Latency increase due to resource contention."
    )

    ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center', family='monospace', bbox=dict(facecolor='#f0f0f0', alpha=0.8, boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n[绘图] 高级分析图表已保存至: {PLOT_FILE}")

def main():
    # 1. 收集数据
    print("正在分析 run.sh 生成的数据...")
    results = collect_data()

    # 2. 保存 CSV
    if results:
        keys = results[0].keys()
        try:
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[保存] 汇总数据已保存至: {CSV_FILE}")
        except IOError as e:
            print(f"保存 CSV 失败: {e}")
    
    # 3. 绘图 (Visual 风格)
    plot_visual_style(results)

if __name__ == "__main__":
    main()