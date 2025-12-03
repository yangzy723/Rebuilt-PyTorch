import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style (removed Chinese font settings to rely on system defaults)
sns.set_theme(style="whitegrid")

# === 1. Data Preparation ===
raw_data = [
    # --- w/o intercept ---
    {'Setting': 'w/o intercept', 'Phase': 'Prefill', 'Metric': 'Throughput', 'Value': np.mean([24742.47, 40157.75, 23580.34, 40108.51])},
    {'Setting': 'w/o intercept', 'Phase': 'Prefill', 'Metric': 'Latency', 'Value': np.mean([1.32436, 0.81598, 1.38963, 0.81698])},
    {'Setting': 'w/o intercept', 'Phase': 'Decode', 'Metric': 'Throughput', 'Value': np.mean([6584.83, 6642.71, 6418.71, 6431.49])},
    {'Setting': 'w/o intercept', 'Phase': 'Decode', 'Metric': 'Latency', 'Value': np.mean([0.00972, 0.00963, 0.00997, 0.00995])},

    # --- w/ intercept sglang side ---
    {'Setting': 'w/ sglang', 'Phase': 'Prefill', 'Metric': 'Throughput', 'Value': np.mean([23290.35, 39796.71, 23813.00, 39907.96])},
    {'Setting': 'w/ sglang', 'Phase': 'Prefill', 'Metric': 'Latency', 'Value': np.mean([1.40693, 0.82338, 1.37606, 0.82109])},
    {'Setting': 'w/ sglang', 'Phase': 'Decode', 'Metric': 'Throughput', 'Value': np.mean([3325.70, 3352.28, 3367.37, 3409.57])},
    {'Setting': 'w/ sglang', 'Phase': 'Decode', 'Metric': 'Latency', 'Value': np.mean([0.01924, 0.01909, 0.01901, 0.01877])},

    # --- w/ intercept pytorch side ---
    {'Setting': 'w/ pytorch', 'Phase': 'Prefill', 'Metric': 'Throughput', 'Value': np.mean([23290.35, 40054.09, 24134.03, 40171.46])},
    {'Setting': 'w/ pytorch', 'Phase': 'Prefill', 'Metric': 'Latency', 'Value': np.mean([1.37631, 0.81809, 1.35775, 0.81570])},
    {'Setting': 'w/ pytorch', 'Phase': 'Decode', 'Metric': 'Throughput', 'Value': np.mean([3055.38, 3069.32, 2987.69, 3007.94])},
    {'Setting': 'w/ pytorch', 'Phase': 'Decode', 'Metric': 'Latency', 'Value': np.mean([0.02095, 0.02085, 0.02142, 0.02128])},

    # --- w/ intercept all sides ---
    {'Setting': 'w/ all sides', 'Phase': 'Prefill', 'Metric': 'Throughput', 'Value': np.mean([23135.48, 39820.94, 23679.73, 39542.21])},
    {'Setting': 'w/ all sides', 'Phase': 'Prefill', 'Metric': 'Latency', 'Value': np.mean([1.41635, 0.82288, 1.38380, 0.82868])},
    {'Setting': 'w/ all sides', 'Phase': 'Decode', 'Metric': 'Throughput', 'Value': np.mean([1981.21, 2039.97, 2094.90, 2076.57])},
    {'Setting': 'w/ all sides', 'Phase': 'Decode', 'Metric': 'Latency', 'Value': np.mean([0.03230, 0.03137, 0.03055, 0.03082])},
]

df = pd.DataFrame(raw_data)

# === 2. Plotting ===
# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Comparison w/ Different Intercept Settings', fontsize=16, y=1.02)

# Define configurations
metrics_config = {
    ('Prefill', 'Throughput'): {'ax': axes[0, 0], 'color': 'skyblue', 'fmt': '{:.0f}', 'ylabel': 'Token/s'},
    ('Prefill', 'Latency'):    {'ax': axes[0, 1], 'color': 'salmon', 'fmt': '{:.3f}', 'ylabel': 'Seconds'},
    ('Decode', 'Throughput'):  {'ax': axes[1, 0], 'color': 'dodgerblue', 'fmt': '{:.0f}', 'ylabel': 'Token/s'},
    ('Decode', 'Latency'):     {'ax': axes[1, 1], 'color': 'firebrick', 'fmt': '{:.3f}', 'ylabel': 'Seconds'},
}

# Loop to draw plots
for (phase, metric), config in metrics_config.items():
    ax = config['ax']
    subset = df[(df['Phase'] == phase) & (df['Metric'] == metric)]
    
    # Draw barplot
    sns.barplot(data=subset, x='Setting', y='Value', ax=ax, color=config['color'], edgecolor='black')
    
    # Set titles and labels (English)
    title_suffix = "(Higher is Better)" if metric == 'Throughput' else "(Lower is Better)"
    ax.set_title(f"{phase} - {metric}\n{title_suffix}", fontsize=12, fontweight='bold')
    ax.set_ylabel(config['ylabel'])
    ax.set_xlabel('') 
    
    # Add value annotations
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(config['fmt'].format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()

# Save the figure instead of showing it (better for headless servers)
output_filename = 'performance_comparison.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {output_filename}")