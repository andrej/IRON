#!/usr/bin/env python3
"""
Plot performance metrics from results.csv showing evolution across commits.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Read the CSV file
df = pd.read_csv('results.csv')

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create short commit hashes for x-axis labels
df['Short Commit'] = df['Commit'].str[:7]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LLaMA 3.2 1B Performance Evolution Across Commits', fontsize=16, fontweight='bold')

# Plot 1: Tokens Per Second (TPS) - Higher is better
ax1 = axes[0, 0]
ax1.plot(range(len(df)), df['TPS (mean)'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.fill_between(range(len(df)), 
                  df['TPS (mean)'] - df['TPS (stddev)'], 
                  df['TPS (mean)'] + df['TPS (stddev)'], 
                  alpha=0.3, color='#2E86AB')
ax1.set_ylabel('Tokens Per Second (TPS)', fontsize=11, fontweight='bold')
ax1.set_title('Throughput Performance', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(df['Short Commit'], rotation=45, ha='right', fontsize=9)

# Plot 2: Time to First Token (TTFT) - Lower is better
ax2 = axes[0, 1]
ax2.plot(range(len(df)), df['TTFT (mean)'], marker='s', linewidth=2, markersize=8, color='#A23B72')
ax2.fill_between(range(len(df)), 
                  df['TTFT (mean)'] - df['TTFT (stddev)'], 
                  df['TTFT (mean)'] + df['TTFT (stddev)'], 
                  alpha=0.3, color='#A23B72')
ax2.set_ylabel('Time to First Token (s)', fontsize=11, fontweight='bold')
ax2.set_title('Latency Performance', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df['Short Commit'], rotation=45, ha='right', fontsize=9)

# Plot 3: Total Time - Lower is better
ax3 = axes[1, 0]
ax3.plot(range(len(df)), df['Total (mean)'], marker='^', linewidth=2, markersize=8, color='#F18F01')
ax3.fill_between(range(len(df)), 
                  df['Total (mean)'] - df['Total (stddev)'], 
                  df['Total (mean)'] + df['Total (stddev)'], 
                  alpha=0.3, color='#F18F01')
ax3.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
ax3.set_title('End-to-End Execution Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Commits (chronological order)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels(df['Short Commit'], rotation=45, ha='right', fontsize=9)

# Plot 4: Combined efficiency view - TPS vs TTFT
ax4 = axes[1, 1]
scatter = ax4.scatter(df['TTFT (mean)'], df['TPS (mean)'], 
                     c=range(len(df)), s=150, cmap='viridis', 
                     edgecolors='black', linewidth=1.5, alpha=0.7)
# Add arrows showing progression
for i in range(len(df)-1):
    ax4.annotate('', xy=(df['TTFT (mean)'].iloc[i+1], df['TPS (mean)'].iloc[i+1]),
                xytext=(df['TTFT (mean)'].iloc[i], df['TPS (mean)'].iloc[i]),
                arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))
ax4.set_xlabel('Time to First Token (s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Tokens Per Second', fontsize=11, fontweight='bold')
ax4.set_title('Latency vs Throughput Trade-off', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
# Add colorbar to show commit progression
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Commit Order (older → newer)', fontsize=10)

# Add annotations for first and last commits
ax4.annotate('Start', (df['TTFT (mean)'].iloc[0], df['TPS (mean)'].iloc[0]),
            xytext=(10, 10), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
ax4.annotate('Latest', (df['TTFT (mean)'].iloc[-1], df['TPS (mean)'].iloc[-1]),
            xytext=(10, 10), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.5))

plt.tight_layout()

# Save the plot
plt.savefig('performance_evolution.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'performance_evolution.png'")

# Display the plot
plt.show()

# Print summary statistics
print("\n=== Performance Summary ===")
print(f"\nFirst commit ({df['Short Commit'].iloc[0]}) on {df['Date'].iloc[0].strftime('%Y-%m-%d')}:")
print(f"  TPS: {df['TPS (mean)'].iloc[0]:.3f} tokens/s")
print(f"  TTFT: {df['TTFT (mean)'].iloc[0]:.3f} s")
print(f"  Total Time: {df['Total (mean)'].iloc[0]:.3f} s")

print(f"\nLatest commit ({df['Short Commit'].iloc[-1]}) on {df['Date'].iloc[-1].strftime('%Y-%m-%d')}:")
print(f"  TPS: {df['TPS (mean)'].iloc[-1]:.3f} tokens/s")
print(f"  TTFT: {df['TTFT (mean)'].iloc[-1]:.3f} s")
print(f"  Total Time: {df['Total (mean)'].iloc[-1]:.3f} s")

# Calculate improvements
tps_improvement = ((df['TPS (mean)'].iloc[-1] - df['TPS (mean)'].iloc[0]) / df['TPS (mean)'].iloc[0]) * 100
ttft_improvement = ((df['TTFT (mean)'].iloc[0] - df['TTFT (mean)'].iloc[-1]) / df['TTFT (mean)'].iloc[0]) * 100
total_improvement = ((df['Total (mean)'].iloc[0] - df['Total (mean)'].iloc[-1]) / df['Total (mean)'].iloc[0]) * 100

print(f"\n=== Performance Improvements ===")
print(f"TPS improvement: {tps_improvement:+.1f}%")
print(f"TTFT improvement: {ttft_improvement:+.1f}%")
print(f"Total time improvement: {total_improvement:+.1f}%")
