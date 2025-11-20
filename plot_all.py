#!/usr/bin/env python3
"""
Plot performance metrics from all.csv showing evolution across commits.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('all.csv')

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create short commit hashes for x-axis labels
df['Short Commit'] = df['Commit'].str[:7]

# Get unique commits in chronological order
unique_commits = df.drop_duplicates('Commit').sort_values('Date')
commit_list = unique_commits['Short Commit'].tolist()
commit_dates = unique_commits['Date'].tolist()

# Calculate aggregated metrics per commit
commit_metrics = []
for commit in unique_commits['Commit']:
    commit_data = df[df['Commit'] == commit]
    
    # Calculate average metrics across all tests
    avg_latency = commit_data['Latency (mean)'].mean()
    avg_bandwidth = commit_data['Bandwidth (mean)'].mean()
    
    commit_metrics.append({
        'Short Commit': commit[:7],
        'Date': commit_data['Date'].iloc[0],
        'Avg Latency': avg_latency,
        'Avg Bandwidth': avg_bandwidth,
        'Pass Rate': (commit_data['Checks'].str.split('/').apply(lambda x: int(x[0]) / int(x[1]) * 100).mean())
    })

metrics_df = pd.DataFrame(commit_metrics)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('AXPY Performance Evolution Across Commits', fontsize=18, fontweight='bold')

# Plot 1: Average Bandwidth Over Time (Higher is better)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(range(len(metrics_df)), metrics_df['Avg Bandwidth'], 
         marker='o', linewidth=2.5, markersize=10, color='#2E86AB', label='Avg Bandwidth')
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
ax1.set_title('Average Bandwidth Evolution', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(range(len(metrics_df)))
ax1.set_xticklabels(metrics_df['Short Commit'], rotation=45, ha='right', fontsize=9)
ax1.legend(loc='best')

# Add trend line
z = np.polyfit(range(len(metrics_df)), metrics_df['Avg Bandwidth'], 1)
p = np.poly1d(z)
ax1.plot(range(len(metrics_df)), p(range(len(metrics_df))), 
         "r--", alpha=0.5, linewidth=2, label='Trend')

# Plot 2: Average Latency Over Time (Lower is better)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(len(metrics_df)), metrics_df['Avg Latency'], 
         marker='s', linewidth=2.5, markersize=10, color='#A23B72', label='Avg Latency')
ax2.set_ylabel('Latency (μs)', fontsize=12, fontweight='bold')
ax2.set_title('Average Latency Evolution', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(range(len(metrics_df)))
ax2.set_xticklabels(metrics_df['Short Commit'], rotation=45, ha='right', fontsize=9)
ax2.legend(loc='best')

# Add trend line
z = np.polyfit(range(len(metrics_df)), metrics_df['Avg Latency'], 1)
p = np.poly1d(z)
ax2.plot(range(len(metrics_df)), p(range(len(metrics_df))), 
         "r--", alpha=0.5, linewidth=2, label='Trend')

# Plot 3: Test Pass Rate
ax3 = fig.add_subplot(gs[1, 0])
colors = ['#2E86AB' if rate == 100 else '#F18F01' for rate in metrics_df['Pass Rate']]
bars = ax3.bar(range(len(metrics_df)), metrics_df['Pass Rate'], color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Pass')
ax3.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('Test Pass Rate by Commit', fontsize=13, fontweight='bold')
ax3.set_ylim([95, 105])
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(len(metrics_df)))
ax3.set_xticklabels(metrics_df['Short Commit'], rotation=45, ha='right', fontsize=9)
ax3.legend(loc='best')

# Plot 4: Bandwidth Distribution by Test Configuration
ax4 = fig.add_subplot(gs[1, 1])
# Group by test name and calculate average bandwidth
test_groups = df.groupby('Test')['Bandwidth (mean)'].mean().sort_values(ascending=False).head(15)
ax4.barh(range(len(test_groups)), test_groups.values, color='#2E86AB', alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(test_groups)))
ax4.set_yticklabels([t.split('_')[-3] + 'x' + t.split('_')[-2] for t in test_groups.index], fontsize=8)
ax4.set_xlabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
ax4.set_title('Top 15 Test Configurations by Bandwidth', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Latency vs Bandwidth Scatter (All Tests)
ax5 = fig.add_subplot(gs[2, :])
# Color by commit chronologically
colors_scatter = plt.cm.viridis(np.linspace(0, 1, len(unique_commits)))
commit_color_map = {commit: color for commit, color in zip(unique_commits['Commit'].tolist(), colors_scatter)}

for idx, commit in enumerate(unique_commits['Commit']):
    commit_data = df[df['Commit'] == commit]
    ax5.scatter(commit_data['Latency (mean)'], commit_data['Bandwidth (mean)'], 
               c=[commit_color_map[commit]], s=50, alpha=0.6, 
               label=unique_commits.iloc[idx]['Short Commit'], edgecolors='black', linewidth=0.5)

ax5.set_xlabel('Latency (μs)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
ax5.set_title('Latency vs Bandwidth Trade-off (All Tests)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Create a simple legend with fewer entries
handles, labels = ax5.get_legend_handles_labels()
ax5.legend(handles[::2], labels[::2], loc='best', ncol=6, fontsize=8, 
          title='Commits', framealpha=0.9)

plt.tight_layout()

# Save the plot
plt.savefig('all_performance_evolution.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'all_performance_evolution.png'")

# Print summary statistics
print("\n=== Performance Summary ===")
print(f"\nFirst commit ({metrics_df['Short Commit'].iloc[0]}) on {metrics_df['Date'].iloc[0].strftime('%Y-%m-%d')}:")
print(f"  Avg Bandwidth: {metrics_df['Avg Bandwidth'].iloc[0]:.4f} GB/s")
print(f"  Avg Latency: {metrics_df['Avg Latency'].iloc[0]:.2f} μs")
print(f"  Pass Rate: {metrics_df['Pass Rate'].iloc[0]:.1f}%")

print(f"\nLatest commit ({metrics_df['Short Commit'].iloc[-1]}) on {metrics_df['Date'].iloc[-1].strftime('%Y-%m-%d')}:")
print(f"  Avg Bandwidth: {metrics_df['Avg Bandwidth'].iloc[-1]:.4f} GB/s")
print(f"  Avg Latency: {metrics_df['Avg Latency'].iloc[-1]:.2f} μs")
print(f"  Pass Rate: {metrics_df['Pass Rate'].iloc[-1]:.1f}%")

# Calculate improvements
bandwidth_improvement = ((metrics_df['Avg Bandwidth'].iloc[-1] - metrics_df['Avg Bandwidth'].iloc[0]) 
                         / metrics_df['Avg Bandwidth'].iloc[0]) * 100
latency_improvement = ((metrics_df['Avg Latency'].iloc[0] - metrics_df['Avg Latency'].iloc[-1]) 
                       / metrics_df['Avg Latency'].iloc[0]) * 100

print(f"\n=== Overall Performance Changes ===")
print(f"Bandwidth change: {bandwidth_improvement:+.1f}%")
print(f"Latency change: {latency_improvement:+.1f}%")

# Find best performing commit
best_bandwidth_idx = metrics_df['Avg Bandwidth'].idxmax()
print(f"\n=== Best Performing Commit ===")
print(f"Commit {metrics_df['Short Commit'].iloc[best_bandwidth_idx]} "
      f"({metrics_df['Date'].iloc[best_bandwidth_idx].strftime('%Y-%m-%d')})")
print(f"  Avg Bandwidth: {metrics_df['Avg Bandwidth'].iloc[best_bandwidth_idx]:.4f} GB/s")
print(f"  Avg Latency: {metrics_df['Avg Latency'].iloc[best_bandwidth_idx]:.2f} μs")

# Number of tests
print(f"\n=== Test Coverage ===")
print(f"Total test configurations: {df['Test'].nunique()}")
print(f"Total test runs: {len(df)}")
print(f"Tests per commit: {len(df) / len(unique_commits):.1f} (average)")
