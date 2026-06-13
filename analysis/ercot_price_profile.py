"""
ERCOT电价分析：日内模式 + 季节模式
分析节点：HB_HUBAVG, HB_NORTH, HB_HOUSTON, HB_WEST
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体（如果需要）
plt.rcParams['font.size'] = 10

# 读取数据
df = pd.read_csv('data/raw/ERCOT/processed/actual_price_hourly.csv')

# 筛选目标节点
target_nodes = ['HB_HUBAVG', 'HB_NORTH', 'HB_HOUSTON', 'HB_WEST']
df = df[df['location'].isin(target_nodes)].copy()

# 转换时间戳（使用本地时间进行日内分析）
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
df['timestamp_local'] = pd.to_datetime(df['timestamp_local'], utc=True).dt.tz_convert('America/Chicago')

# 提取时间特征（使用本地时间，确保日内高峰定位准确）
df['hour'] = df['timestamp_local'].dt.hour
df['month'] = df['timestamp_local'].dt.month
df['year_month'] = df['timestamp_local'].dt.to_period('M')  # 年-月，用于区分不同年份
df['date'] = df['timestamp_local'].dt.date
df['year'] = df['timestamp_local'].dt.year

print(f"数据时间范围: {df['timestamp_local'].min()} 到 {df['timestamp_local'].max()}")
print(f"节点数据量:")
print(df['location'].value_counts())

# ==================== 图1: 日内平均模式（分季节） ====================

# 定义季节
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ERCOT Intraday Price Patterns by Season', fontsize=14, fontweight='bold')

seasons = ['Winter', 'Spring', 'Summer', 'Fall']
colors = {'HB_HUBAVG': '#1f77b4', 'HB_NORTH': '#ff7f0e', 'HB_HOUSTON': '#2ca02c', 'HB_WEST': '#d62728'}

for idx, season in enumerate(seasons):
    ax = axes[idx // 2, idx % 2]
    season_data = df[df['season'] == season]

    for node in target_nodes:
        node_data = season_data[season_data['location'] == node]
        hourly_avg = node_data.groupby('hour')['value'].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o', markersize=3,
                label=node, color=colors[node], linewidth=1.5)

    ax.set_title(f'{season}', fontweight='bold')
    ax.set_xlabel('Hour of Day (Local Time)')
    ax.set_ylabel('Price (USD/MWh)')
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ercot_intraday_by_season.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 已保存: analysis/figures/ercot_intraday_by_season.png")

# ==================== 图2: 全年平均日内模式（所有节点对比） ====================

fig, ax = plt.subplots(figsize=(12, 6))

for node in target_nodes:
    node_data = df[df['location'] == node]
    hourly_avg = node_data.groupby('hour')['value'].mean()
    hourly_std = node_data.groupby('hour')['value'].std()

    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', markersize=4,
            label=f'{node}', color=colors[node], linewidth=2)
    ax.fill_between(hourly_avg.index,
                    hourly_avg.values - hourly_std.values,
                    hourly_avg.values + hourly_std.values,
                    alpha=0.2, color=colors[node])

ax.set_title('ERCOT Average Intraday Price Pattern (All Periods)\nShaded area = ±1 std dev',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day (Local Time)', fontsize=11)
ax.set_ylabel('Price (USD/MWh)', fontsize=11)
ax.set_xticks(range(0, 24))
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ercot_intraday_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 已保存: analysis/figures/ercot_intraday_overall.png")

# ==================== 图3: 季节变化（按年-月显示，区分不同年份） ====================

fig, ax = plt.subplots(figsize=(14, 6))

for node in target_nodes:
    node_data = df[df['location'] == node]
    # 按年-月分组，避免把2025和2026的同月混在一起
    monthly_avg = node_data.groupby('year_month')['value'].mean()

    ax.plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o', markersize=5,
            label=node, color=colors[node], linewidth=2)

ax.set_title('ERCOT Monthly Average Price Pattern (2025-2026)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year-Month', fontsize=11)
ax.set_ylabel('Price (USD/MWh)', fontsize=11)
# 每3个月显示一个x轴标签，避免拥挤
xticks = range(0, len(monthly_avg.index), 3)
ax.set_xticks([monthly_avg.index.astype(str)[i] for i in xticks])
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/figures/ercot_monthly_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 已保存: analysis/figures/ercot_monthly_pattern.png")

# ==================== 图4: 价格分布对比（箱线图） ====================

fig, ax = plt.subplots(figsize=(10, 6))

node_data_list = []
node_labels = []
for node in target_nodes:
    node_prices = df[df['location'] == node]['value'].values
    node_data_list.append(node_prices)
    node_labels.append(node)

bp = ax.boxplot(node_data_list, labels=node_labels, patch_artist=True)

# 设置颜色
for patch, node in zip(bp['boxes'], target_nodes):
    patch.set_facecolor(colors[node])
    patch.set_alpha(0.6)

ax.set_title('ERCOT Price Distribution Comparison', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (USD/MWh)', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis/figures/ercot_price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 已保存: analysis/figures/ercot_price_distribution.png")

# ==================== 统计摘要 ====================

print("\n" + "="*60)
print("统计摘要")
print("="*60)

for node in target_nodes:
    node_data = df[df['location'] == node]['value']
    print(f"\n{node}:")
    print(f"  平均值: {node_data.mean():.2f} USD/MWh")
    print(f"  中位数: {node_data.median():.2f} USD/MWh")
    print(f"  标准差: {node_data.std():.2f} USD/MWh")
    print(f"  最小值: {node_data.min():.2f} USD/MWh")
    print(f"  最大值: {node_data.max():.2f} USD/MWh")
    print(f"  负电价比例: {(node_data < 0).mean()*100:.2f}%")
    print(f"  极端高价(>$200)比例: {(node_data > 200).mean()*100:.2f}%")

print("\n分析完成！图表保存在 analysis/figures/ 目录")
