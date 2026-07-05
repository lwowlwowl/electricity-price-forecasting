"""
四市场电价综合分析：ERCOT / PJM / CAISO / NYISO
================================================
覆盖 TODO 1.5 的可视化需求，同时扩展 analysis/ercot_price_profile.py
到全部四个市场，供论文数据章节使用。

用法（从项目根目录运行）：
    external/timesfm/.venv/bin/python3 analysis/multimarket_price_profile.py

输出（analysis/figures/multimarket/）：
    01_market_overview.png       — 四市场关键统计对比（负价格%、std、最高价）
    02_intraday_patterns.png     — 各市场日内均价模式（代表节点，UTC 小时）
    03_monthly_seasonality.png   — 各市场月均价走势（测试期）
    04_price_distributions.png   — 各市场价格分布（violin plot）
    05_negative_price_heatmap.png — 各市场负价格的小时×月份分布热力图
    06_spike_heatmap.png         — 各市场尖峰（P95）的小时×月份分布热力图
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 从项目根目录运行时的路径设置
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "data_processing"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import loader

# ── 配置 ──────────────────────────────────────────────────────────────────────
OUTDIR = os.path.join(_ROOT, "analysis", "figures", "multimarket")
os.makedirs(OUTDIR, exist_ok=True)

# 各市场代表节点（与 nodes.yaml ablation 节点对应）
MARKET_NODES = {
    "ERCOT": ["LZ_LCRA", "LZ_WEST", "LZ_RAYBN"],
    "PJM":   ["HUB:WESTERN HUB", "HUB:CHICAGO HUB", "ZONE:PJM-RTO"],
    "CAISO": ["TH_SP15_GEN-APND", "TH_NP15_GEN-APND", "TH_ZP26_GEN-APND"],
    "NYISO": ["N.Y.C.", "LONGIL", "WEST"],
}

# 全量数据期（含训练期）
DATA_START = "2025-01-01"
DATA_END   = "2026-06-02"

# 测试期（Lago checklist #1 的 12 个月）
TEST_START = "2025-07-01"
TEST_END   = "2026-06-01"

# 配色
MARKET_COLORS = {
    "ERCOT": "#1f77b4",
    "PJM":   "#ff7f0e",
    "CAISO": "#2ca02c",
    "NYISO": "#d62728",
}

# 节点简短别名（用于图例）
NODE_SHORT = {
    "LZ_LCRA": "LCRA", "LZ_WEST": "WEST", "LZ_RAYBN": "RAYBN",
    "HUB:WESTERN HUB": "W.Hub", "HUB:CHICAGO HUB": "Chi.Hub", "ZONE:PJM-RTO": "RTO",
    "TH_SP15_GEN-APND": "SP15", "TH_NP15_GEN-APND": "NP15", "TH_ZP26_GEN-APND": "ZP26",
    "N.Y.C.": "NYC", "LONGIL": "LongI", "WEST": "West",
}

# ── 数据加载 ──────────────────────────────────────────────────────────────────
print("加载各市场数据（测试期）…")
market_data = {}
market_stats = {}   # 统计汇总

for mkt, nodes in MARKET_NODES.items():
    print(f"  {mkt}: {nodes}")
    df = loader.load_slice(market=mkt, nodes=nodes, freq="1h",
                           start=TEST_START, end=TEST_END)
    market_data[mkt] = df

    # 统计：全部代表节点合并
    price_cols = [c for c in df.columns if c.startswith("price__")]
    all_prices = df[price_cols].values.ravel()
    all_prices = all_prices[~np.isnan(all_prices)]
    p95 = float(np.nanquantile(all_prices, 0.95))
    market_stats[mkt] = {
        "mean":       round(all_prices.mean(), 1),
        "std":        round(all_prices.std(), 1),
        "min":        round(all_prices.min(), 1),
        "max":        round(all_prices.max(), 1),
        "neg_pct":    round((all_prices < 0).mean() * 100, 1),
        "spike_pct":  round((all_prices > p95).mean() * 100, 1),
        "p95":        round(p95, 1),
    }

print()
print("统计摘要（测试期，代表节点合并）：")
stats_df = pd.DataFrame(market_stats).T
print(stats_df.to_string())
stats_df.to_csv(os.path.join(OUTDIR, "market_stats_summary.csv"))

# 同时生成逐节点详表 → analysis/market_data_overview.csv（供图 0 读取）
node_rows = []
for mkt, nodes in MARKET_NODES.items():
    df = market_data[mkt]
    for node in nodes:
        col = f"price__{node}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        p95 = float(np.nanquantile(s, 0.95))
        node_rows.append({
            "Market": mkt, "Node": node,
            "Mean":  round(s.mean(), 1),
            "Std":   round(s.std(), 1),
            "Min":   round(s.min(), 1),
            "Max":   round(s.max(), 1),
            "Neg%":  round((s < 0).mean() * 100, 1),
            "P95":   round(p95, 1),
            "Spike%": round((s > p95).mean() * 100, 1),
            "Missing%": round(df[col].isna().mean() * 100, 1),
        })
node_detail_df = pd.DataFrame(node_rows)
overview_csv = os.path.join(_ROOT, "analysis", "market_data_overview.csv")
node_detail_df.to_csv(overview_csv, index=False)
print(f"\n逐节点详表已保存：{overview_csv}")

# ── 图 0：逐节点数据概况表 ──────────────────────────────────────────────────
print("\n生成图 0：逐节点数据概况表…")
ov = node_detail_df.drop(columns=["Missing%", "Spike%"], errors="ignore")

row_colors = []
for mkt in ov["Market"]:
    base  = MARKET_COLORS.get(mkt, "#ffffff")
    rgb   = mcolors.to_rgb(base)
    light = tuple(0.85 + 0.15 * c for c in rgb)
    row_colors.append([light] * len(ov.columns))

fig, ax = plt.subplots(figsize=(13, 0.55 * len(ov) + 1.5))
ax.axis("off")

tbl = ax.table(
    cellText=ov.values,
    colLabels=ov.columns,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)

for col_idx in range(len(ov.columns)):
    tbl[0, col_idx].set_facecolor("#2c3e50")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

ax.set_title(
    "Market Data Overview — Representative Nodes (Test Period: Jul 2025 – Jun 2026)",
    fontsize=11, fontweight="bold", pad=12,
)
plt.tight_layout()
p = os.path.join(OUTDIR, "00_data_overview_table.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 1：市场关键统计对比 ─────────────────────────────────────────────────────
print("\n生成图 1：市场关键统计对比…")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Four-Market Statistics Comparison (Test Period: Jul 2025 – Jun 2026)",
             fontsize=13, fontweight="bold")

markets = list(MARKET_NODES.keys())
colors  = [MARKET_COLORS[m] for m in markets]
x = np.arange(len(markets))
bar_kw = dict(width=0.6, color=colors, edgecolor="white", linewidth=0.8)

# 1a: 标准差
ax = axes[0]
vals = [market_stats[m]["std"] for m in markets]
bars = ax.bar(x, vals, **bar_kw)
ax.set_title("Price Volatility (Std Dev)", fontweight="bold")
ax.set_ylabel("USD/MWh")
ax.set_xticks(x); ax.set_xticklabels(markets)
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set_ylim(0, max(vals) * 1.25)
ax.grid(axis="y", alpha=0.3)

# 1b: 负价格比例
ax = axes[1]
vals = [market_stats[m]["neg_pct"] for m in markets]
bars = ax.bar(x, vals, **bar_kw)
ax.set_title("Negative Price Rate (%)", fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x); ax.set_xticklabels(markets)
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_ylim(0, max(vals) * 1.3 + 1)
ax.grid(axis="y", alpha=0.3)

# 1c: 最高价（极端事件）
ax = axes[2]
vals = [market_stats[m]["max"] for m in markets]
bars = ax.bar(x, vals, **bar_kw)
ax.set_title("Maximum Observed Price", fontweight="bold")
ax.set_ylabel("USD/MWh")
ax.set_xticks(x); ax.set_xticklabels(markets)
ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
ax.set_ylim(0, max(vals) * 1.15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p = os.path.join(OUTDIR, "01_market_overview.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 2：日内均价模式 ─────────────────────────────────────────────────────────
print("生成图 2：日内均价模式…")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Average Intraday Price Pattern by Market (UTC Hour, Test Period)",
             fontsize=13, fontweight="bold")

for idx, mkt in enumerate(markets):
    ax = axes[idx // 2, idx % 2]
    df = market_data[mkt]
    price_cols = [c for c in df.columns if c.startswith("price__")]

    for col in price_cols:
        node = col.replace("price__", "")
        s = df[col].dropna()
        hourly_mean = s.groupby(s.index.hour).mean()
        hourly_std  = s.groupby(s.index.hour).std()
        label = NODE_SHORT.get(node, node)
        line, = ax.plot(hourly_mean.index, hourly_mean.values,
                        marker="o", markersize=3, linewidth=1.8, label=label)
        ax.fill_between(hourly_mean.index,
                        hourly_mean - hourly_std * 0.3,
                        hourly_mean + hourly_std * 0.3,
                        alpha=0.15, color=line.get_color())

    ax.set_title(mkt, fontweight="bold", color=MARKET_COLORS[mkt])
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Price (USD/MWh)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

plt.tight_layout()
p = os.path.join(OUTDIR, "02_intraday_patterns.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 3：月均价走势 ──────────────────────────────────────────────────────────
print("生成图 3：月均价走势…")
fig, ax = plt.subplots(figsize=(13, 6))

for mkt in markets:
    df = market_data[mkt]
    price_cols = [c for c in df.columns if c.startswith("price__")]
    # 取代表节点均值（节点间平均）
    avg = df[price_cols].mean(axis=1)
    monthly = avg.resample("MS").mean()   # Month Start

    ax.plot(monthly.index, monthly.values,
            marker="o", markersize=5, linewidth=2,
            color=MARKET_COLORS[mkt], label=mkt)
    ax.fill_between(monthly.index,
                    monthly.values * 0.85, monthly.values * 1.15,
                    alpha=0.1, color=MARKET_COLORS[mkt])

ax.set_title("Monthly Average Price by Market (Test Period)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Price (USD/MWh)")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
plt.xticks(rotation=30)

plt.tight_layout()
p = os.path.join(OUTDIR, "03_monthly_seasonality.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 4：价格分布（violin plot）────────────────────────────────────────────
print("生成图 4：价格分布…")
fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=False)
fig.suptitle("Price Distribution by Market — Representative Nodes (Test Period)",
             fontsize=12, fontweight="bold")

for idx, mkt in enumerate(markets):
    ax = axes[idx]
    df = market_data[mkt]
    price_cols = [c for c in df.columns if c.startswith("price__")]

    data_list  = []
    tick_labels = []
    for col in price_cols:
        s = df[col].dropna().values
        # 裁剪极端值以便 violin 可读（仍标注真实最高价）
        p01 = np.percentile(s, 1)
        p99 = np.percentile(s, 99)
        data_list.append(s[(s >= p01) & (s <= p99)])
        tick_labels.append(NODE_SHORT.get(col.replace("price__", ""),
                                          col.replace("price__", "")))

    parts = ax.violinplot(data_list, positions=range(len(data_list)),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(MARKET_COLORS[mkt])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.set_title(mkt, fontweight="bold", color=MARKET_COLORS[mkt])
    ax.set_ylabel("Price (USD/MWh)" if idx == 0 else "")
    ax.set_xticks(range(len(data_list)))
    ax.set_xticklabels(tick_labels, rotation=20, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # 标注真实最高价
    real_max = market_stats[mkt]["max"]
    ax.text(0.98, 0.98, f"max: ${real_max:.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="red")

plt.tight_layout()
p = os.path.join(OUTDIR, "04_price_distributions.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 5：负价格热力图（小时 × 月份）────────────────────────────────────────
print("生成图 5：负价格热力图…")
fig, axes = plt.subplots(1, 4, figsize=(17, 5))
fig.suptitle("Negative Price Frequency: Hour × Month Heatmap (Test Period, % of hours)",
             fontsize=12, fontweight="bold")

for idx, mkt in enumerate(markets):
    ax = axes[idx]
    df = market_data[mkt]
    price_cols = [c for c in df.columns if c.startswith("price__")]
    avg = df[price_cols].mean(axis=1)   # 节点均值

    # 构建 hour × month 的负价格比例矩阵
    tmp = pd.DataFrame({
        "price": avg,
        "hour": avg.index.hour,
        "month": avg.index.month,
    }).dropna()
    tmp["neg"] = (tmp["price"] < 0).astype(float)
    pivot = tmp.pivot_table(values="neg", index="hour", columns="month", aggfunc="mean") * 100

    im = ax.imshow(pivot.values, aspect="auto", origin="upper",
                   cmap="YlOrRd", vmin=0, vmax=max(30, pivot.values.max()))
    ax.set_title(mkt, fontweight="bold", color=MARKET_COLORS[mkt])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour (UTC)" if idx == 0 else "")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{m}" for m in pivot.columns], fontsize=7)
    ax.set_yticks(range(0, 24, 4))
    ax.set_yticklabels(range(0, 24, 4), fontsize=7)
    plt.colorbar(im, ax=ax, label="%" if idx == 3 else "")

plt.tight_layout()
p = os.path.join(OUTDIR, "05_negative_price_heatmap.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

# ── 图 6：尖峰热力图（小时 × 月份）──────────────────────────────────────────
print("生成图 6：尖峰热力图…")
fig, axes = plt.subplots(1, 4, figsize=(17, 5))
fig.suptitle("Price Spike Frequency (>P95): Hour × Month Heatmap (Test Period, % of hours)",
             fontsize=12, fontweight="bold")

for idx, mkt in enumerate(markets):
    ax = axes[idx]
    df = market_data[mkt]
    price_cols = [c for c in df.columns if c.startswith("price__")]
    avg = df[price_cols].mean(axis=1).dropna()
    p95 = float(np.nanquantile(avg.values, 0.95))

    tmp = pd.DataFrame({
        "price": avg,
        "hour": avg.index.hour,
        "month": avg.index.month,
    })
    tmp["spike"] = (tmp["price"] > p95).astype(float)
    pivot = tmp.pivot_table(values="spike", index="hour", columns="month", aggfunc="mean") * 100

    im = ax.imshow(pivot.values, aspect="auto", origin="upper",
                   cmap="Blues", vmin=0, vmax=25)
    ax.set_title(f"{mkt}\n(P95={p95:.0f} $/MWh)", fontweight="bold",
                 color=MARKET_COLORS[mkt])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour (UTC)" if idx == 0 else "")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{m}" for m in pivot.columns], fontsize=7)
    ax.set_yticks(range(0, 24, 4))
    ax.set_yticklabels(range(0, 24, 4), fontsize=7)
    plt.colorbar(im, ax=ax, label="%" if idx == 3 else "")

plt.tight_layout()
p = os.path.join(OUTDIR, "06_spike_heatmap.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ {p}")

print("\n" + "=" * 60)
print(f"全部完成！共 6 张图 + 1 个统计表")
print(f"输出目录：{OUTDIR}")
print("=" * 60)
