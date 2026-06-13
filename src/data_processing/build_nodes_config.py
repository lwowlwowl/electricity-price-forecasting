"""
节点清单生成脚本 build_nodes_config.py
======================================
对每个市场的电价数据做统计，把代表性节点分成三组并固化到 configs/nodes.yaml：
  - volatility : 标准差最大（价格波动最丰富）
  - spikes     : 尖峰次数最多（> 均值 + 2*标准差 的次数）
  - stable     : 最平稳（标准差最小，作为对照组）

固化节点清单的目的：之后所有消融实验都从这份固定清单里选节点，
保证不同实验之间结果可比、可复现。

用法：
  python build_nodes_config.py            # 处理默认市场
  python build_nodes_config.py ERCOT      # 只处理指定市场
"""

import os
import sys
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(SCRIPT_DIR, "../../data/raw")
CONFIG_DIR  = os.path.join(SCRIPT_DIR, "../../configs")
os.makedirs(CONFIG_DIR, exist_ok=True)
OUTPUT_YAML = os.path.join(CONFIG_DIR, "nodes.yaml")

DEFAULT_MARKETS = ["ERCOT", "CAISO", "NYISO", "PJM"]
N_PER_GROUP = 3          # 每组取几个节点
SPIKE_SIGMA = 2.0        # 尖峰定义：> 均值 + SPIKE_SIGMA * 标准差


def analyze_market(market: str):
    """读取某市场小时级电价，返回每节点统计及三组分类。失败返回 None。"""
    path = os.path.join(RAW_DIR, market, "processed", "actual_price_hourly.csv")
    if not os.path.exists(path):
        print(f"  ⚠️ 跳过 {market}：找不到 {path}")
        return None

    df = pd.read_csv(path, usecols=["location", "value"])
    stats = []
    for node, g in df.groupby("location"):
        v = g["value"]
        threshold = v.mean() + SPIKE_SIGMA * v.std()
        stats.append({
            "node":   node,
            "std":    float(v.std()),
            "mean":   float(v.mean()),
            "spikes": int((v > threshold).sum()),
            "max":    float(v.max()),
            "min":    float(v.min()),
        })
    stats_df = pd.DataFrame(stats)

    # 三组互斥：依次挑选并把已选节点从候选池移除，保证对照有效、节点不重叠。
    # 顺序 volatility → spikes → stable，让“整体波动”和“短促尖峰”两个维度都被覆盖。
    remaining = stats_df.copy()
    groups = {}

    pick = remaining.nlargest(N_PER_GROUP, "std")["node"].tolist()
    groups["volatility"] = pick
    remaining = remaining[~remaining["node"].isin(pick)]

    pick = remaining.nlargest(N_PER_GROUP, "spikes")["node"].tolist()
    groups["spikes"] = pick
    remaining = remaining[~remaining["node"].isin(pick)]

    pick = remaining.nsmallest(N_PER_GROUP, "std")["node"].tolist()
    groups["stable"] = pick

    return stats_df, groups


def dump_yaml(all_groups: dict) -> str:
    """手写一个简洁的 YAML（避免依赖 pyyaml）。"""
    lines = [
        "# 各市场代表性电价节点清单（由 build_nodes_config.py 自动生成）",
        "# volatility=波动最大  spikes=尖峰最多  stable=最平稳(对照)",
        "",
    ]
    for market, groups in all_groups.items():
        lines.append(f"{market}:")
        for grp in ("volatility", "spikes", "stable"):
            arr = ", ".join(groups[grp])
            lines.append(f"  {grp}: [{arr}]")
        lines.append("")
    return "\n".join(lines)


def main():
    markets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MARKETS
    print("=" * 70)
    print("生成节点清单 configs/nodes.yaml")
    print("=" * 70)

    all_groups = {}
    for market in markets:
        print(f"\n── 分析 {market} ──")
        res = analyze_market(market)
        if res is None:
            continue
        stats_df, groups = res
        all_groups[market] = groups

        show = stats_df.sort_values("std", ascending=False)
        print(show.round(2).to_string(index=False))
        print(f"  → volatility: {groups['volatility']}")
        print(f"  → spikes    : {groups['spikes']}")
        print(f"  → stable    : {groups['stable']}")

    if not all_groups:
        print("\n❌ 没有任何市场被成功分析")
        return

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        f.write(dump_yaml(all_groups))
    print(f"\n✅ 已写入：{OUTPUT_YAML}")


if __name__ == "__main__":
    main()
