"""
实验结果可视化 plotting.py
==========================
为阶段2/阶段3 提供两个【可复用】绘图函数，统一吃 run_experiment.py 落盘的
summary.csv（列：model, mae_mean, rmse_mean, mase_mean, coverage_mean,
spike_f1_mean_signal, spike_recall, ...），杜绝早期各脚本各画各的。

两个核心函数：
  plot_summary(summary_csv, out_png)
      单次实验的【模型对比】柱状图：横轴=模型，分面板画 MAE / RMSE /
      Spike-F1 / Coverage。用于阶段2 基准对照、以及每个消融某一档的快照。

  plot_ablation(summaries, knob_label, out_png)
      消融的【旋钮扫描】折线图：横轴=旋钮取值，每个模型一条线，
      分面板画 MAE / RMSE / Spike-F1。这是手册 §6 每个消融要求的
      "指标 vs 旋钮取值"图。summaries = [(旋钮值, summary_df), ...]。

也可直接命令行用：
  python plotting.py data/results/baseline/summary.csv
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示器环境保存图片
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# 稳定的模型→颜色映射（同一模型在所有图里颜色一致，方便横向比对）
MODEL_COLORS = {
    "Naive": "#9E9E9E", "SeasonalNaive": "#607D8B",
    "ETS": "#8D6E63", "Theta": "#A1887F",
    "RandomForest": "#66BB6A", "LightGBM": "#26A69A", "XGBoost": "#9CCC65",
    "TimesFM": "#4A90D9", "Chronos2": "#5C6BC0", "Toto": "#E05C5C",
}
_FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _color_for(model: str, idx: int) -> str:
    return MODEL_COLORS.get(model, _FALLBACK[idx % len(_FALLBACK)])


# ── 指标面板配置：(列名, 标题, 越小越好?) ────────────────────────────────────
_PANELS = [
    ("mae_mean", "MAE（越低越好）", True),
    ("rmse_mean", "RMSE（越低越好）", True),
    ("spike_f1_mean_signal", "Spike-F1（越高越好）", False),
    ("coverage_mean", "覆盖率（理想≈0.80）", False),
]


def _load_summary(src: Union[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.read_csv(src) if isinstance(src, str) else src.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1) 单次实验：模型对比柱状图
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary(summary: Union[str, pd.DataFrame],
                 out_png: str,
                 title: str = "模型对比",
                 sort_by: str = "mae_mean") -> str:
    """
    画一份 summary 的模型对比柱状图（MAE/RMSE/Spike-F1/Coverage 四面板）。
    summary : summary.csv 路径或 DataFrame。
    返回输出图片路径。
    """
    df = _load_summary(summary)
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
    models = df["model"].tolist()
    colors = [_color_for(m, i) for i, m in enumerate(models)]

    panels = [(c, t, lo) for c, t, lo in _PANELS if c in df.columns]
    n = len(panels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (col, ptitle, lower_better) in zip(axes, panels):
        vals = df[col].astype(float).values
        bars = ax.bar(range(len(models)), vals, color=colors)
        # coverage 面板画一条 0.8 理想线
        if col == "coverage_mean":
            ax.axhline(0.8, color="red", ls="--", lw=1, alpha=0.7, label="理想0.80")
            ax.legend(fontsize=8)
        ax.set_title(ptitle, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                    ha="center", va="bottom", fontsize=7)

    # 关掉多余空面板
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ══════════════════════════════════════════════════════════════════════════════
# 2) 消融：旋钮扫描折线图
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation(summaries: List[Tuple[object, Union[str, pd.DataFrame]]],
                  knob_label: str,
                  out_png: str,
                  title: Optional[str] = None,
                  metrics: Optional[List[str]] = None) -> str:
    """
    画消融折线图：横轴=旋钮取值，每模型一条线，分面板画各指标。

    summaries : [(旋钮取值, summary 路径或 DataFrame), ...]，按旋钮顺序排列。
                旋钮取值可为数字(如 168/336/720)或字符串标签(如 "无"/"+负荷")。
    knob_label: 横轴名称，如 "上下文长度(h)"、"协变量集合"。
    返回输出图片路径。
    """
    metrics = metrics or ["mae_mean", "rmse_mean", "spike_f1_mean_signal"]
    knob_vals = [k for k, _ in summaries]
    dfs = [_load_summary(s) for _, s in summaries]

    # 收集所有出现过的模型，保证每个模型在所有档都画上
    all_models: List[str] = []
    for d in dfs:
        for m in d["model"].tolist():
            if m not in all_models:
                all_models.append(m)

    metric_titles = {
        "mae_mean": "MAE（越低越好）",
        "rmse_mean": "RMSE（越低越好）",
        "spike_f1_mean_signal": "Spike-F1（越高越好）",
        "mase_mean": "MASE（<1 优于季节朴素）",
        "coverage_mean": "覆盖率（理想≈0.80）",
    }
    panels = [m for m in metrics if any(m in d.columns for d in dfs)]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]

    x = list(range(len(knob_vals)))
    for ax, col in zip(axes, panels):
        for i, model in enumerate(all_models):
            ys = []
            for d in dfs:
                row = d[d["model"] == model]
                ys.append(float(row[col].iloc[0]) if (len(row) and col in d.columns
                                                      and pd.notna(row[col].iloc[0]))
                          else float("nan"))
            ax.plot(x, ys, marker="o", lw=1.8, color=_color_for(model, i), label=model)
        if col == "coverage_mean":
            ax.axhline(0.8, color="red", ls="--", lw=1, alpha=0.6)
        if col == "mase_mean":
            ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
        ax.set_title(metric_titles.get(col, col), fontsize=11, fontweight="bold")
        ax.set_xlabel(knob_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in knob_vals])
        ax.grid(True, alpha=0.3)

    # 只在最后一个面板放图例，避免重复
    axes[-1].legend(fontsize=8, loc="best", ncol=1)
    fig.suptitle(title or f"消融：{knob_label}",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ══════════════════════════════════════════════════════════════════════════════
# 3) 单模型时序图：整个测试期的 actual vs predicted（每模型一张子图）
# ══════════════════════════════════════════════════════════════════════════════
def plot_timeseries(records: Union[str, pd.DataFrame],
                    out_png: str,
                    thresholds: Optional[dict] = None,
                    title: str = "预测时序对比") -> str:
    """
    为每个模型画一张子图：整个测试期的真实电价 vs 预测均值。
    records : backtest 落盘的 records.csv 或 DataFrame。
              列：model, origin, node, ts, actual, mean, q10, q90。
    thresholds : {node: spike_threshold}，有则画阈值线。
    返回输出图片路径。
    """
    df = pd.read_csv(records, parse_dates=["ts", "origin"]) if isinstance(records, str) else records.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    models = df["model"].unique().tolist()
    nodes = df["node"].unique().tolist()
    n_models = len(models)

    # 布局：每行 2 个模型
    ncols = 2
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.5 * nrows),
                             sharex=True)
    if n_models == 1:
        axes = [axes] if not hasattr(axes, '__iter__') else axes
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].sort_values("ts")
        color = _color_for(model, models.index(model))

        # 对每个节点,取每个时刻最晚一个起报点的预测(避免重叠)
        for node in nodes:
            ns = sub[sub["node"] == node].drop_duplicates(subset=["ts"], keep="last")
            ts = ns["ts"]
            # 真实值（只在第一个节点画一次即可，如多节点则叠加）
            if node == nodes[0]:
                ax.plot(ts, ns["actual"], color="black", lw=0.8, alpha=0.85,
                        label="真实电价")
            ax.plot(ts, ns["mean"], color=color, lw=0.7, alpha=0.85,
                    label=f"预测" if node == nodes[0] else None)
            # 阈值线
            if thresholds and node in thresholds and node == nodes[0]:
                ax.axhline(thresholds[node], color="red", ls="--", lw=0.8,
                           alpha=0.5, label=f"尖峰阈值")

        # 如果多节点,只画第一个节点的 actual/pred,避免太乱
        # (已在上面逻辑通过只第一次画 actual 实现)

        # 计算整体 MAE 标注到标题
        sub_first = sub[sub["node"] == nodes[0]].drop_duplicates(subset=["ts"], keep="last")
        mae_val = float(np.abs(sub_first["actual"] - sub_first["mean"]).mean())
        ax.set_title(f"{model}（MAE={mae_val:.1f}）", fontsize=11, fontweight="bold")
        ax.set_ylabel("$/MWh", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    # 关掉多余空面板
    for ax in axes[n_models:]:
        ax.axis("off")

    # x 轴日期格式
    import matplotlib.dates as mdates
    for ax in axes[:n_models]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：")
        print("  python plotting.py <summary.csv> [输出png]          -- 模型对比柱状图")
        print("  python plotting.py --ts <records.csv> [输出png]     -- 时序预测图")
        sys.exit(1)
    if sys.argv[1] == "--ts":
        src = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
            os.path.dirname(os.path.abspath(src)), "timeseries_compare.png")
        p = plot_timeseries(src, out, title="预测时序对比")
        print(f"✅ 时序图已保存：{p}")
    else:
        src = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(os.path.abspath(src)), "summary_compare.png")
        p = plot_summary(src, out, title="模型对比")
        print(f"✅ 对比图已保存：{p}")
