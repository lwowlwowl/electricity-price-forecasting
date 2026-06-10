"""
零样本电价预测结果可视化
========================
读取三个模型的预测结果 CSV，生成对比图表。

运行前准备：
  1. 确保三个预测脚本已运行完毕，生成了以下文件：
       ../../data/results/forecast_toto.csv
       ../../data/results/forecast_timesfm.csv
       ../../data/results/forecast_chronos2.csv
  2. 激活 toto 虚拟环境：source ../../external/toto/.venv/bin/activate

运行方式：
  python plotting.py

输出文件：
  ../../data/results/forecast_comparison.png   — 各节点三模型对比图
  ../../data/results/forecast_metrics.csv      — MAE / RMSE 误差汇总表
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 无显示器环境下保存图片
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
rcParams["font.family"] = ["PingFang HK", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False  # 负号正常显示

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    "Toto-1.0":    os.path.join(RESULTS_DIR, "forecast_toto.csv"),
    "TimesFM-2.5": os.path.join(RESULTS_DIR, "forecast_timesfm.csv"),
    "Chronos-2":   os.path.join(RESULTS_DIR, "forecast_chronos2.csv"),
}
OUTPUT_PNG     = os.path.join(RESULTS_DIR, "forecast_comparison.png")
OUTPUT_METRICS = os.path.join(RESULTS_DIR, "forecast_metrics.csv")

MODEL_COLORS = {
    "Toto-1.0":    "#E05C5C",
    "TimesFM-2.5": "#4A90D9",
    "Chronos-2":   "#5CB85C",
}

print("=" * 60)
print("零样本电价预测结果可视化")
print("=" * 60)

# ── 辅助函数：将可能是字符串数组的列解析为标量 ────────────────────────────────
def _parse_array_col(series, agg):
    """
    若列值为字符串（如 Toto 输出的 numpy 数组字符串），
    则解析为 numpy 数组后按 agg 聚合（'mean'/'q10'/'q90'）；
    否则直接转 float。
    """
    def _convert(val):
        if isinstance(val, str):
            nums = np.fromstring(val.replace('\n', ' ').strip('[] '), sep=' ')
            if agg == 'mean':
                return float(nums.mean())
            elif agg == 'q10':
                return float(np.percentile(nums, 10))
            elif agg == 'q90':
                return float(np.percentile(nums, 90))
        return float(val)
    return series.apply(_convert)


def _normalize_df(df):
    """确保 mean/q10/q90 列均为 float（处理 Toto 的数组字符串格式）。"""
    for col, agg in [('mean', 'mean'), ('q10', 'q10'), ('q90', 'q90')]:
        if df[col].dtype == object:
            df = df.copy()
            df[col] = _parse_array_col(df[col], agg)
    return df


# ── 1. 读取预测结果 ───────────────────────────────────────────────────────────
dfs = {}
missing = []
for model, path in FILES.items():
    if os.path.exists(path):
        raw = pd.read_csv(path, parse_dates=["timestamp"])
        dfs[model] = _normalize_df(raw)
        print(f"✅ 读取 {model}：{path}")
    else:
        missing.append(model)
        print(f"⚠️  缺少 {model} 的预测文件：{path}")

if not dfs:
    print("\n❌ 没有找到任何预测结果，请先运行预测脚本")
    exit(1)

if missing:
    print(f"\n⚠️  以下模型结果缺失，将跳过：{missing}")

# ── 2. 获取节点列表 ───────────────────────────────────────────────────────────
first_df   = next(iter(dfs.values()))
node_list  = first_df["node"].unique().tolist()
n_nodes    = len(node_list)
n_models   = len(dfs)

print(f"\n节点列表：{node_list}")
print(f"模型数量：{n_models}")

# ── 3. 计算误差指标 ───────────────────────────────────────────────────────────
metrics_rows = []
for model, df in dfs.items():
    for node in node_list:
        sub = df[df["node"] == node].sort_values("timestamp")
        actual = sub["actual"].values
        pred   = sub["mean"].values
        mae    = np.mean(np.abs(actual - pred))
        rmse   = np.sqrt(np.mean((actual - pred) ** 2))
        mape   = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-6))) * 100
        metrics_rows.append({
            "model": model,
            "node":  node,
            "MAE":   round(mae, 3),
            "RMSE":  round(rmse, 3),
            "MAPE%": round(mape, 2),
        })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print(f"\n误差指标已保存：{OUTPUT_METRICS}")
print(metrics_df.to_string(index=False))

# ── 4. 绘图 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    n_nodes, 1,
    figsize=(14, 4 * n_nodes),
    sharex=False,
)
if n_nodes == 1:
    axes = [axes]

fig.suptitle("零样本电价预测对比（ENLITEN LMP，预测未来 24 小时）",
             fontsize=14, fontweight="bold", y=1.01)

for ax, node in zip(axes, node_list):
    # 画实际值（取第一个模型的 actual 列，所有模型 actual 相同）
    first_model = next(iter(dfs))
    sub_actual  = dfs[first_model][dfs[first_model]["node"] == node].sort_values("timestamp")
    timestamps  = sub_actual["timestamp"].values
    actual_vals = sub_actual["actual"].values

    ax.plot(timestamps, actual_vals,
            color="black", linewidth=2, label="实际值", zorder=5)

    # 画各模型预测
    for model, df in dfs.items():
        sub  = df[df["node"] == node].sort_values("timestamp")
        ts   = sub["timestamp"].values
        mean = sub["mean"].values
        q10  = sub["q10"].values
        q90  = sub["q90"].values
        color = MODEL_COLORS.get(model, "gray")

        ax.plot(ts, mean, color=color, linewidth=1.5,
                linestyle="--", label=f"{model} 预测", zorder=4)
        ax.fill_between(ts, q10, q90, color=color, alpha=0.15,
                        label=f"{model} 80%区间")

    # 误差标注
    node_metrics = metrics_df[metrics_df["node"] == node]
    metric_text  = "  ".join([
        f"{row['model']}: MAE={row['MAE']:.2f}"
        for _, row in node_metrics.iterrows()
    ])
    ax.set_title(f"节点：{node}    {metric_text}", fontsize=10)
    ax.set_ylabel("LMP ($/MWh)", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\n✅ 对比图已保存：{OUTPUT_PNG}")

print("\n" + "=" * 60)
print("✅ 可视化完成！")
print("=" * 60)
