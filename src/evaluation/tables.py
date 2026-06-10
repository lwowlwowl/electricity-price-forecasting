"""
生成所有模型的对比表格
======================
读取深度学习模型和 baseline 模型的预测结果，生成综合对比表格。

运行方式：
  python generate_comparison_table.py

输出文件：
  ../../data/results/all_models_comparison.csv  — 所有模型的误差指标汇总表
"""

import os
import numpy as np
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    # 深度学习模型
    "Toto-1.0":    os.path.join(RESULTS_DIR, "forecast_toto.csv"),
    "TimesFM-2.5": os.path.join(RESULTS_DIR, "forecast_timesfm.csv"),
    "Chronos-2":   os.path.join(RESULTS_DIR, "forecast_chronos2.csv"),
    # Baseline 模型（包含多个子模型）
    "Baselines":   os.path.join(RESULTS_DIR, "forecast_baselines.csv"),
}

OUTPUT_TABLE = os.path.join(RESULTS_DIR, "all_models_comparison.csv")

print("=" * 60)
print("生成所有模型对比表格")
print("=" * 60)

# ── 辅助函数 ─────────────────────────────────────────────────────────────────
def _parse_array_col(series, agg):
    """处理可能的数组字符串格式"""
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
    """确保 mean/q10/q90 列均为 float"""
    for col, agg in [('mean', 'mean'), ('q10', 'q10'), ('q90', 'q90')]:
        if col in df.columns and df[col].dtype == object:
            df = df.copy()
            df[col] = _parse_array_col(df[col], agg)
    return df


# ── 读取所有预测结果 ─────────────────────────────────────────────────────────
all_data = []
missing_files = []

for model_name, path in FILES.items():
    if not os.path.exists(path):
        missing_files.append(model_name)
        print(f"⚠️  缺少文件：{model_name}")
        continue

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = _normalize_df(df)

    # Baselines 文件包含多个模型，需要拆分
    if model_name == "Baselines":
        for baseline_model in df["model"].unique():
            sub_df = df[df["model"] == baseline_model].copy()
            all_data.append((baseline_model, sub_df))
            print(f"✅ 读取：{baseline_model}")
    else:
        all_data.append((model_name, df))
        print(f"✅ 读取：{model_name}")

if not all_data:
    print("\n❌ 没有找到任何预测结果")
    exit(1)

if missing_files:
    print(f"\n⚠️  以下文件缺失：{missing_files}")

# ── 计算每个模型的误差指标 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("计算误差指标...")
print("=" * 60)

results = []

for model_name, df in all_data:
    nodes = df["node"].unique()

    for node in nodes:
        sub = df[df["node"] == node].sort_values("timestamp")
        actual = sub["actual"].values
        pred = sub["mean"].values

        # 计算指标
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-6))) * 100

        results.append({
            "模型": model_name,
            "节点": node,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "MAPE(%)": round(mape, 2),
        })

# 创建 DataFrame 并保存
results_df = pd.DataFrame(results)

# 按模型和节点排序
model_order = [
    # 预训练深度学习模型（排前面）
    "TimesFM-2.5", "Toto-1.0", "Chronos-2",
    # 训练型深度学习模型
    "LSTM", "GRU", "MLP",
    # 梯度提升树
    "XGBoost", "LightGBM", "RandomForest",
    # 基础 Baseline
    "Seasonal-Naive", "Naive"
]

# 创建排序用的分类类型
results_df["模型"] = pd.Categorical(results_df["模型"], categories=model_order, ordered=True)
results_df = results_df.sort_values(["节点", "模型"]).reset_index(drop=True)

# 保存到 CSV
results_df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8")

# ── 打印结果 ─────────────────────────────────────────────────────────────────
print(f"\n✅ 对比表格已保存：{OUTPUT_TABLE}")
print(f"\n总计模型数：{len(all_data)}")
print(f"总计节点数：{results_df['节点'].nunique()}")

print("\n" + "=" * 60)
print("各模型平均 MAE（所有节点的平均值）")
print("=" * 60)

avg_mae = results_df.groupby("模型")["MAE"].mean().sort_values()
for model, mae in avg_mae.items():
    print(f"  {model:20s}: {mae:.3f}")

print("\n" + "=" * 60)
print("详细结果表格")
print("=" * 60)

# 按节点分组显示
for node in results_df["节点"].unique():
    print(f"\n【{node}】")
    node_data = results_df[results_df["节点"] == node].sort_values("MAE")
    print(node_data.to_string(index=False))

print("\n" + "=" * 60)
print("✅ 完成！")
print("=" * 60)
