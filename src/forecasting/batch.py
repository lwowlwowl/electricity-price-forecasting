"""
零样本预测实验系统
==================
支持：
  - 多种节点选择（波动最大、尖峰最多）
  - 多个时间段（年终、夏季高峰、节假日等）
  - 自动对比结果

运行:
  python experiment_zero_shot.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 配置 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "../../data/processed/lmp_processed.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTEXT_LEN = 168
PRED_LEN = 24
N_NODES = 5

# 定义实验组合
EXPERIMENTS = [
    # (实验名称, 预测开始时间, 节点选择策略)
    ("年终-波动大", "2020-12-31 00:00", "volatility"),
    ("年终-尖峰", "2020-12-31 00:00", "spikes"),
    ("夏季高峰-波动大", "2020-07-15 00:00", "volatility"),
    ("夏季高峰-尖峰", "2020-07-15 00:00", "spikes"),
    ("节假日-波动大", "2020-12-25 00:00", "volatility"),
    ("节假日-尖峰", "2020-12-25 00:00", "spikes"),
]

print("=" * 70)
print("零样本预测实验系统")
print("=" * 70)

# ── 读取数据 ─────────────────────────────────────────────────────────────────
print(f"\n📊 读取数据: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, index_col="timestamp", parse_dates=True)
print(f"    时间范围: {df.index[0]} → {df.index[-1]}")
print(f"    节点总数: {df.shape[1]}")

# ── 节点选择函数 ────────────────────────────────────────────────────────────
def select_nodes(data, strategy="volatility", n=5):
    """选择节点"""
    if strategy == "volatility":
        # 标准差最大
        scores = data.std().sort_values(ascending=False)
        return scores.head(n).index.tolist()

    elif strategy == "spikes":
        # 尖峰最多（超过均值+2倍标准差的次数）
        spike_counts = {}
        for col in data.columns:
            s = data[col]
            threshold = s.mean() + 2 * s.std()
            spike_counts[col] = (s > threshold).sum()

        spike_series = pd.Series(spike_counts).sort_values(ascending=False)
        return spike_series.head(n).index.tolist()

    return data.columns[:n].tolist()

# ── Baseline模型 ───────────────────────────────────────────────────────────
class NaiveForecaster:
    name = "Naive"
    def fit_predict(self, train_data, pred_len):
        last = train_data[-1]
        mean = np.full(pred_len, last)
        return mean, mean*0.9, mean*1.1

class SeasonalNaiveForecaster:
    name = "Seasonal-Naive"
    def fit_predict(self, train_data, pred_len):
        pattern = train_data[-24:]
        repeats = (pred_len // 24) + 1
        mean = np.tile(pattern, repeats)[:pred_len]
        return mean, mean*0.9, mean*1.1

class RandomForestForecaster:
    name = "RandomForest"
    def fit_predict(self, train_data, pred_len):
        try:
            from sklearn.ensemble import RandomForestRegressor

            # 简单特征：滞后+均值
            values = train_data[-500:]  # 用最近500小时
            X, y = [], []

            for i in range(24, len(values)):
                feat = [
                    values[i-1], values[i-2], values[i-24],
                    np.mean(values[i-24:i]),
                    i % 24  # 小时
                ]
                X.append(feat)
                y.append(values[i])

            X, y = np.array(X), np.array(y)
            if len(X) < 50:
                return None, None, None

            model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X, y)

            # 预测
            preds = []
            current = list(values)
            for _ in range(pred_len):
                feat = np.array([[
                    current[-1], current[-2], current[-24],
                    np.mean(current[-24:]),
                    (len(current) % 24)
                ]])
                p = model.predict(feat)[0]
                preds.append(p)
                current.append(p)

            mean = np.array(preds)
            std = np.std(y - model.predict(X))
            return mean, mean-1.28*std, mean+1.28*std

        except Exception as e:
            return None, None, None

# 模型列表
MODELS = [
    NaiveForecaster(),
    SeasonalNaiveForecaster(),
    RandomForestForecaster(),
]

print(f"\n🤖 可用模型: {[m.name for m in MODELS]}")

# ── 运行实验 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("开始实验")
print("=" * 70)

all_results = []

for exp_name, time_str, strategy in EXPERIMENTS:
    print(f"\n📌 {exp_name}")
    print("-" * 50)

    pred_time = pd.Timestamp(time_str)

    # 检查时间有效性
    if pred_time - timedelta(hours=CONTEXT_LEN) < df.index[0]:
        print(f"    ⚠️ 时间点太早，跳过")
        continue
    if pred_time + timedelta(hours=PRED_LEN) > df.index[-1]:
        print(f"    ⚠️ 时间点太晚，跳过")
        continue

    # 选择节点
    nodes = select_nodes(df, strategy, N_NODES)
    print(f"    📍 节点: {nodes}")

    # 准备数据
    context_start = pred_time - timedelta(hours=CONTEXT_LEN)
    context_df = df.loc[context_start:pred_time-timedelta(hours=1), nodes]
    actual_df = df.loc[pred_time:pred_time+timedelta(hours=PRED_LEN-1), nodes]
    train_df = df.loc[:pred_time-timedelta(hours=1), nodes]

    # 运行每个模型
    for model in MODELS:
        maes = []
        for node in nodes:
            train_data = train_df[node].values.astype(np.float32)
            actual = actual_df[node].values

            mean, _, _ = model.fit_predict(train_data, PRED_LEN)
            if mean is not None:
                mae = np.mean(np.abs(actual - mean))
                maes.append(mae)

        if maes:
            avg_mae = np.mean(maes)
            all_results.append({
                "experiment": exp_name,
                "time": time_str,
                "strategy": strategy,
                "model": model.name,
                "mae": avg_mae,
                "nodes": ",".join(nodes),
            })
            print(f"    {model.name:20s}: MAE = {avg_mae:.3f}")

# ── 保存和汇总 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("实验结果汇总")
print("=" * 70)

if all_results:
    results_df = pd.DataFrame(all_results)
    output = os.path.join(RESULTS_DIR, "zero_shot_experiments.csv")
    results_df.to_csv(output, index=False)
    print(f"\n✅ 结果已保存: {output}")

    # 打印表格
    print("\n详细结果:")
    print(results_df.to_string(index=False))

    # 每个实验的最佳模型
    print("\n" + "-" * 70)
    print("各实验最佳模型:")
    print("-" * 70)
    for exp in results_df['experiment'].unique():
        exp_data = results_df[results_df['experiment'] == exp]
        best = exp_data.loc[exp_data['mae'].idxmin()]
        print(f"{exp:25s}: {best['model']:20s} (MAE={best['mae']:.3f})")

else:
    print("⚠️ 没有成功运行的实验")

print("\n" + "=" * 70)
print("实验完成!")
print("=" * 70)
