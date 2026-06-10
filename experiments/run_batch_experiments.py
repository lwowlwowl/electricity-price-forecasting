"""
批量零样本预测实验
====================
支持多种节点选择策略和多个时间段的组合实验

运行方式:
  python run_batch_experiments.py

实验配置:
  - 节点选择: 波动最大、尖峰最多、随机
  - 时间段: 可以指定多个预测时间点
  - 自动汇总所有结果
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 配置 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "../dataset/lmp_processed.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../dataset/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 实验参数
CONTEXT_LEN = 168   # 上下文长度（7天）
PRED_LEN = 24       # 预测长度（24小时）
N_NODES = 5         # 每次实验选几个节点

# 时间段配置：可以添加多个时间点
# 格式: ("名称", "预测开始时间")
TIME_SLOTS = [
    ("年终", "2020-12-31 00:00"),      # 当前使用的
    ("夏季高峰", "2020-07-15 00:00"),  # 夏季用电高峰
    ("春季平日", "2020-04-15 00:00"),  # 普通工作日
    ("冬季平日", "2020-01-15 00:00"),  # 冬季普通日
    ("节假日", "2020-12-25 00:00"),    # 圣诞节
]

# 节点选择策略
NODE_STRATEGIES = {
    "波动最大": "volatility",
    "尖峰最多": "spikes",
    "随机选择": "random",
}

print("=" * 70)
print("批量零样本预测实验")
print("=" * 70)

# ── 1. 读取数据 ─────────────────────────────────────────────────────────────
print(f"\n读取数据: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, index_col="timestamp", parse_dates=True)
print(f"数据形状: {df.shape}")
print(f"时间范围: {df.index[0]} → {df.index[-1]}")

# ── 2. 节点筛选函数 ─────────────────────────────────────────────────────────
def select_nodes(data, strategy="volatility", n=5, seed=42):
    """
    根据策略选择节点

    Args:
        data: DataFrame，列是节点
        strategy: "volatility"(波动), "spikes"(尖峰), "random"(随机)
        n: 选择节点数
        seed: 随机种子
    """
    if strategy == "volatility":
        # 按标准差排序（波动最大）
        scores = data.std().sort_values(ascending=False)
        selected = scores.head(n).index.tolist()
        print(f"    波动最大的{n}个节点: {selected}")

    elif strategy == "spikes":
        # 按"尖峰"程度排序
        # 定义: 超过均值+2倍标准差的次数
        spike_scores = {}
        for col in data.columns:
            series = data[col]
            threshold = series.mean() + 2 * series.std()
            spike_count = (series > threshold).sum()
            spike_scores[col] = spike_count

        spike_df = pd.Series(spike_scores).sort_values(ascending=False)
        selected = spike_df.head(n).index.tolist()
        print(f"    尖峰最多的{n}个节点: {selected}")
        print(f"    尖峰次数: {spike_df.head(n).values}")

    elif strategy == "random":
        np.random.seed(seed)
        selected = np.random.choice(data.columns, n, replace=False).tolist()
        print(f"    随机选择的{n}个节点: {selected}")

    else:
        raise ValueError(f"未知策略: {strategy}")

    return selected

# ── 3. 运行单个实验 ─────────────────────────────────────────────────────────
def run_experiment(df, time_slot_name, pred_start_time, node_strategy,
                   forecaster_class, model_name):
    """
    运行单个实验配置

    Returns:
        dict: 包含MAE、RMSE等指标
    """
    pred_start = pd.Timestamp(pred_start_time)

    # 检查时间是否有效
    if pred_start < df.index[0] + timedelta(hours=CONTEXT_LEN):
        print(f"    ⚠️ 时间点太早，跳过")
        return None

    if pred_start + timedelta(hours=PRED_LEN) > df.index[-1]:
        print(f"    ⚠️ 时间点太晚，跳过")
        return None

    # 选择节点
    node_cols = select_nodes(df, strategy=node_strategy, n=N_NODES)

    # 准备数据
    context_start = pred_start - timedelta(hours=CONTEXT_LEN)
    context_df = df.loc[context_start:pred_start - timedelta(hours=1), node_cols]
    actual_df = df.loc[pred_start:pred_start + timedelta(hours=PRED_LEN-1), node_cols]

    # 训练数据（预测时间点之前的所有数据）
    train_df = df.loc[:pred_start - timedelta(hours=1), node_cols]

    # 实例化模型
    try:
        forecaster = forecaster_class()
        if hasattr(forecaster, 'available') and not forecaster.available:
            return None
        if hasattr(forecaster, 'model') and forecaster.model is None:
            return None
    except Exception as e:
        print(f"    ⚠️ 模型初始化失败: {e}")
        return None

    # 对每个节点预测
    all_mae = []
    all_rmse = []

    for node in node_cols:
        train_data = train_df[node].values.astype(np.float32)
        actual = actual_df[node].values

        try:
            mean, _, _ = forecaster.fit_predict(train_data, PRED_LEN)
            if mean is None:
                continue

            mae = np.mean(np.abs(actual - mean))
            rmse = np.sqrt(np.mean((actual - mean) ** 2))
            all_mae.append(mae)
            all_rmse.append(rmse)
        except Exception as e:
            print(f"    ⚠️ {node} 预测失败: {e}")
            continue

    if len(all_mae) == 0:
        return None

    return {
        "time_slot": time_slot_name,
        "pred_time": pred_start_time,
        "node_strategy": node_strategy,
        "model": model_name,
        "avg_mae": np.mean(all_mae),
        "avg_rmse": np.mean(all_rmse),
        "nodes": ",".join(node_cols),
    }

# ── 4. 导入Baseline模型 ─────────────────────────────────────────────────────
print("\n导入模型...")
sys.path.insert(0, SCRIPT_DIR)

# 从 forecast_baselines.py 导入模型类
exec(open(os.path.join(SCRIPT_DIR, "forecast_baselines.py")).read())

MODELS = [
    ("Naive", NaiveForecaster),
    ("Seasonal-Naive", SeasonalNaiveForecaster),
    ("RandomForest", RandomForestForecaster),
    ("XGBoost", XGBoostForecaster),
    ("LightGBM", LightGBMForecaster),
    ("MLP", MLPForecaster),
]

# 过滤掉不可用的模型
available_models = []
for name, cls in MODELS:
    try:
        instance = cls()
        if hasattr(instance, 'available') and not instance.available:
            continue
        if hasattr(instance, 'model') and instance.model is None:
            continue
        available_models.append((name, cls))
        print(f"  ✓ {name}")
    except:
        print(f"  ✗ {name} (不可用)")

print(f"\n可用模型数: {len(available_models)}")

# ── 5. 运行所有组合 ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("开始批量实验")
print("=" * 70)

all_results = []

for time_name, time_start in TIME_SLOTS:
    print(f"\n📅 时间段: {time_name} ({time_start})")
    print("-" * 50)

    for strategy_name, strategy_key in NODE_STRATEGIES.items():
        print(f"\n  📊 节点选择: {strategy_name}")

        for model_name, model_class in available_models:
            print(f"    🤖 模型: {model_name}...", end=" ")

            result = run_experiment(
                df, time_name, time_start, strategy_key,
                model_class, model_name
            )

            if result:
                all_results.append(result)
                print(f"MAE={result['avg_mae']:.3f}")
            else:
                print("失败")

# ── 6. 保存结果 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("实验完成")
print("=" * 70)

if all_results:
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(RESULTS_DIR, "batch_experiment_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存: {output_file}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    # 按时间段和策略分组显示
    for time_slot in results_df['time_slot'].unique():
        print(f"\n📅 {time_slot}:")
        time_data = results_df[results_df['time_slot'] == time_slot]

        for strategy in time_data['node_strategy'].unique():
            print(f"  📊 {strategy}:")
            strategy_data = time_data[time_data['node_strategy'] == strategy]

            # 按MAE排序
            strategy_data = strategy_data.sort_values('avg_mae')
            for _, row in strategy_data.iterrows():
                print(f"    {row['model']:20s} MAE={row['avg_mae']:.3f} RMSE={row['avg_rmse']:.3f}")

    # 找出最佳组合
    print("\n" + "=" * 70)
    print("🏆 各时间段最佳模型")
    print("=" * 70)
    for time_slot in results_df['time_slot'].unique():
        time_data = results_df[results_df['time_slot'] == time_slot]
        best = time_data.loc[time_data['avg_mae'].idxmin()]
        print(f"{time_slot:15s}: {best['model']} + {best['node_strategy']} (MAE={best['avg_mae']:.3f})")

else:
    print("\n⚠️ 没有成功运行的实验")

print("\n" + "=" * 70)
print("批量实验完成！")
print("=" * 70)
