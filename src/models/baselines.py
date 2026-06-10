"""
Baseline 电价预测模型（机器学习版本）
====================================
读取 ENLITEN LMP 数据，用实用的机器学习模型对选定节点做预测，
结果保存为 CSV 供对比使用。

运行前准备：
  1. 先运行 prepare.py 生成 ../../data/processed/lmp_processed.csv
  2. 激活虚拟环境并安装依赖：
     pip install xgboost lightgbm scikit-learn tensorflow

运行方式：
  python baselines.py

输出文件：
  ../../data/results/forecast_baselines.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── 路径与参数配置 ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "../../data/processed/lmp_processed.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "forecast_baselines.csv")

PRED_LEN = 24      # 预测未来 24 小时
N_NODES = 5        # 取前 N 个节点做预测

print("=" * 60)
print("Baseline 电价预测模型（机器学习）")
print("=" * 60)

# ── 1. 读取数据 ───────────────────────────────────────────────────────────────
print(f"\n读取数据：{INPUT_CSV}")
if not os.path.exists(INPUT_CSV):
    print("❌ 找不到预处理数据，请先运行 prepare_lmp.py")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, index_col="timestamp", parse_dates=True)
print(f"数据形状：{df.shape}  （{df.shape[0]} 小时 × {df.shape[1]} 节点）")

# 按标准差从大到小排序，选取价格波动最丰富的 N_NODES 个节点
node_stds = df.std()
node_cols = node_stds.nlargest(N_NODES).index.tolist()
print(f"选取节点（按波动性排序）：{node_cols}")

# 准备训练数据 - 使用除最后24小时外的所有历史数据
train_df = df[node_cols].iloc[:-PRED_LEN]
actual_df = df[node_cols].iloc[-PRED_LEN:]

print(f"\n训练数据时间段：{train_df.index[0]}  →  {train_df.index[-1]}")
print(f"预测时间段  ：{actual_df.index[0]}  →  {actual_df.index[-1]}")

# ── 2. 定义 Baseline 模型 ─────────────────────────────────────────────────────

class NaiveForecaster:
    """Naive (随机游走) 预测器"""
    name = "Naive"

    def fit_predict(self, train_data, pred_len):
        last_value = train_data[-1]
        mean = np.full(pred_len, last_value)
        q10 = mean * 0.9
        q90 = mean * 1.1
        return mean, q10, q90


class SeasonalNaiveForecaster:
    """季节性 Naive 预测器（24小时周期）"""
    name = "Seasonal-Naive"

    def fit_predict(self, train_data, pred_len):
        seasonal_pattern = train_data[-24:]
        repeats = (pred_len // 24) + 1
        mean = np.tile(seasonal_pattern, repeats)[:pred_len]
        q10 = mean * 0.9
        q90 = mean * 1.1
        return mean, q10, q90


class RandomForestForecaster:
    """随机森林预测器"""
    name = "RandomForest"

    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        self.rf = RandomForestRegressor
        self.model = None

    def _create_features(self, values):
        """为序列创建特征矩阵"""
        n = len(values)
        features = {}

        # 滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            feat = np.full(n, np.nan)
            feat[lag:] = values[:-lag]
            features[f'lag_{lag}h'] = feat

        # 滚动统计特征
        feat_mean_24 = np.full(n, np.nan)
        feat_std_24 = np.full(n, np.nan)
        feat_mean_168 = np.full(n, np.nan)

        for i in range(24, n):
            feat_mean_24[i] = np.mean(values[i-24:i])
            feat_std_24[i] = np.std(values[i-24:i])

        for i in range(168, n):
            feat_mean_168[i] = np.mean(values[i-168:i])

        features['rolling_mean_24h'] = feat_mean_24
        features['rolling_std_24h'] = feat_std_24
        features['rolling_mean_168h'] = feat_mean_168

        # 差分特征
        feat_diff_1 = np.full(n, np.nan)
        feat_diff_24 = np.full(n, np.nan)
        feat_diff_1[1:] = values[1:] - values[:-1]
        feat_diff_24[24:] = values[24:] - values[:-24]
        features['diff_1h'] = feat_diff_1
        features['diff_24h'] = feat_diff_24

        # 时间特征（使用最后24小时的循环）
        hours = np.arange(n) % 24
        features['hour'] = hours
        features['is_night'] = ((hours >= 22) | (hours <= 6)).astype(int)
        features['is_peak'] = ((hours >= 9) & (hours <= 21)).astype(int)

        return pd.DataFrame(features)

    def fit_predict(self, train_data, pred_len):
        # 使用最近2000小时的数据
        train_values = train_data[-2000:]

        features = self._create_features(train_values)
        features['target'] = train_values

        # 删除缺失值
        data = features.dropna()
        if len(data) < 100:
            return None, None, None

        X = data.drop('target', axis=1).values
        y = data['target'].values

        self.model = self.rf(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # 多步预测（递归预测）
        predictions = []
        current_values = list(train_values)

        for _ in range(pred_len):
            feat_df = self._create_features(np.array(current_values))
            feat_row = feat_df.iloc[[-1]].dropna(axis=1)

            if feat_row.isnull().any().any() or len(feat_row.columns) == 0:
                pred = current_values[-1]
            else:
                pred = self.model.predict(feat_row.values)[0]

            predictions.append(pred)
            current_values.append(pred)

        mean = np.array(predictions)
        residuals = y - self.model.predict(X)
        std = np.std(residuals)
        q10 = mean - 1.28 * std
        q90 = mean + 1.28 * std

        return mean, q10, q90


class MLPForecaster:
    """多层感知机（MLP）预测器"""
    name = "MLP"

    def __init__(self):
        from sklearn.neural_network import MLPRegressor
        self.mlp = MLPRegressor
        self.model = None

    def _create_features(self, values):
        """为序列创建特征矩阵"""
        n = len(values)
        features = {}

        for lag in [1, 2, 3, 6, 12, 24]:
            feat = np.full(n, np.nan)
            feat[lag:] = values[:-lag]
            features[f'lag_{lag}h'] = feat

        feat_mean_24 = np.full(n, np.nan)
        feat_std_24 = np.full(n, np.nan)
        for i in range(24, n):
            feat_mean_24[i] = np.mean(values[i-24:i])
            feat_std_24[i] = np.std(values[i-24:i])

        features['rolling_mean_24h'] = feat_mean_24
        features['rolling_std_24h'] = feat_std_24

        feat_diff_1 = np.full(n, np.nan)
        feat_diff_1[1:] = values[1:] - values[:-1]
        features['diff_1h'] = feat_diff_1

        hours = np.arange(n) % 24
        features['hour'] = hours

        return pd.DataFrame(features)

    def fit_predict(self, train_data, pred_len):
        train_values = train_data[-2000:]

        features = self._create_features(train_values)
        features['target'] = train_values

        data = features.dropna()
        if len(data) < 100:
            return None, None, None

        X = data.drop('target', axis=1).values
        y = data['target'].values

        self.model = self.mlp(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        self.model.fit(X, y)

        predictions = []
        current_values = list(train_values)

        for _ in range(pred_len):
            feat_df = self._create_features(np.array(current_values))
            feat_row = feat_df.iloc[[-1]].dropna(axis=1)

            if feat_row.isnull().any().any() or len(feat_row.columns) == 0:
                pred = current_values[-1]
            else:
                pred = self.model.predict(feat_row.values)[0]

            predictions.append(pred)
            current_values.append(pred)

        mean = np.array(predictions)
        residuals = y - self.model.predict(X)
        std = np.std(residuals)
        q10 = mean - 1.28 * std
        q90 = mean + 1.28 * std

        return mean, q10, q90


class XGBoostForecaster:
    """XGBoost 预测器"""
    name = "XGBoost"

    def __init__(self):
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = None
        except ImportError:
            print("  ⚠️ XGBoost 未安装，跳过")
            self.model = None

    def _create_features(self, values):
        """为序列创建特征矩阵"""
        n = len(values)
        features = {}

        for lag in [1, 2, 3, 6, 12, 24]:
            feat = np.full(n, np.nan)
            feat[lag:] = values[:-lag]
            features[f'lag_{lag}h'] = feat

        feat_mean_24 = np.full(n, np.nan)
        feat_mean_168 = np.full(n, np.nan)
        feat_std_24 = np.full(n, np.nan)

        for i in range(24, n):
            feat_mean_24[i] = np.mean(values[i-24:i])
            feat_std_24[i] = np.std(values[i-24:i])

        for i in range(168, n):
            feat_mean_168[i] = np.mean(values[i-168:i])

        features['rolling_mean_24h'] = feat_mean_24
        features['rolling_mean_168h'] = feat_mean_168
        features['rolling_std_24h'] = feat_std_24

        feat_diff_1 = np.full(n, np.nan)
        feat_diff_24 = np.full(n, np.nan)
        feat_diff_1[1:] = values[1:] - values[:-1]
        feat_diff_24[24:] = values[24:] - values[:-24]
        features['diff_1h'] = feat_diff_1
        features['diff_24h'] = feat_diff_24

        hours = np.arange(n) % 24
        features['hour'] = hours
        features['is_night'] = ((hours >= 22) | (hours <= 6)).astype(int)

        return pd.DataFrame(features)

    def fit_predict(self, train_data, pred_len):
        if self.model is None:
            return None, None, None

        train_values = train_data[-2000:]

        features = self._create_features(train_values)
        features['target'] = train_values

        data = features.dropna()
        if len(data) < 100:
            return None, None, None

        X = data.drop('target', axis=1).values
        y = data['target'].values

        self.model = self.xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        predictions = []
        current_values = list(train_values)

        for _ in range(pred_len):
            feat_df = self._create_features(np.array(current_values))
            feat_row = feat_df.iloc[[-1]].dropna(axis=1)

            if feat_row.isnull().any().any() or len(feat_row.columns) == 0:
                pred = current_values[-1]
            else:
                pred = self.model.predict(feat_row.values)[0]

            predictions.append(pred)
            current_values.append(pred)

        mean = np.array(predictions)
        residuals = y - self.model.predict(X)
        std = np.std(residuals)
        q10 = mean - 1.28 * std
        q90 = mean + 1.28 * std

        return mean, q10, q90


class LightGBMForecaster:
    """LightGBM 预测器"""
    name = "LightGBM"

    def __init__(self):
        try:
            import lightgbm as lgb
            self.lgb = lgb
            self.model = None
        except ImportError:
            print("  ⚠️ LightGBM 未安装，跳过")
            self.model = None

    def _create_features(self, values):
        """为序列创建特征矩阵"""
        n = len(values)
        features = {}

        for lag in [1, 2, 3, 6, 12, 24]:
            feat = np.full(n, np.nan)
            feat[lag:] = values[:-lag]
            features[f'lag_{lag}h'] = feat

        feat_mean_24 = np.full(n, np.nan)
        feat_mean_168 = np.full(n, np.nan)
        feat_std_24 = np.full(n, np.nan)

        for i in range(24, n):
            feat_mean_24[i] = np.mean(values[i-24:i])
            feat_std_24[i] = np.std(values[i-24:i])

        for i in range(168, n):
            feat_mean_168[i] = np.mean(values[i-168:i])

        features['rolling_mean_24h'] = feat_mean_24
        features['rolling_mean_168h'] = feat_mean_168
        features['rolling_std_24h'] = feat_std_24

        feat_diff_1 = np.full(n, np.nan)
        feat_diff_24 = np.full(n, np.nan)
        feat_diff_1[1:] = values[1:] - values[:-1]
        feat_diff_24[24:] = values[24:] - values[:-24]
        features['diff_1h'] = feat_diff_1
        features['diff_24h'] = feat_diff_24

        hours = np.arange(n) % 24
        features['hour'] = hours
        features['is_night'] = ((hours >= 22) | (hours <= 6)).astype(int)

        return pd.DataFrame(features)

    def fit_predict(self, train_data, pred_len):
        if self.model is None:
            return None, None, None

        train_values = train_data[-2000:]

        features = self._create_features(train_values)
        features['target'] = train_values

        data = features.dropna()
        if len(data) < 100:
            return None, None, None

        X = data.drop('target', axis=1).values
        y = data['target'].values

        self.model = self.lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X, y)

        predictions = []
        current_values = list(train_values)

        for _ in range(pred_len):
            feat_df = self._create_features(np.array(current_values))
            feat_row = feat_df.iloc[[-1]].dropna(axis=1)

            if feat_row.isnull().any().any() or len(feat_row.columns) == 0:
                pred = current_values[-1]
            else:
                pred = self.model.predict(feat_row.values)[0]

            predictions.append(pred)
            current_values.append(pred)

        mean = np.array(predictions)
        residuals = y - self.model.predict(X)
        std = np.std(residuals)
        q10 = mean - 1.28 * std
        q90 = mean + 1.28 * std

        return mean, q10, q90


class LSTMForecaster:
    """LSTM 预测器"""
    name = "LSTM"

    def __init__(self):
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
            self.tf = tf
            self.available = True
        except ImportError:
            print("  ⚠️ TensorFlow 未安装，跳过 LSTM")
            self.available = False

    def fit_predict(self, train_data, pred_len):
        if not self.available:
            return None, None, None

        # 准备序列数据
        seq_length = 48
        values = train_data[-4000:]  # 使用最近4000小时

        X, y = [], []
        for i in range(seq_length, len(values)):
            X.append(values[i-seq_length:i])
            y.append(values[i])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 100:
            return None, None, None

        # 归一化
        mean = np.mean(values)
        std = np.std(values)
        X_norm = (X - mean) / (std + 1e-6)
        y_norm = (y - mean) / (std + 1e-6)

        # 构建 LSTM 模型
        model = self.tf.keras.Sequential([
            self.tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.LSTM(50),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(
            X_norm.reshape(-1, seq_length, 1),
            y_norm,
            epochs=30,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )

        # 多步预测
        predictions = []
        current_seq = values[-seq_length:].copy()

        for _ in range(pred_len):
            seq_norm = (current_seq - mean) / (std + 1e-6)
            pred_norm = model.predict(seq_norm.reshape(1, seq_length, 1), verbose=0)[0, 0]
            pred = pred_norm * std + mean
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred

        mean_pred = np.array(predictions)
        q10 = mean_pred - 0.5 * std
        q90 = mean_pred + 0.5 * std

        return mean_pred, q10, q90


class GRUForecaster:
    """GRU 预测器"""
    name = "GRU"

    def __init__(self):
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
            self.tf = tf
            self.available = True
        except ImportError:
            print("  ⚠️ TensorFlow 未安装，跳过 GRU")
            self.available = False

    def fit_predict(self, train_data, pred_len):
        if not self.available:
            return None, None, None

        seq_length = 48
        values = train_data[-4000:]

        X, y = [], []
        for i in range(seq_length, len(values)):
            X.append(values[i-seq_length:i])
            y.append(values[i])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 100:
            return None, None, None

        mean = np.mean(values)
        std = np.std(values)
        X_norm = (X - mean) / (std + 1e-6)
        y_norm = (y - mean) / (std + 1e-6)

        model = self.tf.keras.Sequential([
            self.tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.GRU(50),
            self.tf.keras.layers.Dropout(0.2),
            self.tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(
            X_norm.reshape(-1, seq_length, 1),
            y_norm,
            epochs=30,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )

        predictions = []
        current_seq = values[-seq_length:].copy()

        for _ in range(pred_len):
            seq_norm = (current_seq - mean) / (std + 1e-6)
            pred_norm = model.predict(seq_norm.reshape(1, seq_length, 1), verbose=0)[0, 0]
            pred = pred_norm * std + mean
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred

        mean_pred = np.array(predictions)
        q10 = mean_pred - 0.5 * std
        q90 = mean_pred + 0.5 * std

        return mean_pred, q10, q90


# ── 3. 初始化所有 Baseline 模型 ───────────────────────────────────────────────
forecasters = [
    NaiveForecaster(),
    SeasonalNaiveForecaster(),
    RandomForestForecaster(),
    MLPForecaster(),
    XGBoostForecaster(),
    LightGBMForecaster(),
    LSTMForecaster(),
    GRUForecaster(),
]

# 过滤掉不可用的模型
available_forecasters = []
for f in forecasters:
    if not hasattr(f, 'available') or f.available:
        available_forecasters.append(f)
    elif hasattr(f, 'model') and f.model is None:
        pass  # 跳过未安装依赖的模型
    else:
        available_forecasters.append(f)

forecasters = available_forecasters

print(f"\n将运行 {len(forecasters)} 个 baseline 模型：")
for f in forecasters:
    print(f"  - {f.name}")

# ── 4. 运行预测 ───────────────────────────────────────────────────────────────
print(f"\n预测未来 {PRED_LEN} 小时（{N_NODES} 个节点）...")

all_results = []

for forecaster in forecasters:
    print(f"\n  运行 {forecaster.name}...")

    for i, node in enumerate(node_cols):
        # 获取训练数据
        train_data = train_df[node].values.astype(np.float32)

        # 预测
        mean, q10, q90 = forecaster.fit_predict(train_data, PRED_LEN)

        if mean is None:
            print(f"    ⚠️ {node} 预测失败，跳过")
            continue

        # 整理结果
        pred_index = actual_df.index
        for t in range(PRED_LEN):
            all_results.append({
                "timestamp": pred_index[t],
                "node": node,
                "model": forecaster.name,
                "actual": actual_df[node].iloc[t],
                "mean": float(mean[t]),
                "q10": float(q10[t]),
                "q90": float(q90[t]),
            })

    print(f"    ✅ {forecaster.name} 完成")

print("\n✅ 全部 baseline 预测完成")

# ── 5. 整理输出 ───────────────────────────────────────────────────────────────
result_df = pd.DataFrame(all_results)
result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ 结果已保存：{OUTPUT_CSV}")
print(f"   预测节点数：{N_NODES}")
print(f"   预测步数  ：{PRED_LEN}")
print(f"   Baseline 模型数：{len(forecasters)}")

# 打印各模型的预测均值统计
print(f"\n各模型预测均值统计（$/MWh）：")
for forecaster in forecasters:
    model_data = result_df[result_df["model"] == forecaster.name]
    if len(model_data) > 0:
        pred_mean = model_data["mean"].mean()
        actual_mean = model_data["actual"].mean()
        mae = np.mean(np.abs(model_data["actual"] - model_data["mean"]))
        print(f"  {forecaster.name:20s}：预测均值 {pred_mean:7.2f}，实际均值 {actual_mean:7.2f}，MAE {mae:.3f}")

print("\n" + "=" * 60)
print("✅ Baseline 预测完成！")
print("=" * 60)
