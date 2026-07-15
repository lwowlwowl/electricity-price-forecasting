"""
基线预测器适配器 forecasters.py
================================
把常用基线统一封装成 Forecaster 接口，供滚动回测直接调用。
这些基线既是"对照下界"（任何高级模型都该打败它们），也是 MASE 的分母，
更是把整条评估路径跑通的最轻依赖实现。

包含两类基线：

免训练（零样本统计基线，needs_training=False）：
  - NaiveForecaster        : 随机游走，预测=最后一个观测值
  - SeasonalNaiveForecaster: 季节性朴素，预测=上一个周期同相位的值（日前预测常用强基线）
  - ETSForecaster          : 指数平滑 / Holt-Winters（statsmodels）。M4 竞赛官方强基线，
                             能抓日内季节性。参考 Hyndman & Athanasopoulos, FPP。
  - ThetaForecaster        : Theta 法（statsmodels）。M3 竞赛冠军法
                             (Assimakopoulos & Nikolopoulos, 2000)，极简但出奇地强，
                             几乎所有时序 benchmark 都拿它当对照。

需训练（每个起报点用历史重新 fit，无泄漏，needs_training=True）：
  - RandomForestForecaster : 随机森林（sklearn）
  - LightGBMForecaster     : LightGBM 梯度提升树
  - XGBoostForecaster      : XGBoost 梯度提升树
  三者共用一套电价特征工程（多阶 lag / 滚动均值方差 / 差分 / 小时-峰谷），
  递归多步预测，残差 std 估分位。树模型在电价预测里是公认强基线。

基础模型（TimesFM / Chronos-2 / Toto）的适配器放在 foundation.py，
它们同样继承 Forecaster，可与这里的基线在同一回测里公平对比。
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from base import Forecaster, Forecast


# Z 分数：标准正态 10% / 90% 分位（各 predict 内联用残差 std 估分位带）
_Z = 1.2816


class NaiveForecaster(Forecaster):
    """随机游走：未来每一步都等于最后一个观测值。"""

    name = "Naive"
    needs_training = False
    supports_covariates = False
    # supports_multivariate=False：基线逐列独立预测，节点间零交互，不是多变量
    # 联合建模。多变量实验里基线始终按逐列处理，multivariate_used 应为 False。
    supports_multivariate = False

    def predict(self, context_df, future_covariates=None, horizon=24) -> Forecast:
        cols = self._target_columns(context_df)
        hist = context_df[cols].to_numpy(dtype=float)        # (T, n_series)
        last = hist[-1]                                       # (n_series,)
        mean = np.tile(last, (horizon, 1))                   # (horizon, n_series)

        # 用一阶差分的标准差估计不确定性，随预测步长累积（随机游走方差线性增长）
        diff_std = np.nanstd(np.diff(hist, axis=0), axis=0)
        steps = np.sqrt(np.arange(1, horizon + 1))[:, None]
        band = diff_std[None, :] * steps
        q10, q50, q90 = mean - _Z * band, mean.copy(), mean + _Z * band

        return self._pack(mean, q10, q50, q90, context_df, horizon, cols)

    def _pack(self, mean, q10, q50, q90, context_df, horizon, cols):
        """统一打包：单列时压成一维，多列保留二维。"""
        idx = self._future_index(context_df, horizon)
        if mean.shape[1] == 1:
            return Forecast(mean[:, 0], q10[:, 0], q50[:, 0], q90[:, 0],
                            index=idx, series_names=cols)
        return Forecast(mean, q10, q50, q90, index=idx, series_names=cols)


class SeasonalNaiveForecaster(NaiveForecaster):
    """
    季节性朴素：未来第 h 步 = 历史中"上一个周期同相位"的值。
    电价日前预测里，period=24（小时频率）非常强，是最常用的强基线，也是 MASE 分母。
    """

    name = "SeasonalNaive"

    def __init__(self, period: int = 24):
        self.period = period

    def predict(self, context_df, future_covariates=None, horizon=24) -> Forecast:
        cols = self._target_columns(context_df)
        hist = context_df[cols].to_numpy(dtype=float)        # (T, n_series)
        p = self.period
        if hist.shape[0] < p:
            # 历史不足一个周期，退化为 Naive
            return NaiveForecaster().predict(context_df, future_covariates, horizon)

        last_period = hist[-p:]                               # (p, n_series)
        idx_in_period = np.arange(horizon) % p
        mean = last_period[idx_in_period]                    # (horizon, n_series)

        # 残差：历史相对于"上一周期"的偏差
        if hist.shape[0] >= 2 * p:
            resid = hist[p:] - hist[:-p]
            resid_std = np.nanstd(resid, axis=0)
        else:
            resid_std = np.nanstd(np.diff(hist, axis=0), axis=0)
        band = resid_std[None, :]
        q10, q50, q90 = mean - _Z * band, mean.copy(), mean + _Z * band

        return self._pack(mean, q10, q50, q90, context_df, horizon, cols)


# ══════════════════════════════════════════════════════════════════════════════
#  免训练统计基线：ETS / Theta（statsmodels，逐列拟合）
# ══════════════════════════════════════════════════════════════════════════════
class _StatForecaster(NaiveForecaster):
    """
    statsmodels 单变量统计基线的共用骨架：逐列 fit→forecast，失败时退化为
    SeasonalNaive，并用历史残差近似分位带。子类只需实现 `_fit_forecast_1d`。
    """

    name = "_Stat"
    needs_training = False
    # 逐列独立拟合，非多变量联合建模 → supports_multivariate=False
    supports_multivariate = False

    # 拟合用的最大历史长度（电价高频数据，截断以控速）
    MAX_CONTEXT = 2000

    def __init__(self, season: int = 24):
        # SEASON 作为实例变量，允许按频率传入（1h→24，15min→96，5min→288）
        self.SEASON = season

    def _fit_forecast_1d(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """对单条一维序列拟合并预测，返回长度 horizon 的点预测。"""
        raise NotImplementedError

    def predict(self, context_df, future_covariates=None, horizon=24) -> Forecast:
        cols = self._target_columns(context_df)
        hist = context_df[cols].to_numpy(dtype=float)        # (T, n_series)
        n_series = hist.shape[1]

        means = np.empty((horizon, n_series))
        stds = np.empty(n_series)
        for j in range(n_series):
            col = hist[:, j]
            col = col[~np.isnan(col)]
            col = col[-self.MAX_CONTEXT:]
            try:
                if col.size < self.SEASON + 2:
                    raise ValueError("历史太短")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = self._fit_forecast_1d(col, horizon)
                if pred is None or not np.all(np.isfinite(pred)):
                    raise ValueError("预测含非有限值")
                means[:, j] = pred
                # 用季节残差估不确定性（与 SeasonalNaive 一致口径）
                if col.size >= 2 * self.SEASON:
                    resid = col[self.SEASON:] - col[:-self.SEASON]
                else:
                    resid = np.diff(col)
                stds[j] = np.nanstd(resid) if resid.size else 0.0
            except Exception:
                # 任何拟合失败都退化为季节朴素，保证回测不中断
                fb = self._seasonal_naive_1d(col, horizon)
                means[:, j] = fb
                stds[j] = np.nanstd(np.diff(col)) if col.size >= 2 else 0.0

        band = stds[None, :]
        q10, q50, q90 = means - _Z * band, means.copy(), means + _Z * band
        return self._pack(means, q10, q50, q90, context_df, horizon, cols)

    def _seasonal_naive_1d(self, col: np.ndarray, horizon: int) -> np.ndarray:
        p = self.SEASON
        if col.size >= p:
            last_period = col[-p:]
            return last_period[np.arange(horizon) % p]
        last = col[-1] if col.size else 0.0
        return np.full(horizon, last)


class ETSForecaster(_StatForecaster):
    """
    指数平滑 / Holt-Winters（statsmodels ExponentialSmoothing）。
    带加性趋势 + 加性日内季节(period=24)。M4 竞赛官方强基线之一。
    """

    name = "ETS"

    def _fit_forecast_1d(self, series, horizon):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        # 季节性需要至少两个完整周期；不足则去掉季节项
        seasonal = "add" if series.size >= 2 * self.SEASON else None
        sp = self.SEASON if seasonal else None
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=sp,
            initialization_method="estimated",
        )
        fit = model.fit()
        return np.asarray(fit.forecast(horizon), dtype=float)


class ThetaForecaster(_StatForecaster):
    """
    Theta 法（statsmodels ThetaModel）。M3 竞赛冠军法，极简但很强。
    period=24 抓日内季节性。
    """

    name = "Theta"

    def _fit_forecast_1d(self, series, horizon):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        period = self.SEASON if series.size >= 2 * self.SEASON else 1
        model = ThetaModel(series, period=period if period > 1 else None)
        fit = model.fit()
        return np.asarray(fit.forecast(horizon), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  需训练基线：树模型（RandomForest / LightGBM / XGBoost）
#  逐列、每起报点用 context_df 之前的历史重新 fit → 无泄漏；递归多步预测。
# ══════════════════════════════════════════════════════════════════════════════
class _TreeForecaster(NaiveForecaster):
    """
    树模型回归的共用骨架。特征工程：
    多阶 lag / 滚动均值方差 / 差分 / 小时-峰谷。子类只需实现 `_make_model`。

    无泄漏保证：predict 只接触 context_df（起报点之前的历史），在其上 fit，
    再递归外推 horizon 步。每个起报点都会重新训练。
    """

    name = "_Tree"
    needs_training = True
    supports_covariates = False
    # 逐列独立训练，非多变量联合建模 → supports_multivariate=False
    supports_multivariate = False

    MAX_CONTEXT = 2000   # 每列最多取最近这么多小时训练，控速
    MIN_ROWS = 100       # 去掉缺失后至少这么多行才训练，否则退化

    def _make_model(self):
        """返回一个未拟合的 sklearn 风格回归器（有 fit/predict）。"""
        raise NotImplementedError

    # ── 特征工程（向量化的滚动窗口以提速）─────────────────────
    @staticmethod
    def _create_features(values: np.ndarray, hour0: int = 0) -> pd.DataFrame:
        """
        生成特征矩阵（多阶 lag / 滚动均值方差 / 差分 / 小时-峰谷）。用 pandas
        向量化滚动窗口替代 Python for 循环，避免在长训练窗口 + 递归多步预测下
        的性能爆炸。

        hour0：本段序列第 0 个点对应的小时相位（默认 0）。递归预测时只对尾部
        窗口重算特征，靠它保持小时相位连续。

        注意：滚动统计用 shift(1) 严格只看【过去】窗口（不含当前点），口径与
        原实现 values[i-24:i] 完全一致 → 无未来信息泄露。
        """
        s = pd.Series(np.asarray(values, dtype=float))
        n = len(s)
        feats = {}

        for lag in (1, 2, 3, 6, 12, 24):
            feats[f"lag_{lag}h"] = s.shift(lag)

        # rolling 窗口取当前点之前的 w 个值：先 shift(1) 再 rolling(w)
        prev = s.shift(1)
        feats["rolling_mean_24h"] = prev.rolling(24).mean()
        feats["rolling_std_24h"] = prev.rolling(24).std(ddof=0)
        feats["rolling_mean_168h"] = prev.rolling(168).mean()

        feats["diff_1h"] = s.diff(1)
        feats["diff_24h"] = s.diff(24)

        hours = (np.arange(n) + hour0) % 24
        feats["hour"] = pd.Series(hours, dtype=float)
        feats["is_night"] = pd.Series(((hours >= 22) | (hours <= 6)).astype(float))
        feats["is_peak"] = pd.Series(((hours >= 9) & (hours <= 21)).astype(float))

        return pd.DataFrame(feats)

    def _fit_forecast_1d(self, values: np.ndarray, horizon: int):
        """在单列历史上训练并递归预测 horizon 步；失败返回 None。"""
        values = values[-self.MAX_CONTEXT:]
        feats = self._create_features(values)
        feats["target"] = values
        data = feats.dropna()
        if len(data) < self.MIN_ROWS:
            return None, None

        feat_cols = [c for c in data.columns if c != "target"]
        X = data[feat_cols].to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=float)

        model = self._make_model()
        model.fit(X, y)

        # 残差 std（样本内）用于估分位带
        resid_std = float(np.std(y - model.predict(X)))

        # 递归多步预测：每步只对尾部窗口重算特征（覆盖最长依赖 168h 即可），
        # 避免对整段历史重复算特征。tail_len 需 > 168（rolling_mean_168h）。
        tail_len = 200
        preds = []
        cur = list(values[-tail_len:])
        # 尾窗第 0 个点的小时相位：跟随原序列相位
        base_phase = (len(values) - len(cur)) % 24
        for step in range(horizon):
            frow = self._create_features(
                np.asarray(cur, dtype=float), hour0=base_phase).iloc[-1]
            xrow = frow[feat_cols].to_numpy(dtype=float).reshape(1, -1)
            if np.isnan(xrow).any():
                p = cur[-1]
            else:
                p = float(model.predict(xrow)[0])
            preds.append(p)
            cur.append(p)
            # 保持尾窗长度恒定，相位随之前移
            if len(cur) > tail_len:
                cur.pop(0)
                base_phase = (base_phase + 1) % 24
        return np.asarray(preds, dtype=float), resid_std

    def predict(self, context_df, future_covariates=None, horizon=24) -> Forecast:
        cols = self._target_columns(context_df)
        hist = context_df[cols].to_numpy(dtype=float)        # (T, n_series)
        n_series = hist.shape[1]

        means = np.empty((horizon, n_series))
        stds = np.empty(n_series)
        for j in range(n_series):
            col = hist[:, j]
            col = col[~np.isnan(col)]
            pred, resid_std = (None, None)
            if col.size >= self.MIN_ROWS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred, resid_std = self._fit_forecast_1d(col, horizon)
                except ImportError as e:
                    # 依赖未安装（如 lightgbm/xgboost），只打印一次
                    if not getattr(self, "_import_error_shown", False):
                        print(f"  ⚠️  {self.name} 依赖未安装（{e}）→ 退化为 SeasonalNaive")
                        self._import_error_shown = True
                    pred = None
                except Exception:
                    pred = None
            if pred is None or not np.all(np.isfinite(pred)):
                # 历史不足或训练失败 → 退化为季节朴素，保证回测不中断
                pred = self._seasonal_naive_1d(col, horizon)
                resid_std = np.nanstd(np.diff(col)) if col.size >= 2 else 0.0
            means[:, j] = pred
            stds[j] = resid_std if resid_std is not None else 0.0

        band = stds[None, :]
        q10, q50, q90 = means - _Z * band, means.copy(), means + _Z * band
        return self._pack(means, q10, q50, q90, context_df, horizon, cols)

    def _seasonal_naive_1d(self, col, horizon):
        p = 24
        if col.size >= p:
            last_period = col[-p:]
            return last_period[np.arange(horizon) % p]
        last = col[-1] if col.size else 0.0
        return np.full(horizon, last)


class RandomForestForecaster(_TreeForecaster):
    """随机森林回归基线（sklearn）。"""

    name = "RandomForest"

    def _make_model(self):
        from sklearn.ensemble import RandomForestRegressor
        # n_estimators=60 在速度与质量间折中（每起报点都要重训，回测起报点多）
        return RandomForestRegressor(
            n_estimators=60, max_depth=10, random_state=42, n_jobs=-1
        )


class LightGBMForecaster(_TreeForecaster):
    """LightGBM 梯度提升树基线。"""

    name = "LightGBM"

    def _make_model(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300, max_depth=-1, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )


class XGBoostForecaster(_TreeForecaster):
    """XGBoost 梯度提升树基线。"""

    name = "XGBoost"

    def _make_model(self):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LEAR baseline（Lasso Estimated AutoRegressive，Lago 2021）
#  对应 Lago checklist #2/#13：开源 SOTA 基线 + epftoolbox 工具
# ══════════════════════════════════════════════════════════════════════════════
class LEARForecaster(NaiveForecaster):
    """
    LEAR（Lasso Estimated AutoRegressive）日前电价预测基线。

    实现依据：Lago et al. (2021) Applied Energy 293, Section 4.2
    代码基础：epftoolbox (https://github.com/jeslago/epftoolbox) LEAR 类

    特征结构（对应 Lago Eq. 3）：
      [p_{d-1,0:24}, p_{d-2,0:24}, p_{d-3,0:24}, p_{d-7,0:24}, z_{d,0:7}]
      ─ 4 天价格 lag（4×24=96 维）＋ 7 维星期哑变量 = 103 特征
      ─ 对每个小时 h=0..23 独立拟合一个 LASSO 模型（共 24 个）
      ─ lambda 每次用 LassoLarsIC (AIC) 选定（高效、准确）
      ─ 价格做 asinh-median Invariant 变换后再估计（epftoolbox 内部处理）

    回测用法（BacktestConfig 推荐）：
      needs_training = True → backtest 使用 train_context_len 作为历史长度
      建议 train_context_len = 84 * 24 = 2016（12 周滚动窗口）
        → context_len 必须 ≥ (min_calib_days + 7) * 24
      LEAR 不支持协变量（当前版本）：supports_covariates = False

    定位说明（TODO 4d / 2d）：
      参数消融 / 结构消融中：LEAR 是"外部定标参照"，不与消融配置并列排名
      ElecFM 评估中：LEAR 是必须超越的基准，做 DM/GW test 对比
    """

    name = "LEAR"
    needs_training = True         # 每日 recalibrate → 对 backtest 引擎：needs_training=True
    supports_covariates = False   # 当前版本仅使用价格 lag（Lago Eq. 3 基本版）
    supports_multivariate = False # 逐节点独立预测

    #: 最小校准天数（Lago §5.1：≥ 8 周）
    MIN_CALIB_DAYS: int = 56

    def _to_daily(self, series_1d: np.ndarray,
                  index: pd.DatetimeIndex) -> np.ndarray:
        """
        把 1-D 小时序列按 UTC 日期重组成 [n_days, 24] 矩阵。

        只保留完整的 24 小时天（不完整天丢弃），从最早的完整天开始。
        返回 None 若有效天数不足 MIN_CALIB_DAYS + 7（无法构建特征）。
        """
        dates = pd.DatetimeIndex(index).normalize()          # UTC 日期（无时分秒）
        days, first_idx = [], None
        for date, group in pd.Series(series_1d, index=index).groupby(dates):
            if len(group) == 24:                              # 跳过不完整天
                days.append(group.to_numpy(dtype=float))
        if len(days) < self.MIN_CALIB_DAYS + 7:
            return None
        return np.stack(days, axis=0)                        # [n_days, 24]

    @staticmethod
    def _dow_dummies(prices_daily: np.ndarray,
                     ref_index: pd.DatetimeIndex) -> np.ndarray:
        """
        生成 [n_days, 7] 的星期哑变量矩阵（epftoolbox 特征格式末 7 列）。
        ref_index：与 prices_daily 行数等长的 DatetimeIndex（每天一个时间戳）。
        """
        n = prices_daily.shape[0]
        dows = ref_index[:n].dayofweek.to_numpy()   # 0=Mon … 6=Sun
        dummies = np.zeros((n, 7), dtype=float)
        for i, d in enumerate(dows):
            dummies[i, d] = 1.0
        return dummies

    @staticmethod
    def _build_lear_matrices(
        prices_daily: np.ndarray,
        dow_dummies: np.ndarray,
        n_calib_days: int,
    ):
        """
        构建 LEAR 特征矩阵。

        prices_daily : [n_days, 24]  — 按日重组的小时价格
        dow_dummies  : [n_days, 7]   — 星期哑变量（Monday=0 … Sunday=6）
        n_calib_days : 校准窗口天数

        返回 (Xtrain, Ytrain, Xtest)
          Xtrain : [n_calib_days, 103]  — 训练特征（96 价格 lag + 7 星期哑变量）
          Ytrain : [n_calib_days, 24]   — 训练目标（各天 24 小时价格）
          Xtest  : [1, 103]             — 测试特征（预测"下一天"用）
        """
        n = prices_daily.shape[0]
        # 构建所有合法天的特征向量（需 d ≥ 7 才有 d-7 lag）
        X_rows, Y_rows, dow_rows = [], [], []
        for d in range(7, n):
            x_price = np.concatenate([
                prices_daily[d - 1],   # d-1，前一天 24h
                prices_daily[d - 2],   # d-2
                prices_daily[d - 3],   # d-3
                prices_daily[d - 7],   # d-7，上周同一天
            ])                          # (96,)
            X_rows.append(x_price)
            Y_rows.append(prices_daily[d])
            dow_rows.append(dow_dummies[d])

        X_price = np.stack(X_rows)    # [n-7, 96]
        Y_all = np.stack(Y_rows)      # [n-7, 24]
        dow_all = np.stack(dow_rows)  # [n-7, 7]
        X_all = np.concatenate([X_price, dow_all], axis=1)   # [n-7, 103]

        # 取最后 n_calib_days 行作为训练集
        Xtrain = X_all[-n_calib_days:]
        Ytrain = Y_all[-n_calib_days:]

        # 测试特征：预测"prices_daily[n]"（即下一天），lag 来自现有最后几天
        x_test_price = np.concatenate([
            prices_daily[-1],   # d-1 = 最后一天（完整 24h）
            prices_daily[-2],   # d-2
            prices_daily[-3],   # d-3
            prices_daily[-7],   # d-7
        ])
        # 下一天星期 = (最后一天星期 + 1) % 7
        last_dow = np.argmax(dow_dummies[n - 1])
        next_dow_dummy = np.zeros(7, dtype=float)
        next_dow_dummy[(last_dow + 1) % 7] = 1.0
        Xtest = np.concatenate([x_test_price, next_dow_dummy]).reshape(1, -1)

        return Xtrain, Ytrain, Xtest

    def predict(self, context_df: pd.DataFrame,
                future_covariates=None, horizon: int = 24) -> "Forecast":
        try:
            from epftoolbox.models import LEAR as _LEAR
        except ImportError:
            warnings.warn("epftoolbox 未安装，LEAR 退化为 SeasonalNaive。"
                          "安装：python -m pip install "
                          "git+https://github.com/jeslago/epftoolbox.git")
            return SeasonalNaiveForecaster(period=24).predict(
                context_df, future_covariates, horizon)

        # LEAR 是日前（24h）预测器，不支持其他 horizon：退化为 SeasonalNaive
        if horizon != 24:
            warnings.warn(
                f"LEAR 仅支持 horizon=24（日前预测），"
                f"收到 horizon={horizon}，退化为 SeasonalNaive。",
                stacklevel=2)
            return SeasonalNaiveForecaster(period=24).predict(
                context_df, future_covariates, horizon)

        cols = self._target_columns(context_df)
        n_series = len(cols)

        means = np.empty((horizon, n_series))
        for j, col in enumerate(cols):
            series_1d = context_df[col].to_numpy(dtype=float)
            series_1d_clean = series_1d.copy()
            # 简单线性插值填 NaN（LEAR 不能有 NaN）
            mask = np.isnan(series_1d_clean)
            if mask.any():
                idx = np.arange(len(series_1d_clean))
                series_1d_clean[mask] = np.interp(
                    idx[mask], idx[~mask], series_1d_clean[~mask])

            # 按 UTC 日期重组为 [n_days, 24]
            prices_daily = self._to_daily(series_1d_clean, context_df.index)

            if prices_daily is None:
                # 完整 UTC 日不足（需 ≥ MIN_CALIB_DAYS+7 天），退化为 SeasonalNaive
                if not getattr(self, "_calib_warn_shown", False):
                    dates = pd.DatetimeIndex(context_df.index).normalize()
                    n_complete = sum(
                        1 for _, g in pd.Series(series_1d_clean,
                                                 index=context_df.index).groupby(dates)
                        if len(g) == 24)
                    print(f"  ⚠️  LEAR: 完整 UTC 日数={n_complete} < "
                          f"{self.MIN_CALIB_DAYS + 7}（需增大 train_context_len，"
                          f"当前={len(context_df)}h）→ 退化为 SeasonalNaive")
                    self._calib_warn_shown = True
                fb = SeasonalNaiveForecaster(period=24).predict(
                    context_df[[col]], future_covariates, horizon)
                means[:, j] = (fb.mean[:, 0] if fb.mean.ndim == 2
                               else fb.mean[:horizon])
                continue

            n_days = prices_daily.shape[0]
            # LEAR 特征维度 = 4×24 + 7 = 103，LassoLarsIC 要求 n_samples > n_features
            # 因此校准窗口下限 = 103+10 = 113，上限 130（约 4.5 个月，平衡稳定性与适应性）
            n_calib_days = min(n_days - 7, 130)
            n_calib_days = max(n_calib_days, self.MIN_CALIB_DAYS)

            # 星期哑变量：用 context_df 的 DatetimeIndex 按 UTC 日对齐
            daily_dates = (pd.DatetimeIndex(context_df.index)
                           .normalize().unique().sort_values())
            # 仅保留完整天（与 _to_daily 同口径）
            full_days = [
                d for d in daily_dates
                if context_df.loc[context_df.index.normalize() == d].shape[0] == 24
            ]
            full_day_idx = pd.DatetimeIndex(full_days)
            dow_dummies = self._dow_dummies(prices_daily, full_day_idx)

            try:
                Xtrain, Ytrain, Xtest = self._build_lear_matrices(
                    prices_daily, dow_dummies, n_calib_days)
                lear_model = _LEAR(calibration_window=n_calib_days)
                lear_model.recalibrate(Xtrain, Ytrain)
                # epftoolbox predict() 在新版 numpy 下有 array→scalar 赋值 bug，
                # 直接手动逐小时预测绕开（等价于 recalibrate_predict 但无 bug）
                pred_24 = np.array([
                    float(lear_model.models[h].predict(Xtest)[0])
                    for h in range(24)
                ], dtype=float)
                if not np.all(np.isfinite(pred_24)):
                    raise ValueError("LEAR 预测含非有限值")
                means[:, j] = pred_24[:horizon]
            except Exception as e:
                print(f"  ⚠️  LEAR 预测失败（{col}）：{e}，退化为 SeasonalNaive")
                fb = SeasonalNaiveForecaster(period=24).predict(
                    context_df[[col]], future_covariates, horizon)
                means[:, j] = (fb.mean[:, 0] if fb.mean.ndim == 2
                               else fb.mean[:horizon])

        # LEAR 是点预测器，无分位数（q10/q90 用残差估计作 stub）
        band = np.nanstd(means, axis=0, keepdims=True) * _Z
        q10 = means - band
        q90 = means + band

        return self._pack(means, q10, means.copy(), q90,
                          context_df, horizon, cols)


# ── 注册表：名字 → 构造器 ─────────────────────────────────────────────────────
BASELINE_REGISTRY = {
    # 免训练（零样本）
    "Naive":         lambda **kw: NaiveForecaster(),
    "SeasonalNaive": lambda **kw: SeasonalNaiveForecaster(period=kw.get("period", 24)),
    # ETS/Theta 的季节周期同样按频率传入（1h→24，15min→96，5min→288）
    "ETS":           lambda **kw: ETSForecaster(season=kw.get("period", 24)),
    "Theta":         lambda **kw: ThetaForecaster(season=kw.get("period", 24)),
    # 需训练（每起报点 recalibrate）
    "LEAR":          lambda **kw: LEARForecaster(),        # Lago 2021 开源基准
    "RandomForest":  lambda **kw: RandomForestForecaster(),
    "LightGBM":      lambda **kw: LightGBMForecaster(),
    "XGBoost":       lambda **kw: XGBoostForecaster(),
}


def build_forecaster(name: str, **kwargs) -> Forecaster:
    """按名字构造一个预测器。基础模型(TimesFM/Chronos/Toto)在 foundation.py 注册。"""
    if name in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[name](**kwargs)
    # 延迟导入基础模型，避免没装大模型依赖时也能跑基线
    try:
        from foundation import FOUNDATION_REGISTRY
        if name in FOUNDATION_REGISTRY:
            return FOUNDATION_REGISTRY[name](**kwargs)
    except ImportError:
        pass
    raise ValueError(f"未知模型 {name!r}。可用基线：{list(BASELINE_REGISTRY)}")


# ── 自测 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 造一段带日周期+噪声的假数据测试所有基线
    rng = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    t = np.arange(400)
    daily = 30 + 20 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168)
    daily = daily + np.random.default_rng(0).normal(0, 2, size=400)
    df = pd.DataFrame({"price__TEST": daily}, index=rng)

    print("=" * 64)
    print("基线预测器自测（单节点，horizon=24）")
    print("=" * 64)
    for name in BASELINE_REGISTRY:
        fc = build_forecaster(name).predict(df, horizon=24)
        flag = "train" if build_forecaster(name).needs_training else "zeroshot"
        print(f"{name:14s} [{flag:8s}] mean[:3]={np.round(fc.mean[:3], 2)}  "
              f"有分位数={fc.has_quantiles}  horizon={fc.horizon}")
    print("\n✅ 基线预测器工作正常")
