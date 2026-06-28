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


# ── 注册表：名字 → 构造器 ─────────────────────────────────────────────────────
BASELINE_REGISTRY = {
    # 免训练（零样本）
    "Naive":         lambda **kw: NaiveForecaster(),
    "SeasonalNaive": lambda **kw: SeasonalNaiveForecaster(period=kw.get("period", 24)),
    # ETS/Theta 的季节周期同样按频率传入（1h→24，15min→96，5min→288）
    "ETS":           lambda **kw: ETSForecaster(season=kw.get("period", 24)),
    "Theta":         lambda **kw: ThetaForecaster(season=kw.get("period", 24)),
    # 需训练
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
