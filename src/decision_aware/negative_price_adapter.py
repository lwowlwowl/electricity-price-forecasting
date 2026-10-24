"""可审计的两阶段负价适配器。

NEG-E 用稀疏逻辑回归识别负价发生，NEG-A 用稳健线性回归估计条件幅度。
本模块只负责特征、模型和预测变换，不读取2025，也不执行结算。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CALENDAR_FEATURES = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday",
)

FEATURE_NAMES = (
    "da_lag168", "rt_lag168", "load_lag168", "wind_lag168",
    "solar_lag168", "net_load_lag168",
    "da_ctx_median", "da_ctx_mad", "da_ctx_min", "da_ctx_max",
    "da_ctx_negative_share", "da_ctx_last",
    "rt_ctx_median", "rt_ctx_mad", "rt_ctx_min", "rt_ctx_max",
    "rt_ctx_negative_share", "rt_ctx_last",
    "load_ctx_median", "load_ctx_mad", "load_ctx_min", "load_ctx_last",
    "wind_ctx_median", "wind_ctx_mad", "wind_ctx_min", "wind_ctx_last",
    "solar_ctx_median", "solar_ctx_mad", "solar_ctx_min", "solar_ctx_last",
    "net_load_ctx_median", "net_load_ctx_mad", "net_load_ctx_min",
    "net_load_ctx_last",
    *CALENDAR_FEATURES,
)

SEASONAL_EVENT_FEATURE_NAMES = (
    "da_lag168", "da_ctx_negative_share", "da_ctx_median", "da_ctx_mad",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin",
    "month_cos", "is_weekend", "is_holiday",
)
SEASONAL_EVENT_FEATURE_INDICES = tuple(
    FEATURE_NAMES.index(name) for name in SEASONAL_EVENT_FEATURE_NAMES
)


@dataclass(frozen=True)
class AdapterFeatureBundle:
    x: np.ndarray
    target: np.ndarray
    delivery_dates: np.ndarray
    hours: np.ndarray

    def __post_init__(self) -> None:
        rows = self.x.shape[0]
        if self.x.ndim != 2 or self.x.shape[1] != len(FEATURE_NAMES):
            raise ValueError("适配器特征形状错误")
        if any(len(value) != rows for value in (
            self.target, self.delivery_dates, self.hours
        )):
            raise ValueError("适配器特征与标签/元数据长度不一致")
        if not np.isfinite(self.x).all() or not np.isfinite(self.target).all():
            raise ValueError("适配器输入包含非有限值")


def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, float(np.min(values)), float(values[-1])


def extract_da_adapter_features(dataset) -> AdapterFeatureBundle:
    """从 ``DecisionAwareDADataset`` 提取严格符合起报边界的特征。"""
    n_rows = len(dataset.windows) * 24
    x = np.empty((n_rows, len(FEATURE_NAMES)), dtype=np.float32)
    target = np.empty(n_rows, dtype=np.float32)
    dates = np.empty(n_rows, dtype="U10")
    hours = np.tile(np.arange(24, dtype=np.int8), len(dataset.windows))
    net_load = (
        dataset.streams["load"][:, 0]
        - dataset.streams["system"][:, 0]
        - dataset.streams["system"][:, 1]
    )

    cursor = 0
    for window in dataset.windows:
        context = slice(window.context_start, window.context_end)
        positions = window.target_positions
        lag = positions - 168
        if np.any(lag < window.context_start) or np.any(lag >= window.context_end):
            raise ValueError("lag168越过DA起报信息边界")
        delta = (
            dataset.index[positions].as_unit("ns").asi8
            - dataset.index[lag].as_unit("ns").asi8
        )
        if np.any(delta != np.timedelta64(168, "h").astype("timedelta64[ns]").astype(np.int64)):
            raise ValueError("lag168并非严格相隔168小时")

        da = dataset.price_da[context]
        rt = dataset.price_rt[context]
        load = dataset.streams["load"][context, 0]
        wind = dataset.streams["system"][context, 0]
        solar = dataset.streams["system"][context, 1]
        pressure = net_load[context]
        da_summary = (*_summary(da), float(np.mean(da < 0.0)), float(da[-1]))
        rt_summary = (*_summary(rt), float(np.mean(rt < 0.0)), float(rt[-1]))
        # _summary为median/MAD/min/last；价格还需要max，按预注册顺序重排。
        da_values = (
            da_summary[0], da_summary[1], da_summary[2], float(np.max(da)),
            da_summary[4], da_summary[5],
        )
        rt_values = (
            rt_summary[0], rt_summary[1], rt_summary[2], float(np.max(rt)),
            rt_summary[4], rt_summary[5],
        )
        context_values = (
            *da_values, *rt_values, *_summary(load), *_summary(wind),
            *_summary(solar), *_summary(pressure),
        )
        block = np.column_stack((
            dataset.price_da[lag], dataset.price_rt[lag],
            dataset.streams["load"][lag, 0],
            dataset.streams["system"][lag, 0],
            dataset.streams["system"][lag, 1], net_load[lag],
            np.broadcast_to(np.asarray(context_values, dtype=np.float32), (24, 28)),
            dataset.streams["cal"][positions],
        )).astype(np.float32, copy=False)
        if block.shape != (24, len(FEATURE_NAMES)):
            raise RuntimeError(f"内部特征形状错误: {block.shape}")
        x[cursor:cursor + 24] = block
        target[cursor:cursor + 24] = dataset.price_da[positions]
        dates[cursor:cursor + 24] = str(window.delivery_date)
        cursor += 24
    return AdapterFeatureBundle(x=x, target=target, delivery_dates=dates, hours=hours)


def balanced_event_weights(labels: np.ndarray, markets: Sequence[str]) -> np.ndarray:
    """令每个市场等权、且市场内负价/非负价类别等权。"""
    labels = np.asarray(labels, dtype=bool)
    markets = np.asarray(markets)
    if len(labels) != len(markets):
        raise ValueError("labels与markets长度不一致")
    weights = np.zeros(len(labels), dtype=np.float64)
    unique_markets = sorted(set(markets.tolist()))
    for market in unique_markets:
        market_mask = markets == market
        for label in (False, True):
            mask = market_mask & (labels == label)
            count = int(mask.sum())
            if count:
                weights[mask] = 1.0 / (len(unique_markets) * 2.0 * count)
    if not np.any(weights):
        raise ValueError("无法构造事件权重")
    return weights / weights.mean()


def balanced_market_weights(markets: Sequence[str]) -> np.ndarray:
    markets = np.asarray(markets)
    weights = np.zeros(len(markets), dtype=np.float64)
    unique_markets = sorted(set(markets.tolist()))
    for market in unique_markets:
        mask = markets == market
        weights[mask] = 1.0 / (len(unique_markets) * int(mask.sum()))
    return weights / weights.mean()


def fit_event_model(
    x: np.ndarray,
    target: np.ndarray,
    markets: Sequence[str],
    *,
    seed: int = 20261019,
) -> Pipeline:
    model = Pipeline((
        ("scale", StandardScaler()),
        ("event", LogisticRegression(
            l1_ratio=1.0, C=0.1, solver="saga", max_iter=3000,
            tol=1e-3, random_state=seed,
        )),
    ))
    labels = np.asarray(target) < 0.0
    if not labels.any() or labels.all():
        raise ValueError("NEG-E训练集必须同时包含负价与非负价")
    model.fit(
        np.asarray(x), labels.astype(np.int8),
        event__sample_weight=balanced_event_weights(labels, markets),
    )
    return model


def fit_amplitude_model(
    x: np.ndarray,
    target: np.ndarray,
    markets: Sequence[str],
) -> Pipeline:
    target = np.asarray(target, dtype=np.float64)
    negative = target < 0.0
    if not negative.any():
        raise ValueError("NEG-A训练集没有负价样本")
    negative_markets = np.asarray(markets)[negative]
    model = Pipeline((
        ("scale", StandardScaler()),
        ("amplitude", HuberRegressor(
            epsilon=1.35, alpha=1e-4, max_iter=5000, tol=1e-4,
        )),
    ))
    model.fit(
        np.asarray(x)[negative], target[negative],
        amplitude__sample_weight=balanced_market_weights(negative_markets),
    )
    return model


def fit_conservative_seasonal_event_model(
    x: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 20261019,
) -> Pipeline:
    """拟合第二版稀疏事件头；样本频率本身表达负价先验。"""
    model = Pipeline((
        ("scale", StandardScaler()),
        ("event", LogisticRegression(
            l1_ratio=1.0, C=0.1, solver="liblinear", max_iter=3000,
            tol=1e-5, random_state=seed,
        )),
    ))
    labels = np.asarray(target) < 0.0
    if not labels.any() or labels.all():
        raise ValueError("保守NEG-E训练集必须同时包含负价与非负价")
    model.fit(
        np.asarray(x)[:, SEASONAL_EVENT_FEATURE_INDICES],
        labels.astype(np.int8),
    )
    return model


def fit_unweighted_amplitude_model(
    x: np.ndarray,
    target: np.ndarray,
) -> Pipeline:
    """按真实源市场事件频率拟合条件幅度，避免十几个事件获得过大权重。"""
    target = np.asarray(target, dtype=np.float64)
    negative = target < 0.0
    if not negative.any():
        raise ValueError("NEG-A训练集没有负价样本")
    model = Pipeline((
        ("scale", StandardScaler()),
        ("amplitude", HuberRegressor(
            epsilon=1.35, alpha=1e-4, max_iter=5000, tol=1e-4,
        )),
    ))
    model.fit(np.asarray(x)[negative], target[negative])
    return model


def select_source_fpr_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    markets: Sequence[str],
    *,
    maximum_false_positive_rate: float = 0.01,
) -> tuple[float, dict[str, float]]:
    """选择使每个源市场非负时段误报率不高于预算的最小阈值。"""
    if not 0.0 < maximum_false_positive_rate < 1.0:
        raise ValueError("maximum_false_positive_rate必须位于(0,1)")
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    markets = np.asarray(markets)
    quantile = 1.0 - maximum_false_positive_rate
    by_market = {}
    for market in sorted(set(markets.tolist())):
        nonnegative = probability[(markets == market) & (target >= 0.0)]
        if not len(nonnegative):
            raise ValueError(f"{market}没有非负验证样本")
        value = float(np.quantile(nonnegative, quantile, method="higher"))
        # 部署规则是 >=，因此越过同分值点，确保经验误报预算不会因并列破坏。
        by_market[market] = float(np.nextafter(value, np.inf))
    return max(by_market.values()), by_market


def event_metrics(probability, target, threshold: float) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(target) < 0.0
    predicted = probability >= threshold
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def select_market_macro_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    markets: Sequence[str],
    candidates: Iterable[float] = tuple(np.arange(0.05, 0.951, 0.05)),
) -> tuple[float, list[dict]]:
    markets = np.asarray(markets)
    rows = []
    for threshold in candidates:
        by_market = {
            market: event_metrics(
                np.asarray(probability)[markets == market],
                np.asarray(target)[markets == market],
                float(threshold),
            )
            for market in sorted(set(markets.tolist()))
        }
        rows.append({
            "threshold": float(threshold),
            "macro_f1": float(np.mean([x["f1"] for x in by_market.values()])),
            "macro_false_positive_rate": float(np.mean([
                x["false_positive_rate"] for x in by_market.values()
            ])),
            "by_market": by_market,
        })
    # 最大F1；并列时更低FPR；仍并列时选择更高阈值。
    best = min(
        rows,
        key=lambda row: (
            -round(row["macro_f1"], 12),
            round(row["macro_false_positive_rate"], 12),
            -row["threshold"],
        ),
    )
    return float(best["threshold"]), rows


def apply_negative_price_adapter(
    base_prediction: np.ndarray,
    event_probability: np.ndarray,
    amplitude_prediction: np.ndarray,
    *,
    threshold: float,
    lower_bound: float = -500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_prediction, dtype=np.float64)
    probability = np.asarray(event_probability, dtype=np.float64)
    amplitude = np.asarray(amplitude_prediction, dtype=np.float64)
    if base.shape != probability.shape or base.shape != amplitude.shape:
        raise ValueError("基础预测、事件概率与幅度预测必须同形")
    triggered = probability >= threshold
    event_only = base.copy()
    event_only[triggered] = np.minimum(event_only[triggered], -1.0)
    event_and_amplitude = base.copy()
    event_and_amplitude[triggered] = np.clip(
        amplitude[triggered], lower_bound, -1.0
    )
    return event_only, event_and_amplitude, triggered

