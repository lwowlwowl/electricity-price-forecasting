"""正尖峰日级跨市场可预测性审计的纯函数。

只使用DA起报时可见的168小时历史、严格lag-168和目标日历。模块不加载
任何市场文件，不包含ERCOT评价或2025入口。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision_aware.negative_price_adapter import FEATURE_NAMES


DAY_FEATURE_NAMES = (
    "da_lag168_peak_z", "da_lag168_median_z", "da_lag168_range_robust",
    "da_context_max_z", "da_context_last_z", "da_context_negative_share",
    "rt_lag168_peak_z", "rt_lag168_median_z", "rt_lag168_range_robust",
    "rt_context_max_z", "rt_context_last_z", "rt_context_negative_share",
    "load_lag168_peak_z", "load_lag168_mean_z",
    "net_load_lag168_peak_z", "net_load_lag168_mean_z",
    "wind_lag168_mean_z", "solar_lag168_mean_z",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday",
)


@dataclass(frozen=True)
class DailyEventBundle:
    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    node_ids: np.ndarray
    nodes: np.ndarray
    event_thresholds: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.y)
        if self.x.shape != (rows, len(DAY_FEATURE_NAMES)):
            raise ValueError(f"日级特征形状错误: {self.x.shape}")
        if any(len(value) != rows for value in (self.dates, self.node_ids)):
            raise ValueError("日级特征与标签/元数据长度不一致")
        if not np.isfinite(self.x).all():
            raise ValueError("日级特征包含非有限值")


def robust_scale(median: float, mad: float) -> float:
    return max(float(mad), 0.05 * abs(float(median)), 1e-3)


def robust_z(value: float, median: float, mad: float) -> float:
    return (float(value) - float(median)) / robust_scale(median, mad)


def node_event_thresholds(
    bundle: Mapping[str, np.ndarray],
    *,
    end_year: int = 2021,
    quantile: float = 0.99,
) -> np.ndarray:
    dates = np.asarray(bundle["dates"]).astype(str)
    years = np.asarray([int(value[:4]) for value in dates], dtype=np.int16)
    node_ids = np.asarray(bundle["node_id"], dtype=np.int64)
    target = np.asarray(bundle["target"], dtype=np.float64)
    nodes = np.asarray(bundle["nodes"])
    thresholds = np.empty(len(nodes), dtype=np.float64)
    for node_id in range(len(nodes)):
        values = target[(node_ids == node_id) & (years <= end_year)]
        if not len(values):
            raise ValueError(f"节点{nodes[node_id]}没有事件阈值样本")
        thresholds[node_id] = float(np.quantile(values, quantile, method="linear"))
    return thresholds


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _daily_row(block: np.ndarray) -> np.ndarray:
    first = block[0]

    def summary(prefix: str, lag_name: str) -> tuple[float, float, float]:
        values = block[:, _index(lag_name)].astype(np.float64)
        median = float(first[_index(f"{prefix}_ctx_median")])
        mad = float(first[_index(f"{prefix}_ctx_mad")])
        scale = robust_scale(median, mad)
        return (
            robust_z(float(np.max(values)), median, mad),
            robust_z(float(np.median(values)), median, mad),
            float((np.max(values) - np.min(values)) / scale),
        )

    da = summary("da", "da_lag168")
    rt = summary("rt", "rt_lag168")

    def fundamental(lag_name: str, prefix: str) -> tuple[float, float]:
        values = block[:, _index(lag_name)].astype(np.float64)
        median = float(first[_index(f"{prefix}_ctx_median")])
        mad = float(first[_index(f"{prefix}_ctx_mad")])
        return (
            robust_z(float(np.max(values)), median, mad),
            robust_z(float(np.mean(values)), median, mad),
        )

    load = fundamental("load_lag168", "load")
    net_load = fundamental("net_load_lag168", "net_load")

    def mean_z(lag_name: str, prefix: str) -> float:
        values = block[:, _index(lag_name)].astype(np.float64)
        median = float(first[_index(f"{prefix}_ctx_median")])
        mad = float(first[_index(f"{prefix}_ctx_mad")])
        return robust_z(float(np.mean(values)), median, mad)

    row = (
        *da,
        robust_z(first[_index("da_ctx_max")], first[_index("da_ctx_median")], first[_index("da_ctx_mad")]),
        robust_z(first[_index("da_ctx_last")], first[_index("da_ctx_median")], first[_index("da_ctx_mad")]),
        float(first[_index("da_ctx_negative_share")]),
        *rt,
        robust_z(first[_index("rt_ctx_max")], first[_index("rt_ctx_median")], first[_index("rt_ctx_mad")]),
        robust_z(first[_index("rt_ctx_last")], first[_index("rt_ctx_median")], first[_index("rt_ctx_mad")]),
        float(first[_index("rt_ctx_negative_share")]),
        *load, *net_load,
        mean_z("wind_lag168", "wind"), mean_z("solar_lag168", "solar"),
        float(first[_index("dow_sin")]), float(first[_index("dow_cos")]),
        float(first[_index("month_sin")]), float(first[_index("month_cos")]),
        float(first[_index("is_weekend")]), float(first[_index("is_holiday")]),
    )
    return np.asarray(row, dtype=np.float64)


def make_daily_bundle(
    bundle: Mapping[str, np.ndarray],
    thresholds: np.ndarray,
    *,
    years: Sequence[int],
) -> DailyEventBundle:
    x = np.asarray(bundle["x"], dtype=np.float64)
    target = np.asarray(bundle["target"], dtype=np.float64)
    dates = np.asarray(bundle["dates"]).astype(str)
    hours = np.asarray(bundle["hours"], dtype=np.int64)
    node_ids = np.asarray(bundle["node_id"], dtype=np.int64)
    nodes = np.asarray(bundle["nodes"]).astype(str)
    allowed_years = set(int(value) for value in years)
    rows, labels, out_dates, out_nodes = [], [], [], []
    for node_id in range(len(nodes)):
        node_positions = np.flatnonzero(node_ids == node_id)
        order = np.lexsort((hours[node_positions], dates[node_positions]))
        node_positions = node_positions[order]
        ordered_dates = dates[node_positions]
        node_dates, starts = np.unique(ordered_dates, return_index=True)
        ends = np.r_[starts[1:], len(node_positions)]
        for date, start, end in zip(node_dates, starts, ends, strict=True):
            if int(date[:4]) not in allowed_years:
                continue
            selected = node_positions[start:end]
            block = x[selected]
            truth = target[selected]
            day_hours = hours[selected]
            if len(block) != 24 or not np.array_equal(day_hours, np.arange(24)):
                raise ValueError(f"{nodes[node_id]} {date}不是完整24小时")
            rows.append(_daily_row(block))
            labels.append(bool(np.max(truth) > thresholds[node_id]))
            out_dates.append(date)
            out_nodes.append(node_id)
    return DailyEventBundle(
        x=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=bool),
        dates=np.asarray(out_dates),
        node_ids=np.asarray(out_nodes, dtype=np.int16),
        nodes=nodes,
        event_thresholds=np.asarray(thresholds, dtype=np.float64),
    )


def market_equal_weights(markets: Sequence[str]) -> np.ndarray:
    markets = np.asarray(markets).astype(str)
    weights = np.zeros(len(markets), dtype=np.float64)
    unique = sorted(set(markets.tolist()))
    for market in unique:
        mask = markets == market
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return weights / np.mean(weights)


def fit_sparse_event_model(
    x: np.ndarray,
    y: np.ndarray,
    markets: Sequence[str],
    *,
    seed: int = 20261020,
) -> Pipeline:
    labels = np.asarray(y, dtype=np.int8)
    if not labels.any() or labels.all():
        raise ValueError("尖峰事件训练集必须同时包含事件与非事件")
    model = Pipeline((
        ("scale", StandardScaler()),
        ("event", LogisticRegression(
            penalty="l1", C=0.1, solver="liblinear", max_iter=3000,
            random_state=seed,
        )),
    ))
    model.fit(
        np.asarray(x, dtype=np.float64), labels,
        event__sample_weight=market_equal_weights(markets),
    )
    return model


def binary_metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    predicted = probability >= float(threshold)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    event_rate = float(np.mean(truth))
    pr_auc = float(average_precision_score(truth, probability)) if truth.any() else 0.0
    brier = float(brier_score_loss(truth, probability))
    constant_brier = event_rate * (1.0 - event_rate)
    return {
        "samples": int(len(truth)), "events": int(truth.sum()),
        "event_rate": event_rate, "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / event_rate if event_rate else None,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "brier": brier,
        "constant_event_rate_brier": constant_brier,
        "brier_improvement_vs_constant": constant_brier - brier,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_macro_f1_threshold(
    probability: np.ndarray,
    truth: np.ndarray,
    markets: Sequence[str],
    *,
    candidates: Sequence[float] = tuple(np.arange(0.01, 0.501, 0.01)),
) -> tuple[float, list[dict]]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    markets = np.asarray(markets).astype(str)
    rows = []
    for candidate in candidates:
        by_market = {
            market: binary_metrics(
                probability[markets == market], truth[markets == market], float(candidate)
            )
            for market in sorted(set(markets.tolist()))
        }
        rows.append({
            "threshold": float(candidate),
            "macro_f1": float(np.mean([value["f1"] for value in by_market.values()])),
            "by_market": by_market,
        })
    best_value = max(row["macro_f1"] for row in rows)
    selected = max(
        row["threshold"] for row in rows
        if np.isclose(row["macro_f1"], best_value, atol=1e-12)
    )
    return float(selected), rows


def quantile_calibration(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> list[dict]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    order = np.argsort(probability, kind="stable")
    rows = []
    for index, selected in enumerate(np.array_split(order, bins), start=1):
        if not len(selected):
            continue
        rows.append({
            "bin": index, "samples": int(len(selected)),
            "mean_probability": float(np.mean(probability[selected])),
            "observed_rate": float(np.mean(truth[selected])),
            "minimum_probability": float(np.min(probability[selected])),
            "maximum_probability": float(np.max(probability[selected])),
        })
    return rows


def bootstrap_pr_auc_difference(
    probability: np.ndarray,
    truth: np.ndarray,
    dates: Sequence[str],
    *,
    repeats: int = 2000,
    seed: int = 20261020,
) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=bool)
    dates = np.asarray(dates).astype(str)
    unique_dates = np.unique(dates)
    by_date = {date: np.flatnonzero(dates == date) for date in unique_dates}
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        indices = np.concatenate([by_date[date] for date in sampled])
        labels = truth[indices]
        event_rate = float(np.mean(labels))
        pr_auc = float(average_precision_score(labels, probability[indices])) if labels.any() else 0.0
        differences[repeat] = pr_auc - event_rate
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "repeats": int(repeats), "resampling_unit": "delivery_date_all_nodes",
        "mean": float(np.mean(differences)),
        "ci95": [float(low), float(high)],
    }


def coefficient_rows(model: Pipeline) -> list[dict]:
    values = np.asarray(model.named_steps["event"].coef_).reshape(-1)
    return [
        {"feature": name, "standardized_coefficient": float(value), "nonzero": bool(value != 0.0)}
        for name, value in zip(DAY_FEATURE_NAMES, values, strict=True)
    ]
