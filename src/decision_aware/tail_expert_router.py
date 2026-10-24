"""Interpretable day-level routing features for tail-aware forecast experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


def _series_features(value: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    ramp = np.diff(value, axis=-1)
    arrays = [
        value.min(axis=-1),
        value.max(axis=-1),
        np.ptp(value, axis=-1),
        value.mean(axis=-1),
        value.std(axis=-1),
        np.quantile(value, 0.10, axis=-1),
        np.quantile(value, 0.90, axis=-1),
        ramp.max(axis=-1),
        ramp.min(axis=-1),
        (value < 0.0).sum(axis=-1).astype(np.float64),
    ]
    names = [
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_range",
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_q10",
        f"{prefix}_q90",
        f"{prefix}_max_up_ramp",
        f"{prefix}_max_down_ramp",
        f"{prefix}_negative_hours",
    ]
    return arrays, names


def build_feature_cube(
    universal_da: np.ndarray,
    universal_rt_at_da: np.ndarray,
    dlinear_da: np.ndarray,
    dlinear_rt_at_da: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return [node, day, feature] using only information visible at DA issue."""
    arrays = [universal_da, universal_rt_at_da, dlinear_da, dlinear_rt_at_da]
    if any(item.ndim != 3 or item.shape[-1] != 24 for item in arrays):
        raise ValueError("DA与RT-at-DA预测必须为[node,day,24]")
    if any(item.shape != arrays[0].shape for item in arrays[1:]):
        raise ValueError("四组日前预测形状必须一致")
    if len(dates) != arrays[0].shape[1]:
        raise ValueError("日期数与预测天数不一致")

    values: list[np.ndarray] = []
    names: list[str] = []
    for label, item in (
        ("transformer_da", universal_da),
        ("transformer_rt_at_da", universal_rt_at_da),
        ("dlinear_da", dlinear_da),
        ("dlinear_rt_at_da", dlinear_rt_at_da),
    ):
        item_values, item_names = _series_features(item, label)
        values.extend(item_values)
        names.extend(item_names)

    for label, da, rt in (
        ("transformer_spread", universal_da, universal_rt_at_da),
        ("dlinear_spread", dlinear_da, dlinear_rt_at_da),
    ):
        spread = da - rt
        spread_values, spread_names = _series_features(spread, label)
        values.extend(spread_values)
        names.extend(spread_names)
        values.extend([
            (spread > 6.0).sum(axis=-1).astype(np.float64),
            (spread < -6.0).sum(axis=-1).astype(np.float64),
            (np.abs(spread) > 6.0).sum(axis=-1).astype(np.float64),
        ])
        names.extend([
            f"{label}_positive_6h_count",
            f"{label}_negative_6h_count",
            f"{label}_absolute_6h_count",
        ])

    transformer_spread = universal_da - universal_rt_at_da
    dlinear_spread = dlinear_da - dlinear_rt_at_da
    values.extend([
        np.mean(np.abs(universal_da - dlinear_da), axis=-1),
        np.max(np.abs(universal_da - dlinear_da), axis=-1),
        np.mean(np.abs(universal_rt_at_da - dlinear_rt_at_da), axis=-1),
        np.max(np.abs(universal_rt_at_da - dlinear_rt_at_da), axis=-1),
        np.mean(np.abs(transformer_spread - dlinear_spread), axis=-1),
        np.max(np.abs(transformer_spread - dlinear_spread), axis=-1),
        np.abs(np.argmax(universal_da, axis=-1) - np.argmax(dlinear_da, axis=-1)),
        np.abs(np.argmin(universal_da, axis=-1) - np.argmin(dlinear_da, axis=-1)),
    ])
    names.extend([
        "da_disagreement_mae",
        "da_disagreement_max",
        "rt_at_da_disagreement_mae",
        "rt_at_da_disagreement_max",
        "spread_disagreement_mae",
        "spread_disagreement_max",
        "da_peak_hour_disagreement",
        "da_trough_hour_disagreement",
    ])

    parsed = pd.DatetimeIndex(pd.to_datetime(dates))
    month_angle = 2.0 * np.pi * (parsed.month.to_numpy() - 1.0) / 12.0
    dow_angle = 2.0 * np.pi * parsed.dayofweek.to_numpy() / 7.0
    calendar = [
        np.sin(month_angle),
        np.cos(month_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
        (parsed.dayofweek.to_numpy() >= 5).astype(np.float64),
    ]
    node_count = arrays[0].shape[0]
    values.extend([np.broadcast_to(item[None, :], (node_count, len(dates))) for item in calendar])
    names.extend(["month_sin", "month_cos", "dow_sin", "dow_cos", "weekend"])
    cube = np.stack(values, axis=-1).astype(np.float64)
    if not np.isfinite(cube).all():
        raise ValueError("路由特征包含非有限值")
    return cube, tuple(names)


def economic_sample_weights(advantage: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(advantage, dtype=np.float64))
    positive = magnitude[magnitude > 0.0]
    cap = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    weights = np.minimum(magnitude, max(cap, 1e-6))
    weights = np.maximum(weights, np.median(positive) * 0.05 if positive.size else 1.0)
    return weights / max(float(weights.mean()), 1e-12)


def router_estimator(kind: str, *, random_state: int = 20261020):
    if kind == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l1",
                C=0.1,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )),
        ])
    if kind == "tree":
        return DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=100,
            random_state=random_state,
        )
    raise ValueError(f"未知路由器: {kind}")


@dataclass
class ConstantRouter:
    probability: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.full(len(features), self.probability, dtype=np.float64)
        return np.column_stack([1.0 - probability, probability])


def fit_router(kind: str, features: np.ndarray, advantage: np.ndarray):
    target = np.asarray(advantage, dtype=np.float64) > 0.0
    if np.unique(target).size < 2:
        return ConstantRouter(float(target[0]))
    estimator = clone(router_estimator(kind))
    estimator.fit(features, target.astype(np.int64), **{
        "model__sample_weight" if kind == "logistic" else "sample_weight": economic_sample_weights(advantage)
    })
    return estimator


def fit_regret_tree(
    features: np.ndarray,
    advantage: np.ndarray,
    *,
    min_samples_leaf: int = 100,
    random_state: int = 20261021,
    sample_weight: np.ndarray | None = None,
) -> DecisionTreeRegressor:
    """Fit a fixed shallow tree to conditional economic advantage.

    The target is alternate-minus-default realized value.  A positive
    prediction therefore has a direct decision meaning: use the alternate
    expert.  No probability threshold or target-market calibration is needed.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(advantage, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("经济遗憾树输入形状不一致")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("经济遗憾树输入包含非有限值")
    model = DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != y.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("经济遗憾树样本权重无效")
    else:
        weight = None
    model.fit(x, y, sample_weight=weight)
    return model


def predict_regret(model: DecisionTreeRegressor, features: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=np.float64)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("经济遗憾树输出无效")
    return values


def regret_tree_explanation(
    model: DecisionTreeRegressor,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    used = sorted({int(value) for value in model.tree_.feature if int(value) >= 0})
    leaves = []
    for node_index in range(model.tree_.node_count):
        if model.tree_.children_left[node_index] != model.tree_.children_right[node_index]:
            continue
        leaves.append({
            "node": node_index,
            "samples": int(model.tree_.n_node_samples[node_index]),
            "predicted_advantage": float(model.tree_.value[node_index, 0, 0]),
        })
    return {
        "kind": "economic_regret_tree",
        "feature_count": len(used),
        "features": [feature_names[index] for index in used],
        "rules": export_text(model, feature_names=list(feature_names)),
        "leaves": leaves,
    }


def router_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.shape != (len(features), 2):
        raise ValueError("路由器predict_proba形状错误")
    return probabilities[:, 1]


def model_explanation(model: Any, feature_names: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(model, ConstantRouter):
        return {"kind": "constant", "probability": model.probability, "feature_count": 0}
    if isinstance(model, Pipeline):
        coefficients = model.named_steps["model"].coef_[0]
        selected = [
            {"feature": name, "standardized_coefficient": float(value)}
            for name, value in zip(feature_names, coefficients)
            if abs(float(value)) > 1e-12
        ]
        selected.sort(key=lambda item: abs(item["standardized_coefficient"]), reverse=True)
        return {"kind": "logistic", "feature_count": len(selected), "coefficients": selected}
    if isinstance(model, DecisionTreeClassifier):
        used = sorted({
            int(value) for value in model.tree_.feature if int(value) >= 0
        })
        return {
            "kind": "tree",
            "feature_count": len(used),
            "features": [feature_names[index] for index in used],
            "rules": export_text(model, feature_names=list(feature_names)),
        }
    raise TypeError(f"不支持的路由器类型: {type(model)!r}")


def blocked_oof_probability(
    kind: str,
    features: np.ndarray,
    advantage: np.ndarray,
    row_dates: np.ndarray,
    *,
    blocks: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unique_dates = np.asarray(sorted(set(str(value) for value in row_dates)))
    if len(unique_dates) < blocks:
        raise ValueError("日期数不足以进行阻塞折外评价")
    probabilities = np.full(len(features), np.nan, dtype=np.float64)
    explanations: list[dict[str, Any]] = []
    for fold, held_dates in enumerate(np.array_split(unique_dates, blocks)):
        held = np.isin(row_dates.astype(str), held_dates.astype(str))
        model = fit_router(kind, features[~held], advantage[~held])
        probabilities[held] = router_probability(model, features[held])
        explanations.append({"fold": fold, "held_dates": held_dates.astype(str).tolist(), "model": model})
    if not np.isfinite(probabilities).all():
        raise AssertionError("折外路由概率未完整填充")
    return probabilities, explanations
