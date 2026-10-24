"""负价、尖峰和极端DA-RT价差的冻结阈值诊断。

阈值只能由相应训练段真实价格计算；验证/测试标签只用于把样本放入已经
冻结的切片。模块本身不训练模型，也不参与LP/MPC决策。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TailThresholds:
    da_lower_1pct: float
    da_upper_1pct: float
    rt_lower_1pct: float
    rt_upper_1pct: float
    abs_spread_upper_5pct: float
    abs_spread_upper_1pct: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_tail_thresholds(train_da, train_rt) -> TailThresholds:
    """只用训练段真值冻结价格与价差阈值。"""
    da = _finite_flat(train_da, "train_da")
    rt = _finite_flat(train_rt, "train_rt")
    if da.shape != rt.shape:
        raise ValueError("train_da与train_rt必须同形")
    spread = np.abs(da - rt)
    return TailThresholds(
        da_lower_1pct=float(np.quantile(da, 0.01)),
        da_upper_1pct=float(np.quantile(da, 0.99)),
        rt_lower_1pct=float(np.quantile(rt, 0.01)),
        rt_upper_1pct=float(np.quantile(rt, 0.99)),
        abs_spread_upper_5pct=float(np.quantile(spread, 0.95)),
        abs_spread_upper_1pct=float(np.quantile(spread, 0.99)),
    )


def price_slice_masks(target, *, lower_1pct: float, upper_1pct: float) -> dict[str, np.ndarray]:
    truth = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(truth)
    negative = finite & (truth < 0.0)
    lower = finite & (truth <= lower_1pct)
    upper = finite & (truth >= upper_1pct)
    return {
        "all": finite,
        "negative_price": negative,
        "lowest_1pct": lower,
        "highest_1pct": upper,
        "ordinary": finite & ~negative & ~lower & ~upper,
    }


def spread_slice_masks(true_da, true_rt, thresholds: TailThresholds) -> dict[str, np.ndarray]:
    da = np.asarray(true_da, dtype=np.float64)
    rt = np.asarray(true_rt, dtype=np.float64)
    if da.shape != rt.shape:
        raise ValueError("true_da与true_rt必须同形")
    spread = da - rt
    absolute = np.abs(spread)
    finite = np.isfinite(spread)
    top_5 = finite & (absolute >= thresholds.abs_spread_upper_5pct)
    top_1 = finite & (absolute >= thresholds.abs_spread_upper_1pct)
    return {
        "all": finite,
        "abs_spread_top_5pct": top_5,
        "abs_spread_top_1pct": top_1,
        "ordinary": finite & ~top_5,
    }


def price_slice_report(
    prediction,
    target,
    *,
    lower_1pct: float,
    upper_1pct: float,
) -> dict[str, dict]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("prediction与target必须同形")
    masks = price_slice_masks(
        truth, lower_1pct=lower_1pct, upper_1pct=upper_1pct
    )
    return {
        name: _price_metrics(pred, truth, mask, upper_1pct=upper_1pct)
        for name, mask in masks.items()
    }


def spread_slice_report(
    predicted_da,
    predicted_rt,
    true_da,
    true_rt,
    thresholds: TailThresholds,
) -> dict[str, dict]:
    pred_da = np.asarray(predicted_da, dtype=np.float64)
    pred_rt = np.asarray(predicted_rt, dtype=np.float64)
    da = np.asarray(true_da, dtype=np.float64)
    rt = np.asarray(true_rt, dtype=np.float64)
    if not (pred_da.shape == pred_rt.shape == da.shape == rt.shape):
        raise ValueError("预测DA/RT和真实DA/RT必须同形")
    pred_spread = pred_da - pred_rt
    true_spread = da - rt
    masks = spread_slice_masks(da, rt, thresholds)
    return {
        name: _spread_metrics(pred_spread, true_spread, mask)
        for name, mask in masks.items()
    }


def settlement_slice_report(
    hourly: Mapping[str, np.ndarray],
    thresholds: TailThresholds,
) -> dict[str, dict]:
    """按真实价格/价差切片汇总逐小时完整结算账本。"""
    da = np.asarray(hourly["realized_da_price"], dtype=np.float64)
    rt = np.asarray(hourly["realized_rt_price"], dtype=np.float64)
    if da.shape != rt.shape:
        raise ValueError("逐小时DA与RT真值必须同形")
    masks: dict[str, np.ndarray] = {}
    for prefix, target, lower, upper in (
        ("da", da, thresholds.da_lower_1pct, thresholds.da_upper_1pct),
        ("rt", rt, thresholds.rt_lower_1pct, thresholds.rt_upper_1pct),
    ):
        for name, mask in price_slice_masks(
            target, lower_1pct=lower, upper_1pct=upper
        ).items():
            if name != "all":
                masks[f"{prefix}_{name}"] = mask
    for name, mask in spread_slice_masks(da, rt, thresholds).items():
        if name != "all":
            masks[name] = mask
    masks = {"all": np.isfinite(da) & np.isfinite(rt), **masks}
    fields = (
        "da_leg", "rt_deviation_leg", "degradation_cost",
        "deviation_penalty", "net_revenue",
    )
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        record = {"hours": count, "hour_share": float(mask.mean())}
        for field in fields:
            values = np.asarray(hourly[field], dtype=np.float64)
            if values.shape != da.shape:
                raise ValueError(f"逐小时字段{field}形状不一致")
            record[f"total_{field}"] = float(values[mask].sum())
            record[f"mean_hourly_{field}"] = (
                float(values[mask].mean()) if count else None
            )
        result[name] = record
    return result


def event_detection_report(
    prediction,
    target,
    *,
    threshold: float,
    direction: str,
    tolerance_hours: int = 1,
) -> dict:
    """把尾部事件的“是否发生”与“发生后幅度”分开报告。

    最后一维视为连续小时，因此容差指标只在同一条曲线内匹配，
    不会跨节点或交付日。`direction='above'`用于正尖峰，
    `direction='below'`用于负价或低尾。
    """
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("预测与真值必须同形")
    if pred.ndim < 1 or pred.shape[-1] < 1:
        raise ValueError("预测必须包含小时维")
    if tolerance_hours < 0:
        raise ValueError("tolerance_hours不能为负")
    finite = np.isfinite(pred) & np.isfinite(truth)
    if direction == "above":
        true_event = finite & (truth >= threshold)
        pred_event = finite & (pred >= threshold)
        true_magnitude = np.maximum(truth - threshold, 0.0)
        pred_magnitude = np.maximum(pred - threshold, 0.0)
    elif direction == "below":
        true_event = finite & (truth < threshold)
        pred_event = finite & (pred < threshold)
        true_magnitude = np.maximum(threshold - truth, 0.0)
        pred_magnitude = np.maximum(threshold - pred, 0.0)
    else:
        raise ValueError("direction必须为above或below")

    exact = _binary_metrics(true_event, pred_event)
    dilated_prediction = _dilate_last_axis(pred_event, tolerance_hours)
    dilated_truth = _dilate_last_axis(true_event, tolerance_hours)
    tolerant = _binary_metrics(
        true_event,
        pred_event,
        recall_match=dilated_prediction,
        precision_match=dilated_truth,
    )
    event_count = int(true_event.sum())
    denominator = float(np.sum(true_magnitude[true_event]))
    predicted_total = float(np.sum(pred_magnitude[true_event]))
    return {
        "true_events": event_count,
        "predicted_events": int(pred_event.sum()),
        "exact": exact,
        f"within_{tolerance_hours}h": tolerant,
        "mean_true_magnitude": (
            float(np.mean(true_magnitude[true_event])) if event_count else None
        ),
        "mean_predicted_magnitude_at_true_events": (
            float(np.mean(pred_magnitude[true_event])) if event_count else None
        ),
        "magnitude_ratio_at_true_events": (
            predicted_total / denominator if denominator > 1e-12 else None
        ),
        "wrong_side_rate_at_true_events": (
            float(np.mean(~pred_event[true_event])) if event_count else None
        ),
    }


def daily_peak_report(
    prediction,
    target,
    *,
    positive_threshold: float,
) -> dict:
    """报告含真实正尖峰的节点-日曲线上，日峰值的时点和幅度。"""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("预测与真值必须同形")
    if pred.ndim < 2 or pred.shape[-1] < 2:
        raise ValueError("日峰值评价需要最后一维为小时")
    finite_curve = np.all(np.isfinite(pred) & np.isfinite(truth), axis=-1)
    true_peak_value = np.max(truth, axis=-1)
    selected = finite_curve & (true_peak_value >= positive_threshold)
    count = int(selected.sum())
    if not count:
        return {
            "node_days": 0,
            "exact_hour_rate": None,
            "within_1h_rate": None,
            "mean_absolute_timing_error_hours": None,
            "predicted_daily_peak_to_true_peak_ratio": None,
            "prediction_at_true_peak_to_true_peak_ratio": None,
        }
    true_hour = np.argmax(truth, axis=-1)[selected]
    pred_hour = np.argmax(pred, axis=-1)[selected]
    timing_error = np.abs(pred_hour - true_hour)
    selected_truth = truth[selected]
    selected_pred = pred[selected]
    row = np.arange(count)
    actual_peak = selected_truth[row, true_hour]
    predicted_daily_peak = selected_pred[row, pred_hour]
    prediction_at_true_peak = selected_pred[row, true_hour]
    denominator = float(np.mean(actual_peak))
    return {
        "node_days": count,
        "exact_hour_rate": float(np.mean(timing_error == 0)),
        "within_1h_rate": float(np.mean(timing_error <= 1)),
        "mean_absolute_timing_error_hours": float(np.mean(timing_error)),
        "predicted_daily_peak_to_true_peak_ratio": (
            float(np.mean(predicted_daily_peak) / denominator)
            if abs(denominator) > 1e-12 else None
        ),
        "prediction_at_true_peak_to_true_peak_ratio": (
            float(np.mean(prediction_at_true_peak) / denominator)
            if abs(denominator) > 1e-12 else None
        ),
    }


def _finite_flat(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    result = result[np.isfinite(result)]
    if result.size == 0:
        raise ValueError(f"{name}没有有限值")
    return result


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _dilate_last_axis(mask: np.ndarray, tolerance: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if tolerance == 0:
        return result
    original = np.asarray(mask, dtype=bool)
    width = original.shape[-1]
    for offset in range(1, tolerance + 1):
        if offset >= width:
            break
        result[..., offset:] |= original[..., :-offset]
        result[..., :-offset] |= original[..., offset:]
    return result


def _binary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    recall_match: np.ndarray | None = None,
    precision_match: np.ndarray | None = None,
) -> dict[str, float | None]:
    recall_hits = truth & (prediction if recall_match is None else recall_match)
    precision_hits = prediction & (truth if precision_match is None else precision_match)
    recall = _safe_rate(int(recall_hits.sum()), int(truth.sum()))
    precision = _safe_rate(int(precision_hits.sum()), int(prediction.sum()))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _price_metrics(pred, truth, mask, *, upper_1pct: float) -> dict:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    count = int(valid.sum())
    if not count:
        return {
            "samples": 0, "mae": None, "rmse": None, "bias": None,
            "negative_recall": None, "negative_precision": None,
            "spike_recall": None,
        }
    error = pred[valid] - truth[valid]
    true_negative = valid & (truth < 0.0)
    pred_negative = valid & (pred < 0.0)
    true_spike = valid & (truth >= upper_1pct)
    pred_spike = valid & (pred >= upper_1pct)
    return {
        "samples": count,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "negative_recall": _safe_rate(
            int(np.sum(true_negative & pred_negative)), int(true_negative.sum())
        ),
        "negative_precision": _safe_rate(
            int(np.sum(true_negative & pred_negative)), int(pred_negative.sum())
        ),
        "spike_recall": _safe_rate(
            int(np.sum(true_spike & pred_spike)), int(true_spike.sum())
        ),
    }


def _spread_metrics(predicted, truth, mask) -> dict:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(predicted) & np.isfinite(truth)
    count = int(valid.sum())
    if not count:
        return {
            "samples": 0, "spread_mae": None, "spread_bias": None,
            "direction_accuracy": None, "magnitude_ratio": None,
        }
    pred = predicted[valid]
    target = truth[valid]
    denominator = float(np.mean(np.abs(target)))
    return {
        "samples": count,
        "spread_mae": float(np.mean(np.abs(pred - target))),
        "spread_bias": float(np.mean(pred - target)),
        "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(target))),
        "magnitude_ratio": (
            float(np.mean(np.abs(pred)) / denominator)
            if denominator > 1e-12 else None
        ),
    }
