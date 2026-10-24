"""多节点三任务预测的日期对齐与完整双结算输入构造。"""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


def enumerate_fusion_triplets(
    families: Sequence[str] = ("f0", "f1"),
) -> tuple[dict[str, str], ...]:
    """返回DA、RT-at-DA、RT的全部融合家族组合。"""
    cleaned = tuple(str(value).strip().lower() for value in families)
    if not cleaned or any(not value for value in cleaned):
        raise ValueError("families必须包含至少一个非空名称")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("families不能重复")
    return tuple(
        {"da": da, "rt_at_da": rt_at_da, "rt": rt}
        for da, rt_at_da, rt in product(cleaned, repeat=3)
    )


def _date_index(record: Mapping[str, Any], label: str) -> dict[str, int]:
    dates = [str(value) for value in record["dates"]]
    if len(dates) != len(set(dates)):
        raise ValueError(f"{label}包含重复交付日")
    return {value: index for index, value in enumerate(dates)}


def align_node_task_predictions(
    da: Mapping[str, Any],
    rt_at_da: Mapping[str, Any],
    rt: Mapping[str, Any],
) -> dict[str, Any]:
    """按交付日交集构造 ``locked_dual_backtest`` 的单节点输入。

    三个任务允许因窗口尾部或DST拥有不同天数，但同一天的24小时真值必须
    一致。滚动RT只用每个起报窗口的第一步作为该小时真实结算价格。
    """
    da_index = _date_index(da, "DA")
    rt_da_index = _date_index(rt_at_da, "RT-at-DA")
    rt_index = _date_index(rt, "RT")
    dates = sorted(set(da_index) & set(rt_da_index) & set(rt_index))
    if not dates:
        raise ValueError("三个任务没有共同交付日")

    da_prediction = np.asarray(da["prediction"], dtype=np.float32)
    da_true = np.asarray(da["target_da"], dtype=np.float32)
    da_true_rt = np.asarray(da["target_rt"], dtype=np.float32)
    rt_da_prediction = np.asarray(rt_at_da["prediction"], dtype=np.float32)
    rt_da_true = np.asarray(rt_at_da["target_rt"], dtype=np.float32)
    rt_prediction = np.asarray(rt["prediction"], dtype=np.float32)
    rt_true_windows = np.asarray(rt["target_rt"], dtype=np.float32)
    timestamps = np.asarray(rt["executed_timestamps"])

    expected_da = (len(da_index), 24)
    expected_rt_da = (len(rt_da_index), 24)
    expected_rt_prefix = (len(rt_index), 24)
    if da_prediction.shape != expected_da or da_true.shape != expected_da:
        raise ValueError("DA预测或真值形状必须为[day,24]")
    if da_true_rt.shape != expected_da:
        raise ValueError("DA数据携带的RT真值形状必须为[day,24]")
    if rt_da_prediction.shape != expected_rt_da or rt_da_true.shape != expected_rt_da:
        raise ValueError("RT-at-DA预测或真值形状必须为[day,24]")
    if rt_prediction.ndim != 3 or rt_prediction.shape[:2] != expected_rt_prefix:
        raise ValueError("滚动RT预测形状必须为[day,24,horizon]")
    if rt_true_windows.shape != rt_prediction.shape:
        raise ValueError("滚动RT预测和真值形状必须一致")
    if timestamps.shape != expected_rt_prefix:
        raise ValueError("executed_timestamps形状必须为[day,24]")

    da_rows = np.asarray([da_index[value] for value in dates], dtype=np.int64)
    rt_da_rows = np.asarray([rt_da_index[value] for value in dates], dtype=np.int64)
    rt_rows = np.asarray([rt_index[value] for value in dates], dtype=np.int64)
    selected_true_da = da_true[da_rows]
    selected_da_rt = da_true_rt[da_rows]
    selected_rt_da_true = rt_da_true[rt_da_rows]
    selected_rt_actual = rt_true_windows[rt_rows, :, 0]
    # The task exporters reconstruct the same source prices through independent
    # float32 paths.  Permit one-ULP drift at ERCOT spike magnitudes while still
    # rejecting economically meaningful truth mismatches.
    truth_atol = 5e-4
    if not np.allclose(selected_da_rt, selected_rt_da_true, rtol=0, atol=truth_atol):
        raise ValueError("DA与RT-at-DA数据中的真实RT价格不一致")
    if not np.allclose(selected_da_rt, selected_rt_actual, rtol=0, atol=truth_atol):
        raise ValueError("日前任务与滚动RT第一步真实价格不一致")

    timezone = str(rt.get("timezone", ""))
    if not timezone:
        raise ValueError("滚动RT记录必须显式提供市场IANA时区")
    flat_timestamps = pd.DatetimeIndex(
        pd.to_datetime(timestamps[rt_rows].reshape(-1), utc=True)
    ).tz_convert(timezone)
    if len(flat_timestamps) != 24 * len(dates):
        raise AssertionError("时间戳展平长度异常")
    return {
        "dates": dates,
        "da_price_forecasts": {
            pd.Timestamp(value).date(): torch.from_numpy(da_prediction[da_index[value]])
            for value in dates
        },
        "rt_at_da_price_forecasts": {
            pd.Timestamp(value).date(): torch.from_numpy(
                rt_da_prediction[rt_da_index[value]]
            )
            for value in dates
        },
        "rt_price_forecasts": torch.from_numpy(rt_prediction[rt_rows].reshape(-1, rt_prediction.shape[-1])),
        "realized_da_price": torch.from_numpy(selected_true_da.reshape(-1)),
        "realized_rt_price": torch.from_numpy(selected_rt_actual.reshape(-1)),
        "local_timestamps": flat_timestamps,
    }


def compact_backtest_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """去掉逐日明细，保留跨组合排名所需的审计指标。"""
    node_reports = report["node_reports"]
    compact_nodes = {}
    for node, item in node_reports.items():
        compact_nodes[node] = {
            "days": item["days"],
            "mean_daily_revenue": item["mean_daily_revenue"],
            "positive_day_rate": item["positive_day_rate"],
            "total_revenue": item["total_revenue"],
            "revenue_components": item["revenue_components"],
            "segments": item["segments"],
            "state_resets_due_to_gaps": item["state_resets_due_to_gaps"],
            "terminal_inventory_accounting": item["terminal_inventory_accounting"],
            "actual_final_soc_mwh": item["actual_state"]["final_soc_mwh"],
            "plan_final_soc_mwh": item["plan_feasibility"]["final_soc_mwh"],
        }
    return {
        "node_count": report["node_count"],
        "soc_scope": report["soc_scope"],
        "macro_mean_daily_revenue": report["macro_mean_daily_revenue"],
        "nodes": compact_nodes,
    }
