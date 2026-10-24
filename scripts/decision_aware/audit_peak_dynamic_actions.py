#!/usr/bin/env python
"""Audit how DA peak-dynamic post-training changes LP/MPC actions and profit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.continuous_mpc import locked_continuous_mpc_backtest  # noqa: E402
from decision_aware.global_evaluation import align_node_task_predictions  # noqa: E402
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _thresholds,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


TASKS = ("da", "rt_at_da", "rt")
TOLERANCE_MWH = 1e-6


def _node_input(archives: dict, family: str, index: int) -> dict:
    return align_node_task_predictions(
        _node_record(archives[(family, "da")], index),
        _node_record(archives[(family, "rt_at_da")], index),
        _node_record(archives[(family, "rt")], index),
    )


def _worker(config_path: str, node: str, inputs: dict[str, dict]) -> tuple:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = PilotConfig.from_yaml(config_path)
    reports = {
        family: locked_continuous_mpc_backtest(
            item["da_price_forecasts"], item["rt_at_da_price_forecasts"],
            item["rt_price_forecasts"], item["realized_da_price"],
            item["realized_rt_price"], item["local_timestamps"], cfg,
            include_hourly_details=True,
        )
        for family, item in inputs.items()
    }
    return node, reports


def _state(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.int8)
    result[values > TOLERANCE_MWH] = 1
    result[values < -TOLERANCE_MWH] = -1
    return result


def _counts(values: np.ndarray) -> dict[str, int]:
    state = _state(values)
    return {
        "charge_hours": int(np.sum(state < 0)),
        "idle_hours": int(np.sum(state == 0)),
        "discharge_hours": int(np.sum(state > 0)),
    }


def _macro_components(reports: dict[str, dict]) -> dict[str, float]:
    fields = ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty")
    return {
        field: float(np.mean([
            report["revenue_components"][field] / report["days"]
            for report in reports.values()
        ]))
        for field in fields
    }


def _advantage(rows: list[dict], mask: np.ndarray, plan_state: np.ndarray) -> dict:
    if not np.any(mask):
        return {"hours": 0, "mean_predicted_da_advantage": None,
                "mean_realized_da_advantage": None,
                "positive_realized_advantage_rate": None}
    predicted = np.asarray([row["predicted_da_price"] for row in rows], dtype=float)
    realized = np.asarray([row["realized_da_price"] for row in rows], dtype=float)
    dates = np.asarray([row["delivery_date"] for row in rows])
    predicted_median = np.empty_like(predicted)
    realized_median = np.empty_like(realized)
    for day in np.unique(dates):
        day_mask = dates == day
        predicted_median[day_mask] = np.median(predicted[day_mask])
        realized_median[day_mask] = np.median(realized[day_mask])
    direction = plan_state.astype(float)
    predicted_advantage = direction * (predicted - predicted_median)
    realized_advantage = direction * (realized - realized_median)
    return {
        "hours": int(mask.sum()),
        "mean_predicted_da_advantage": float(predicted_advantage[mask].mean()),
        "mean_realized_da_advantage": float(realized_advantage[mask].mean()),
        "positive_realized_advantage_rate": float(np.mean(realized_advantage[mask] > 0)),
    }


def _remove_top_days(daily_delta: dict[str, float]) -> dict[str, float | None]:
    values = np.asarray(list(daily_delta.values()), dtype=float)
    ordered = np.sort(values)
    result: dict[str, float | None] = {"all_days": float(values.mean())}
    for count in (1, 5, 10):
        result[f"without_top_{count}_delta_days"] = (
            float(ordered[:-count].mean()) if len(ordered) > count else None
        )
    return result


def _forecast_artifacts(
    archives: dict, current_family: str, candidate_family: str, phase: str
) -> dict:
    current = archives[(current_family, "da")]
    candidate = archives[(candidate_family, "da")]
    thresholds = _thresholds("ERCOT", current["nodes"], phase)
    threshold = thresholds.da_upper_1pct
    truth = current["target_da"].astype(float)
    current_prediction = current["prediction"].astype(float)
    candidate_prediction = candidate["prediction"].astype(float)
    no_true_spike_day = np.max(truth, axis=-1) < threshold

    def false_peak(prediction: np.ndarray) -> dict:
        predicted_peak = np.max(prediction, axis=-1) >= threshold
        denominator = int(no_true_spike_day.sum())
        count = int(np.sum(predicted_peak & no_true_spike_day))
        return {
            "non_spike_node_days": denominator,
            "false_high_node_days": count,
            "false_high_rate": count / denominator if denominator else None,
        }

    ordinary = (
        np.isfinite(truth) & (truth >= 0.0)
        & (truth > thresholds.da_lower_1pct) & (truth < threshold)
    )
    node_rows = []
    for index, node in enumerate(current["nodes"]):
        mask = ordinary[index]
        original_mae = float(np.mean(np.abs(current_prediction[index][mask] - truth[index][mask])))
        candidate_mae = float(np.mean(np.abs(candidate_prediction[index][mask] - truth[index][mask])))
        node_rows.append({
            "node": str(node),
            "original_ordinary_mae": original_mae,
            "candidate_ordinary_mae": candidate_mae,
            "delta_candidate_minus_original": candidate_mae - original_mae,
        })
    node_rows.sort(key=lambda row: row["delta_candidate_minus_original"], reverse=True)
    positive_total = sum(max(0.0, row["delta_candidate_minus_original"]) for row in node_rows)
    top3_positive = sum(max(0.0, row["delta_candidate_minus_original"]) for row in node_rows[:3])
    return {
        "positive_threshold": threshold,
        "non_spike_false_high": {
            "original": false_peak(current_prediction),
            "peak_dynamic": false_peak(candidate_prediction),
        },
        "ordinary_mae_by_node": node_rows,
        "ordinary_mae_worsening_top3_share": (
            top3_positive / positive_total if positive_total > 1e-12 else 0.0
        ),
    }


def audit(
    prediction_dir: Path, phase: str, current_family: str,
    candidate_family: str, config_path: Path, workers: int,
) -> dict:
    archives = {
        (family, task): _load_archive(prediction_dir, family, task, phase)
        for family in (current_family, candidate_family) for task in TASKS
    }
    nodes = archives[(current_family, "da")]["nodes"]
    for task in TASKS:
        if not np.array_equal(nodes, archives[(candidate_family, task)]["nodes"]):
            raise ValueError(f"{task}候选节点与原系统不一致")
    reports: dict[str, dict[str, dict]] = {current_family: {}, candidate_family: {}}
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, node in enumerate(nodes):
            inputs = {
                family: _node_input(archives, family, index)
                for family in (current_family, candidate_family)
            }
            jobs.append(pool.submit(_worker, str(config_path), str(node), inputs))
        for future in as_completed(jobs):
            node, item = future.result()
            for family in item:
                reports[family][node] = item[family]

    aggregate_rows: dict[str, list[dict]] = {current_family: [], candidate_family: []}
    for family in aggregate_rows:
        for node in sorted(reports[family]):
            rows = reports[family][node]["hourly"]
            for row in rows:
                row["node"] = node
            aggregate_rows[family].extend(rows)
    current_rows = aggregate_rows[current_family]
    candidate_rows = aggregate_rows[candidate_family]
    keys = [(row["node"], row["timestamp_local"]) for row in current_rows]
    if keys != [(row["node"], row["timestamp_local"]) for row in candidate_rows]:
        raise ValueError("原系统与候选逐小时记录未对齐")
    current_plan = np.asarray([row["da_plan_mwh"] for row in current_rows])
    candidate_plan = np.asarray([row["da_plan_mwh"] for row in candidate_rows])
    current_actual = np.asarray([row["actual_dispatch_mwh"] for row in current_rows])
    candidate_actual = np.asarray([row["actual_dispatch_mwh"] for row in candidate_rows])
    original_plan_state = _state(current_plan)
    candidate_plan_state = _state(candidate_plan)
    added = (original_plan_state == 0) & (candidate_plan_state != 0)
    cancelled = (original_plan_state != 0) & (candidate_plan_state == 0)
    reversed_direction = original_plan_state * candidate_plan_state == -1
    changed_magnitude = (
        (original_plan_state == candidate_plan_state)
        & (candidate_plan_state != 0)
        & (np.abs(candidate_plan - current_plan) > TOLERANCE_MWH)
    )

    node_delta = []
    daily_by_node: dict[str, dict[str, float]] = {}
    for node in sorted(reports[current_family]):
        original = reports[current_family][node]
        candidate = reports[candidate_family][node]
        dates = sorted(original["daily"])
        if dates != sorted(candidate["daily"]):
            raise ValueError(f"{node}日期不一致")
        daily_by_node[node] = {
            day: candidate["daily"][day]["net_revenue"] - original["daily"][day]["net_revenue"]
            for day in dates
        }
        node_delta.append({
            "node": node,
            "mean_daily_profit_difference": float(np.mean(list(daily_by_node[node].values()))),
        })
    node_delta.sort(key=lambda row: row["mean_daily_profit_difference"], reverse=True)
    dates = sorted(next(iter(daily_by_node.values())))
    macro_daily_delta = {
        day: float(np.mean([daily_by_node[node][day] for node in daily_by_node]))
        for day in dates
    }
    cfg = PilotConfig.from_yaml(config_path)

    def system_summary(family: str, rows: list[dict]) -> dict:
        node_reports = reports[family]
        days = len(next(iter(node_reports.values()))["daily"])
        throughput = float(np.mean([
            report["actual_throughput_mwh"] / days for report in node_reports.values()
        ]))
        components = _macro_components(node_reports)
        net = (
            components["da_leg"] + components["rt_deviation_leg"]
            - components["degradation_cost"] - components["deviation_penalty"]
        )
        return {
            "da_plan_actions": _counts(np.asarray([row["da_plan_mwh"] for row in rows])),
            "actual_dispatch_actions": _counts(np.asarray([row["actual_dispatch_mwh"] for row in rows])),
            "macro_mean_daily_throughput_mwh": throughput,
            "macro_mean_daily_equivalent_cycles": throughput / (2.0 * cfg.bess_energy_mwh),
            "macro_mean_daily_revenue_components": components,
            "macro_mean_daily_net_revenue": net,
        }

    return {
        "phase": phase,
        "year": 2023 if phase == "inner" else 2024,
        "test_2025_read": False,
        "action_threshold_mwh": TOLERANCE_MWH,
        "systems": {
            "original": system_summary(current_family, current_rows),
            "peak_dynamic": system_summary(candidate_family, candidate_rows),
        },
        "da_plan_changes": {
            "added_active_hours": int(added.sum()),
            "cancelled_active_hours": int(cancelled.sum()),
            "reversed_direction_hours": int(reversed_direction.sum()),
            "same_direction_changed_magnitude_hours": int(changed_magnitude.sum()),
            "added_action_price_advantage": _advantage(
                candidate_rows, added, candidate_plan_state
            ),
            "reversed_action_price_advantage": _advantage(
                candidate_rows, reversed_direction, candidate_plan_state
            ),
        },
        "profit_difference_concentration": {
            "daily_macro_difference": macro_daily_delta,
            "mean_after_removing_top_delta_days": _remove_top_days(macro_daily_delta),
            "node_mean_daily_difference": node_delta,
            "positive_node_rate": float(np.mean([
                row["mean_daily_profit_difference"] > 0 for row in node_delta
            ])),
        },
        "forecast_artifacts": _forecast_artifacts(
            archives, current_family, candidate_family, phase
        ),
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# ERCOT {report['year']} DA局部峰值动作机制审计", "",
        "> 纯事后解释；不参与选模；未读取2025。", "",
        "| 系统 | 计划充电小时 | 计划放电小时 | 实际充电小时 | 实际放电小时 | 日吞吐量 | 等效循环 | 净收益 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("original", "原Transformer"), ("peak_dynamic", "峰值动态")):
        item = report["systems"][name]
        lines.append(
            f"| {label} | {item['da_plan_actions']['charge_hours']} | "
            f"{item['da_plan_actions']['discharge_hours']} | "
            f"{item['actual_dispatch_actions']['charge_hours']} | "
            f"{item['actual_dispatch_actions']['discharge_hours']} | "
            f"{item['macro_mean_daily_throughput_mwh']:.2f} | "
            f"{item['macro_mean_daily_equivalent_cycles']:.3f} | "
            f"{item['macro_mean_daily_net_revenue']:.2f} |"
        )
    changes = report["da_plan_changes"]
    added = changes["added_action_price_advantage"]
    concentration = report["profit_difference_concentration"]
    removed = concentration["mean_after_removing_top_delta_days"]
    lines.extend([
        "", "## 动作变化", "",
        f"- 新增有效DA计划小时：{changes['added_active_hours']}；取消：{changes['cancelled_active_hours']}；反向：{changes['reversed_direction_hours']}。",
        f"- 新增动作的平均预测价差优势：{added['mean_predicted_da_advantage']};平均真实价差优势：{added['mean_realized_da_advantage']}。",
        "", "## 稳健性", "",
        f"- 全部日期平均收益差：{removed['all_days']:+.2f}。",
        f"- 删除收益差最高1/5/10日后：{removed['without_top_1_delta_days']:+.2f} / {removed['without_top_5_delta_days']:+.2f} / {removed['without_top_10_delta_days']:+.2f}。",
        f"- 正收益节点占比：{concentration['positive_node_rate']:.1%}。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--phase", choices=("inner", "outer"), required=True)
    parser.add_argument("--current-family", required=True)
    parser.add_argument("--candidate-family", required=True)
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        ROOT / args.prediction_dir, args.phase, args.current_family,
        args.candidate_family, ROOT / args.config, args.workers,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "event": "peak_dynamic_action_audit_complete",
        "year": result["year"],
        "output": output.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
