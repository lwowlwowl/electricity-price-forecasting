#!/usr/bin/env python
"""Post-hoc 2025 DA profit decomposition between Transformer and DLinear.

This script is diagnostic only: 2025 was already opened, selection is forbidden.
It trains nothing and uses truth only for reporting/settlement after constructing
the fixed prediction-only rank/value counterfactuals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
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
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _block_bootstrap_ci,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import _node_record  # noqa: E402


SYSTEM_LABELS = {
    "transformer": "T/T/T",
    "dlinear": "D/D/D",
    "dlinear_da_transformer_rt": "D/T/T",
    "transformer_da_dlinear_rt": "T/D/D",
    "transformer_rank_dlinear_values": "T-rank/D-values",
    "dlinear_rank_transformer_values": "D-rank/T-values",
}
TOLERANCE = 3.0e-4


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files if key != "metadata"}
        result["metadata"] = json.loads(str(archive["metadata"].item()))
    return result


def rank_value_recombine(rank_source: np.ndarray, value_source: np.ndarray) -> np.ndarray:
    """Assign value_source's sorted daily values to rank_source's hour ordering."""
    rank_source = np.asarray(rank_source)
    value_source = np.asarray(value_source)
    if rank_source.shape != value_source.shape or rank_source.shape[-1] != 24:
        raise ValueError("DA排序/数值反事实要求相同的(...,24)形状")
    order = np.argsort(rank_source, axis=-1, kind="stable")
    sorted_values = np.sort(value_source, axis=-1, kind="stable")
    result = np.empty_like(sorted_values)
    np.put_along_axis(result, order, sorted_values, axis=-1)
    return result


def _truncate_days(archive: dict, days: int) -> dict:
    result = dict(archive)
    result["dates"] = archive["dates"][:days]
    for key in ("prediction", "target_da", "target_rt"):
        if key in archive:
            result[key] = archive[key][:, :days]
    if "executed_timestamps" in archive:
        result["executed_timestamps"] = archive["executed_timestamps"][:days]
    return result


def _validate_and_align(directory: Path) -> dict[str, dict[str, dict]]:
    source = {
        family: {
            task: _load(directory / f"{family}_{task}_final_seed0.npz")
            for task in ("da", "rt_at_da", "rt")
        }
        for family in ("ensemble", "dlinear")
    }
    day_count = min(
        len(item["dates"])
        for family in source.values() for item in family.values()
    )
    for family in source:
        for task in source[family]:
            source[family][task] = _truncate_days(source[family][task], day_count)
    reference = source["ensemble"]["da"]
    for family, tasks in source.items():
        for task, item in tasks.items():
            if not np.array_equal(reference["nodes"], item["nodes"]):
                raise ValueError(f"节点不一致: {family}/{task}")
            if not np.array_equal(reference["dates"], item["dates"]):
                raise ValueError(f"日期不一致: {family}/{task}")
    if not np.allclose(
        reference["target_da"], source["dlinear"]["da"]["target_da"],
        rtol=0.0, atol=TOLERANCE,
    ):
        raise ValueError("Transformer与DLinear的DA真值不一致")
    return source


def _archive_like(template: dict, prediction: np.ndarray) -> dict:
    result = dict(template)
    result["prediction"] = np.asarray(prediction, dtype=np.float32)
    return result


def _systems(source: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    t = source["ensemble"]
    d = source["dlinear"]
    t_da = t["da"]["prediction"].astype(np.float64)
    d_da = d["da"]["prediction"].astype(np.float64)
    t_rank_d_values = rank_value_recombine(t_da, d_da)
    d_rank_t_values = rank_value_recombine(d_da, t_da)
    return {
        "transformer": t,
        "dlinear": d,
        "dlinear_da_transformer_rt": {
            "da": d["da"], "rt_at_da": t["rt_at_da"], "rt": t["rt"],
        },
        "transformer_da_dlinear_rt": {
            "da": t["da"], "rt_at_da": d["rt_at_da"], "rt": d["rt"],
        },
        "transformer_rank_dlinear_values": {
            "da": _archive_like(t["da"], t_rank_d_values),
            "rt_at_da": t["rt_at_da"], "rt": t["rt"],
        },
        "dlinear_rank_transformer_values": {
            "da": _archive_like(t["da"], d_rank_t_values),
            "rt_at_da": t["rt_at_da"], "rt": t["rt"],
        },
    }


def _worker(config_path: str, system: str, node: str, node_input: dict) -> tuple:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = PilotConfig.from_yaml(config_path)
    report = locked_continuous_mpc_backtest(
        node_input["da_price_forecasts"],
        node_input["rt_at_da_price_forecasts"],
        node_input["rt_price_forecasts"],
        node_input["realized_da_price"],
        node_input["realized_rt_price"],
        node_input["local_timestamps"],
        cfg,
        include_hourly_details=True,
    )
    return system, node, report


def _state_count(values: np.ndarray) -> dict:
    return {
        "charge": int(np.sum(values < -1.0e-6)),
        "idle": int(np.sum(np.abs(values) <= 1.0e-6)),
        "discharge": int(np.sum(values > 1.0e-6)),
    }


def _tail_drawdown(daily: np.ndarray) -> tuple[float, float]:
    tail_count = max(1, int(math.ceil(0.10 * len(daily))))
    cumulative = np.cumsum(daily)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return float(np.sort(daily)[:tail_count].mean()), float(np.max(running_peak - cumulative))


def _aggregate_reports(node_reports: dict[str, dict], energy_mwh: float) -> dict:
    nodes = sorted(node_reports)
    dates = sorted(node_reports[nodes[0]]["daily"])
    daily_matrix = np.asarray([
        [node_reports[node]["daily"][date]["net_revenue"] for date in dates]
        for node in nodes
    ])
    macro_daily = daily_matrix.mean(axis=0)
    components = {}
    for field in ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty"):
        components[field] = float(np.mean([
            node_reports[node]["revenue_components"][field] / node_reports[node]["days"]
            for node in nodes
        ]))
    plan = np.concatenate([
        np.asarray([row["da_plan_mwh"] for row in node_reports[node]["hourly"]])
        for node in nodes
    ])
    actual = np.concatenate([
        np.asarray([row["actual_dispatch_mwh"] for row in node_reports[node]["hourly"]])
        for node in nodes
    ])
    throughput = float(np.mean([
        node_reports[node]["actual_throughput_mwh"] / node_reports[node]["days"]
        for node in nodes
    ]))
    tail, drawdown = _tail_drawdown(macro_daily)
    node_means = daily_matrix.mean(axis=1)
    return {
        "days": len(dates),
        "macro_mean_daily_net_revenue": float(macro_daily.mean()),
        "macro_daily_net_revenue": dict(zip(dates, macro_daily.tolist())),
        "components_per_node_day": components,
        "daily_throughput_mwh": throughput,
        "daily_equivalent_cycles": throughput / (2.0 * energy_mwh),
        "da_plan_hours": _state_count(plan),
        "actual_dispatch_hours": _state_count(actual),
        "tail_10pct_daily_revenue": tail,
        "maximum_drawdown_usd": drawdown,
        "positive_node_rate": float(np.mean(node_means > 0.0)),
        "worst_node_mean_daily_revenue": float(node_means.min()),
        "best_node_mean_daily_revenue": float(node_means.max()),
    }


def _forecast_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction = prediction.astype(np.float64)
    truth = truth.astype(np.float64)
    pred_order = np.argsort(prediction, axis=-1, kind="stable")
    truth_order = np.argsort(truth, axis=-1, kind="stable")
    pred_low = pred_order[..., 0]
    pred_high = pred_order[..., -1]
    truth_low = truth_order[..., 0]
    truth_high = truth_order[..., -1]
    selected_low = np.take_along_axis(truth, pred_low[..., None], axis=-1)[..., 0]
    selected_high = np.take_along_axis(truth, pred_high[..., None], axis=-1)[..., 0]
    true_range = np.max(truth, axis=-1) - np.min(truth, axis=-1)
    captured = selected_high - selected_low
    pair_i, pair_j = np.triu_indices(24, k=1)
    pred_sign = np.sign(prediction[..., pair_j] - prediction[..., pair_i])
    truth_sign = np.sign(truth[..., pair_j] - truth[..., pair_i])
    pairwise = np.mean(pred_sign == truth_sign, axis=-1)
    pred_p90, pred_p10 = np.quantile(prediction, [0.9, 0.1], axis=-1)
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "bias": float(np.mean(prediction - truth)),
        "mean_daily_p90_p10_spread": float(np.mean(pred_p90 - pred_p10)),
        "mean_daily_max_min_spread": float(np.mean(np.max(prediction, axis=-1) - np.min(prediction, axis=-1))),
        "pairwise_hour_order_accuracy": float(np.mean(pairwise)),
        "top_hour_exact_rate": float(np.mean(pred_high == truth_high)),
        "top_hour_within_1_rate": float(np.mean(np.abs(pred_high - truth_high) <= 1)),
        "bottom_hour_exact_rate": float(np.mean(pred_low == truth_low)),
        "bottom_hour_within_1_rate": float(np.mean(np.abs(pred_low - truth_low) <= 1)),
        "realized_spread_at_predicted_extrema": float(np.mean(captured)),
        "true_daily_max_min_spread": float(np.mean(true_range)),
        "realized_spread_capture_ratio": float(np.sum(captured) / np.sum(true_range)),
    }


def _cross_task_spread_metrics(tasks: dict[str, dict], economic_threshold: float) -> dict:
    predicted = (
        tasks["da"]["prediction"].astype(np.float64)
        - tasks["rt_at_da"]["prediction"].astype(np.float64)
    )
    realized = (
        tasks["da"]["target_da"].astype(np.float64)
        - tasks["rt_at_da"]["target_rt"].astype(np.float64)
    )
    nonzero = np.abs(realized) > 1.0e-9
    return {
        "mean_predicted_da_minus_rt_at_da": float(np.mean(predicted)),
        "mean_realized_da_minus_rt_at_da": float(np.mean(realized)),
        "mean_absolute_predicted_spread": float(np.mean(np.abs(predicted))),
        "mean_absolute_realized_spread": float(np.mean(np.abs(realized))),
        "absolute_spread_magnitude_ratio": float(
            np.mean(np.abs(predicted)) / np.mean(np.abs(realized))
        ),
        "spread_mae": float(np.mean(np.abs(predicted - realized))),
        "spread_direction_accuracy": float(np.mean(
            np.sign(predicted[nonzero]) == np.sign(realized[nonzero])
        )),
        "economic_threshold_usd_per_mwh": float(economic_threshold),
        "predicted_economic_spread_rate": float(
            np.mean(np.abs(predicted) >= economic_threshold)
        ),
        "realized_economic_spread_rate": float(
            np.mean(np.abs(realized) >= economic_threshold)
        ),
    }


def _comparison(candidate: dict, baseline: dict) -> dict:
    dates = sorted(baseline["macro_daily_net_revenue"])
    difference = np.asarray([
        candidate["macro_daily_net_revenue"][date]
        - baseline["macro_daily_net_revenue"][date]
        for date in dates
    ])
    ordered = np.sort(difference)
    return {
        "mean_daily_difference": float(difference.mean()),
        "7day_block_bootstrap_95ci": _block_bootstrap_ci(
            difference, seed=20261020, draws=2000
        ),
        "without_top_1_difference_day": float(ordered[:-1].mean()),
        "without_top_5_difference_days": float(ordered[:-5].mean()),
        "without_top_10_difference_days": float(ordered[:-10].mean()),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="data/results/allmarket_locked_2025_all_profit_v2"
    )
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="只补算预测与跨任务价差指标，不重复运行LP/MPC",
    )
    parser.add_argument(
        "--output", default="data/results/ercot_2025_da_profit_rank_amplitude_audit_v1/summary.json"
    )
    args = parser.parse_args()
    source = _validate_and_align(ROOT / args.input_dir)
    systems = _systems(source)
    cfg = PilotConfig.from_yaml(ROOT / args.config)
    output = ROOT / args.output
    if args.reuse_existing:
        if not output.exists():
            raise FileNotFoundError("--reuse-existing要求已有结算结果")
        result = json.loads(output.read_text(encoding="utf-8"))
        truth = systems["transformer"]["da"]["target_da"]
        for system, tasks in systems.items():
            result["systems"][system]["da_forecast"] = _forecast_metrics(
                tasks["da"]["prediction"], truth
            )
            result["systems"][system]["da_rt_at_da_spread"] = (
                _cross_task_spread_metrics(
                    tasks, cfg.bess_kappa / cfg.bess_eta
                )
            )
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "event": "prediction_metrics_augmented",
            "output": output.relative_to(ROOT).as_posix(),
        }, ensure_ascii=False))
        return
    reference_nodes = systems["transformer"]["da"]["nodes"].astype(str)
    jobs = []
    config_path = str((ROOT / args.config).resolve())
    reports = {system: {} for system in systems}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for system, tasks in systems.items():
            for node_index, node in enumerate(reference_nodes):
                node_input = align_node_task_predictions(
                    _node_record(tasks["da"], node_index),
                    _node_record(tasks["rt_at_da"], node_index),
                    _node_record(tasks["rt"], node_index),
                )
                jobs.append(pool.submit(
                    _worker, config_path, system, str(node), node_input
                ))
        for count, future in enumerate(as_completed(jobs), start=1):
            system, node, report = future.result()
            reports[system][node] = report
            print(json.dumps({
                "event": "node_complete", "completed": count,
                "total": len(jobs), "system": system, "node": node,
            }), flush=True)

    aggregates = {
        system: _aggregate_reports(node_reports, cfg.bess_energy_mwh)
        for system, node_reports in reports.items()
    }
    truth = systems["transformer"]["da"]["target_da"]
    forecast_metrics = {
        system: _forecast_metrics(tasks["da"]["prediction"], truth)
        for system, tasks in systems.items()
    }
    baseline = aggregates["transformer"]
    comparisons = {
        system: _comparison(item, baseline)
        for system, item in aggregates.items() if system != "transformer"
    }
    result = {
        "experiment": "ercot_2025_da_profit_rank_amplitude_posthoc_audit_v1",
        "year": 2025,
        "status": "complete",
        "selection_allowed": False,
        "post_original_test_followup": True,
        "trains_model": False,
        "counterfactuals_use_truth_to_construct_predictions": False,
        "systems": {
            system: {
                "label": SYSTEM_LABELS[system],
                "profit": aggregates[system],
                "da_forecast": forecast_metrics[system],
                "da_rt_at_da_spread": _cross_task_spread_metrics(
                    systems[system], cfg.bess_kappa / cfg.bess_eta
                ),
            }
            for system in systems
        },
        "paired_vs_transformer": comparisons,
        "interpretation_boundary": (
            "Diagnostic only. 2025 was already opened and these results cannot "
            "select or validate a new model."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "audit_complete", "output": output.relative_to(ROOT).as_posix()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
