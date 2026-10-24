#!/usr/bin/env python
"""Audit the frozen tail-expert router with profit-primary paired evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROUTER_LABEL = "da_router_followup__rt_at_da_router_followup__rt_router_followup"
HYBRID_LABEL = "da_dlinear__rt_at_da_dlinear__rt_ensemble"
DLINEAR_LABEL = "da_dlinear__rt_at_da_dlinear__rt_dlinear"
XGB_LABEL = "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"
TRANSFORMER_LABEL = "da_ensemble__rt_at_da_ensemble__rt_ensemble"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _daily(system: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in system["macro_daily_net_revenue"].items()}


def _block_ci(values: np.ndarray, *, seed: int, samples: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    block = 7
    starts = np.arange(max(1, len(values) - block + 1))
    block_count = int(np.ceil(len(values) / block))
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = rng.choice(starts, size=block_count, replace=True)
        draw = np.concatenate([values[start:min(start + block, len(values))] for start in chosen])
        means[sample] = draw[:len(values)].mean()
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def _maximum_drawdown(values: np.ndarray) -> float:
    cumulative = np.cumsum(np.asarray(values, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    path = np.concatenate(([0.0], cumulative))
    return float(np.max(peak - path))


def _paired(router: dict[str, Any], baseline: dict[str, Any], name: str) -> dict[str, Any]:
    router_daily = _daily(router)
    baseline_daily = _daily(baseline)
    dates = sorted(set(router_daily) & set(baseline_daily))
    router_values = np.asarray([router_daily[date] for date in dates])
    baseline_values = np.asarray([baseline_daily[date] for date in dates])
    difference = router_values - baseline_values
    node_differences: dict[str, float] = {}
    for node, report in router["nodes"].items():
        other = baseline["nodes"][node]
        common = sorted(set(report["daily_net_revenue"]) & set(other["daily_net_revenue"]))
        node_differences[node] = float(np.mean([
            float(report["daily_net_revenue"][date])
            - float(other["daily_net_revenue"][date]) for date in common
        ]))
    removed: dict[str, Any] = {}
    ranked_dates = [dates[index] for index in np.argsort(router_values)[::-1]]
    for count in (1, 5, 10):
        retained = [date for date in dates if date not in set(ranked_dates[:count])]
        retained_diff = np.asarray([
            router_daily[date] - baseline_daily[date] for date in retained
        ])
        removed[str(count)] = {
            "mean_difference": float(retained_diff.mean()),
            "positive_day_rate": float((retained_diff > 0.0).mean()),
        }
    return {
        "baseline": name,
        "days": len(dates),
        "router_mean": float(router_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "mean_difference": float(difference.mean()),
        "difference_7day_block_bootstrap_95ci": _block_ci(
            difference, seed=20261020 + len(name)
        ),
        "positive_day_rate": float((difference > 0.0).mean()),
        "nodes_improved": int(sum(value > 0.0 for value in node_differences.values())),
        "node_count": len(node_differences),
        "node_differences": node_differences,
        "without_router_top_days": removed,
    }


def _aligned_prediction(archive: dict[str, np.ndarray], dates: list[str]) -> np.ndarray:
    index = {str(value): idx for idx, value in enumerate(archive["dates"].astype(str))}
    if not set(dates) <= set(index):
        raise ValueError("预测文件缺少结算日期")
    return archive["prediction"][:, [index[date] for date in dates]]


def _tail_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    price_threshold: np.ndarray,
    ramp_threshold: np.ndarray,
) -> dict[str, Any]:
    actual_spike = target.max(axis=-1) > price_threshold[:, None]
    predicted_spike = prediction.max(axis=-1) > price_threshold[:, None]
    actual_negative = target.min(axis=-1) < 0.0
    predicted_negative = prediction.min(axis=-1) < 0.0
    actual_ramp = np.max(np.abs(np.diff(target, axis=-1)), axis=-1) > ramp_threshold[:, None]
    predicted_ramp = np.max(np.abs(np.diff(prediction, axis=-1)), axis=-1) > ramp_threshold[:, None]
    spike_rows = np.where(actual_spike)
    actual_peak = target[spike_rows].max(axis=-1)
    predicted_peak = prediction[spike_rows].max(axis=-1)
    actual_hour = target[spike_rows].argmax(axis=-1)
    predicted_hour = prediction[spike_rows].argmax(axis=-1)
    hour_distance = np.minimum(
        np.abs(actual_hour - predicted_hour), 24 - np.abs(actual_hour - predicted_hour)
    )
    return {
        "positive_spike_event_count": int(actual_spike.sum()),
        "positive_spike_recall": float(predicted_spike[actual_spike].mean()),
        "positive_spike_false_positive_rate": float(predicted_spike[~actual_spike].mean()),
        "positive_spike_peak_amplitude_ratio_mean": float(
            np.mean(predicted_peak / np.maximum(actual_peak, 1e-6))
        ),
        "positive_spike_peak_within_1h_rate": float((hour_distance <= 1).mean()),
        "negative_event_count": int(actual_negative.sum()),
        "negative_recall": float(predicted_negative[actual_negative].mean()),
        "negative_false_positive_rate": float(predicted_negative[~actual_negative].mean()),
        "rapid_ramp_event_count": int(actual_ramp.sum()),
        "rapid_ramp_recall": float(predicted_ramp[actual_ramp].mean()),
        "rapid_ramp_false_positive_rate": float(predicted_ramp[~actual_ramp].mean()),
    }


def _tail_attribution(
    router: dict[str, Any],
    hybrid: dict[str, Any],
    history_root: Path,
    router_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    history = [
        _archive(history_root / f"universal_da_{phase}_seed0.npz")
        for phase in ("inner", "outer")
    ]
    historical_target = np.concatenate([item["target_da"] for item in history], axis=1)
    price_threshold = np.quantile(historical_target, 0.99, axis=(1, 2))
    historical_ramp = np.abs(np.diff(historical_target, axis=-1))
    ramp_threshold = np.quantile(historical_ramp, 0.99, axis=(1, 2))

    router_da = _archive(router_root / "router_followup_da_final_seed0.npz")
    transformer_da = _archive(final_root / "ensemble_da_final_seed0.npz")
    dlinear_da = _archive(final_root / "dlinear_da_final_seed0.npz")
    dates = sorted(_daily(router))
    date_index = {str(value): idx for idx, value in enumerate(router_da["dates"].astype(str))}
    indices = [date_index[date] for date in dates]
    target = router_da["target_da"][:, indices]
    router_prediction = router_da["prediction"][:, indices]
    transformer_prediction = _aligned_prediction(transformer_da, dates)
    dlinear_prediction = _aligned_prediction(dlinear_da, dates)

    actual_spike = target.max(axis=-1) > price_threshold[:, None]
    actual_negative = target.min(axis=-1) < 0.0
    actual_ramp = np.max(np.abs(np.diff(target, axis=-1)), axis=-1) > ramp_threshold[:, None]
    ordinary = ~(actual_spike | actual_negative | actual_ramp)
    switch = np.max(np.abs(router_prediction - transformer_prediction), axis=-1) > 1e-6

    nodes = router_da["nodes"].astype(str)
    router_profit = np.asarray([
        [float(router["nodes"][node]["daily_net_revenue"][date]) for date in dates]
        for node in nodes
    ])
    hybrid_profit = np.asarray([
        [float(hybrid["nodes"][node]["daily_net_revenue"][date]) for date in dates]
        for node in nodes
    ])

    groups = {
        "positive_spike": actual_spike,
        "negative_price": actual_negative,
        "rapid_ramp": actual_ramp,
        "ordinary": ordinary,
    }
    attribution = {}
    for name, mask in groups.items():
        attribution[name] = {
            "node_days": int(mask.sum()),
            "router_mean_profit": float(router_profit[mask].mean()),
            "hybrid_mean_profit": float(hybrid_profit[mask].mean()),
            "mean_profit_difference": float((router_profit - hybrid_profit)[mask].mean()),
            "hybrid_expert_selection_rate": float(switch[mask].mean()),
        }
    prediction_archives = {
        "router": router_da,
        "static_hybrid": dlinear_da,
        "dlinear": dlinear_da,
        "transformer": transformer_da,
    }
    for name in (
        "per_node_xgboost", "pooled_xgboost", "timesfm3", "timesfm25",
        "seasonal", "oracle",
    ):
        path = final_root / f"{name}_da_final_seed0.npz"
        if path.exists():
            prediction_archives[name] = _archive(path)
    prediction_metrics = {
        name: _tail_prediction_metrics(
            _aligned_prediction(archive, dates), target, price_threshold, ramp_threshold
        )
        for name, archive in prediction_archives.items()
    }
    # Preserve the exact routed array rather than reloading/alignment only for
    # clarity in case a future archive contains unmatched boundary dates.
    prediction_metrics["router"] = _tail_prediction_metrics(
        router_prediction, target, price_threshold, ramp_threshold
    )
    return {
        "threshold_definition": {
            "positive_spike": "node-level q99 of 2023+2024 out-of-sample DA truth",
            "rapid_ramp": "node-level q99 of 2023+2024 absolute hourly DA ramps",
            "negative_price": "at least one negative DA hour",
        },
        "groups": attribution,
        "prediction_metrics": prediction_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router-root", default="data/results/tail_expert_router_v1"
    )
    parser.add_argument(
        "--history-root", default="data/results/historical_transformer_dlinear_profit_v1"
    )
    parser.add_argument(
        "--final-root", default="data/results/allmarket_locked_2025_all_profit_v2"
    )
    parser.add_argument(
        "--cross-task", default="data/results/allmarket_locked_2025_transformer_dlinear_cross_task_screen_v1.json"
    )
    parser.add_argument(
        "--all-baselines", default="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json"
    )
    parser.add_argument(
        "--output", default="data/results/tail_expert_router_v1/followup_2025_audit_v1.json"
    )
    args = parser.parse_args()

    router_root = ROOT / args.router_root
    router_report = _json(router_root / "followup_2025_mpc_v1.json")
    router = router_report["systems"][ROUTER_LABEL]
    cross = _json(ROOT / args.cross_task)["systems"]
    all_baselines = _json(ROOT / args.all_baselines)["systems"]
    baselines = {
        "static_hybrid": cross[HYBRID_LABEL],
        "dlinear": cross[DLINEAR_LABEL],
        "per_node_xgboost": all_baselines[XGB_LABEL],
        "transformer": cross[TRANSFORMER_LABEL],
    }
    comparisons = {
        name: _paired(router, baseline, name) for name, baseline in baselines.items()
    }
    router_daily_map = _daily(router)
    router_dates = sorted(router_daily_map)
    router_daily = np.asarray([router_daily_map[date] for date in router_dates], dtype=np.float64)
    router_ranked = np.sort(router_daily)
    summary = {
        "protocol": "tail_expert_router_v1",
        "status": "post_original_2025_test_fixed_followup",
        "primary_metric": "complete net revenue after degradation and deviation penalty",
        "router_macro_mean_daily_revenue": float(router["macro_mean_daily_revenue"]),
        "router_7day_block_bootstrap_95ci": router["macro_mean_daily_revenue_7day_block_bootstrap_95ci"],
        "router_maximum_drawdown_macro_usd": _maximum_drawdown(router_daily),
        "router_mean_without_top_days": {
            str(count): float(router_ranked[:-count].mean()) for count in (1, 5, 10)
        },
        "router_components": router["macro_mean_daily_revenue_components"],
        "comparisons": comparisons,
        "tail_attribution": _tail_attribution(
            router,
            baselines["static_hybrid"],
            ROOT / args.history_root,
            router_root,
            ROOT / args.final_root,
        ),
        "claim_boundary": {
            "nominal_best": all(item["mean_difference"] > 0.0 for item in comparisons.values()),
            "closest_baseline_difference_ci_excludes_zero": not (
                comparisons["static_hybrid"]["difference_7day_block_bootstrap_95ci"][0]
                <= 0.0
                <= comparisons["static_hybrid"]["difference_7day_block_bootstrap_95ci"][1]
            ),
            "further_2025_tuning_allowed": False,
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "tail_expert_router_audit_saved",
        "output": str(output.relative_to(ROOT)),
        "router_profit": summary["router_macro_mean_daily_revenue"],
        "comparisons": {
            name: {
                "difference": value["mean_difference"],
                "ci": value["difference_7day_block_bootstrap_95ci"],
            } for name, value in comparisons.items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
