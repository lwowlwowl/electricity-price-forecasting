#!/usr/bin/env python
"""汇总无锚点任务专属损失三模型，并执行对XGBoost的冻结晋级门槛。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.global_evaluation import align_node_task_predictions  # noqa: E402
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _node_baseline_archives,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


TRIPLET_PATTERN = re.compile(
    r"^da_(?P<da>.+)__rt_at_da_(?P<rt_at_da>.+)__rt_(?P<rt>.+)$"
)


def _parse_triplet(label: str) -> dict[str, str]:
    match = TRIPLET_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"无法解析三元组标签: {label}")
    return match.groupdict()


def _maximum_drawdown(daily: dict[str, float]) -> float:
    values = np.asarray([daily[key] for key in sorted(daily)], dtype=np.float64)
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return float(np.max(running_peak - cumulative, initial=0.0))


def _reshape_daily_prices(values, day_count: int, label: str) -> np.ndarray:
    """把对齐器的小时展平值恢复为[day,24]，并拒绝静默错位。"""
    array = np.asarray(values)
    expected = int(day_count) * 24
    if array.size != expected:
        raise ValueError(
            f"{label}元素数必须为day_count*24={expected}，实际为{array.size}"
        )
    return array.reshape(day_count, 24)


def _system_forecast_diagnostics(
    archives: dict, family_by_task: dict, nodes, important_threshold: float = 6.0
) -> dict:
    da_predictions, rt_da_predictions, rt_predictions = [], [], []
    da_targets, rt_targets = [], []
    dates = None
    for node_index, _ in enumerate(nodes):
        aligned = align_node_task_predictions(
            _node_record(archives[(family_by_task["da"], "da")], node_index),
            _node_record(
                archives[(family_by_task["rt_at_da"], "rt_at_da")], node_index
            ),
            _node_record(archives[(family_by_task["rt"], "rt")], node_index),
        )
        if dates is None:
            dates = aligned["dates"]
        elif dates != aligned["dates"]:
            raise ValueError("节点间共同日期不一致")
        day_count = len(aligned["dates"])
        da_predictions.append(np.stack([
            np.asarray(value) for value in aligned["da_price_forecasts"].values()
        ]))
        rt_da_predictions.append(np.stack([
            np.asarray(value)
            for value in aligned["rt_at_da_price_forecasts"].values()
        ]))
        rt_predictions.append(_reshape_daily_prices(
            np.asarray(aligned["rt_price_forecasts"])[:, 0],
            day_count,
            "滚动RT第一步预测",
        ))
        da_targets.append(_reshape_daily_prices(
            aligned["realized_da_price"], day_count, "DA真值"
        ))
        rt_targets.append(_reshape_daily_prices(
            aligned["realized_rt_price"], day_count, "RT真值"
        ))

    da_pred = np.stack(da_predictions)
    rt_da_pred = np.stack(rt_da_predictions)
    rt_pred = np.stack(rt_predictions)
    true_da = np.stack(da_targets)
    true_rt = np.stack(rt_targets)
    predicted_spread = da_pred.reshape(-1) - rt_da_pred.reshape(-1)
    true_spread = true_da.reshape(-1) - true_rt.reshape(-1)
    important = np.abs(true_spread) >= important_threshold
    da_peak_threshold = float(np.quantile(true_da, 0.95))
    rt_peak_threshold = float(np.quantile(true_rt, 0.95))
    da_peak_mask = true_da >= da_peak_threshold
    rt_peak_mask = true_rt >= rt_peak_threshold
    return {
        "nodes": int(len(nodes)),
        "common_days": int(len(dates or [])),
        "da_mae": float(np.mean(np.abs(da_pred - true_da))),
        "rt_at_da_mae": float(np.mean(np.abs(rt_da_pred - true_rt))),
        "rt_first_step_mae": float(np.mean(np.abs(rt_pred - true_rt))),
        "da_peak_hour_accuracy": float(np.mean(
            np.argmax(da_pred, axis=-1) == np.argmax(true_da, axis=-1)
        )),
        "rt_at_da_peak_hour_accuracy": float(np.mean(
            np.argmax(rt_da_pred, axis=-1) == np.argmax(true_rt, axis=-1)
        )),
        "da_true_price_q95": da_peak_threshold,
        "da_q95_peak_mae": float(np.mean(np.abs(da_pred - true_da)[da_peak_mask])),
        "rt_true_price_q95": rt_peak_threshold,
        "rt_at_da_q95_peak_mae": float(
            np.mean(np.abs(rt_da_pred - true_rt)[rt_peak_mask])
        ),
        "rt_first_step_q95_peak_mae": float(
            np.mean(np.abs(rt_pred - true_rt)[rt_peak_mask])
        ),
        "predicted_spread_abs_mean": float(np.mean(np.abs(predicted_spread))),
        "predicted_spread_abs_q95": float(np.quantile(np.abs(predicted_spread), 0.95)),
        "predicted_spread_abs_q99": float(np.quantile(np.abs(predicted_spread), 0.99)),
        "true_spread_abs_mean": float(np.mean(np.abs(true_spread))),
        "true_spread_abs_q95": float(np.quantile(np.abs(true_spread), 0.95)),
        "true_spread_abs_q99": float(np.quantile(np.abs(true_spread), 0.99)),
        "spread_pearson_correlation": float(
            np.corrcoef(predicted_spread, true_spread)[0, 1]
        ),
        "spread_direction_accuracy": float(
            np.mean(np.sign(predicted_spread) == np.sign(true_spread))
        ),
        "important_spread_abs_true_threshold": float(important_threshold),
        "important_spread_fraction": float(np.mean(important)),
        "important_spread_direction_accuracy": (
            float(np.mean(
                np.sign(predicted_spread[important])
                == np.sign(true_spread[important])
            ))
            if important.any() else None
        ),
    }


def _economic_summary(system: dict) -> dict:
    components = system["macro_mean_daily_revenue_components"]
    return {
        "macro_mean_daily_revenue": float(system["macro_mean_daily_revenue"]),
        "macro_positive_day_rate": float(system["macro_positive_day_rate"]),
        "macro_min_daily_revenue": float(system["macro_min_daily_revenue"]),
        "macro_maximum_drawdown_usd": _maximum_drawdown(
            system["macro_daily_net_revenue"]
        ),
        "macro_mean_daily_deviation_penalty": float(
            components["deviation_penalty"]
        ),
        "macro_mean_daily_degradation_cost": float(components["degradation_cost"]),
        "macro_mean_daily_revenue_without_top_5_days": float(
            system["macro_mean_daily_revenue_without_top_5_days"]
        ),
        "top_5_days_total_revenue_share": float(
            system["top_5_days_total_revenue_share"]
        ),
    }


def _best_candidate_label(ranking: list[dict], family_prefix: str) -> str:
    for record in ranking:
        mapping = _parse_triplet(record["triplet"])
        if all(value.startswith(family_prefix) for value in mapping.values()):
            return record["triplet"]
    raise ValueError(f"结算排名中没有纯{family_prefix}三元组")


def summarize(
    *, prediction_dir: Path, backtest: dict, families: tuple[str, ...],
    family_prefix: str = "taskloss",
) -> dict:
    archives = {
        (family, task): _load_archive(prediction_dir, family, task)
        for family in families for task in ("da", "rt_at_da", "rt")
    }
    nodes = archives[(families[0], "da")]["nodes"]
    archives.update(
        _node_baseline_archives(
            archives, nodes, "xgboost_robust", families[0], "inner"
        )
    )
    best_label = _best_candidate_label(backtest["ranking"], family_prefix)
    best_mapping = _parse_triplet(best_label)
    xgb_label = (
        "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
    )
    best_system = backtest["systems"][best_label]
    xgb_system = backtest["systems"][xgb_label]
    paired = backtest["paired_vs_baselines"]["xgboost_robust"][best_label]
    candidate_forecast = _system_forecast_diagnostics(
        archives, best_mapping, nodes
    )
    xgb_mapping = {task: "xgboost_robust" for task in ("da", "rt_at_da", "rt")}
    xgb_forecast = _system_forecast_diagnostics(archives, xgb_mapping, nodes)
    candidate_economic = _economic_summary(best_system)
    xgb_economic = _economic_summary(xgb_system)
    paired_ci = paired["7day_block_bootstrap_95ci"]
    gates = {
        "revenue_point_estimate_beats_xgboost": (
            candidate_economic["macro_mean_daily_revenue"]
            > xgb_economic["macro_mean_daily_revenue"]
        ),
        "paired_95ci_lower_bound_above_zero": float(paired_ci[0]) > 0.0,
        "maximum_drawdown_no_worse_than_xgboost": (
            candidate_economic["macro_maximum_drawdown_usd"]
            <= xgb_economic["macro_maximum_drawdown_usd"]
        ),
        "da_mae_no_worse_than_xgboost": (
            candidate_forecast["da_mae"] <= xgb_forecast["da_mae"]
        ),
        "rt_at_da_mae_no_worse_than_xgboost": (
            candidate_forecast["rt_at_da_mae"] <= xgb_forecast["rt_at_da_mae"]
        ),
        "rt_mae_no_worse_than_xgboost": (
            candidate_forecast["rt_first_step_mae"]
            <= xgb_forecast["rt_first_step_mae"]
        ),
    }
    return {
        "experiment": f"global_ercot_{family_prefix}_v1_summary",
        "status": "inner_validation_candidate_not_final",
        "split": "2023 inner validation only",
        "2024_used": False,
        "2025_used": False,
        "candidate_family_prefix": family_prefix,
        "best_candidate_triplet": best_label,
        "best_candidate_mapping": best_mapping,
        "forecast_diagnostics": {
            family_prefix: candidate_forecast,
            "xgboost_robust": xgb_forecast,
        },
        "economic_diagnostics": {
            family_prefix: candidate_economic,
            "xgboost_robust": xgb_economic,
            f"paired_{family_prefix}_minus_xgboost": paired,
        },
        "promotion_gates": gates,
        "ready_for_multiseed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir", default="data/results/global_taskloss_predictions_v1"
    )
    parser.add_argument(
        "--backtest", default="data/results/global_taskloss_continuous_mpc_v1.json"
    )
    parser.add_argument("--families", default="taskloss0,taskloss1")
    parser.add_argument("--family-prefix", default="taskloss")
    parser.add_argument(
        "--output", default="data/results/global_taskloss_summary_v1.json"
    )
    args = parser.parse_args()
    families = tuple(value.strip() for value in args.families.split(",") if value.strip())
    if not families:
        raise ValueError("families不能为空")
    backtest = json.loads((ROOT / args.backtest).read_text(encoding="utf-8"))
    report = summarize(
        prediction_dir=ROOT / args.prediction_dir,
        backtest=backtest,
        families=families,
        family_prefix=args.family_prefix,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved": output.relative_to(ROOT).as_posix(),
        "best_candidate_triplet": report["best_candidate_triplet"],
        "ready_for_multiseed": report["ready_for_multiseed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
