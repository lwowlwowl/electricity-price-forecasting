#!/usr/bin/env python
"""解释三轮ERCOT global筛选为何价格MAE尚可但经济收益落后。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.global_evaluation import align_node_task_predictions  # noqa: E402
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _seasonal_archives,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


CHAMPION = {"da": "f1", "rt_at_da": "f0", "rt": "f0"}
GLOBAL_LABEL = "da_f1__rt_at_da_f0__rt_f0"
SEASONAL_LABEL = "da_seasonal__rt_at_da_seasonal__rt_seasonal"


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
        da_predictions.append(np.stack([
            np.asarray(value) for value in aligned["da_price_forecasts"].values()
        ]))
        rt_da_predictions.append(np.stack([
            np.asarray(value)
            for value in aligned["rt_at_da_price_forecasts"].values()
        ]))
        rt_predictions.append(np.asarray(aligned["rt_price_forecasts"])[:, 0])
        da_targets.append(np.asarray(aligned["realized_da_price"]))
        rt_targets.append(np.asarray(aligned["realized_rt_price"]))

    da_pred = np.stack(da_predictions)
    rt_da_pred = np.stack(rt_da_predictions)
    rt_pred = np.stack(rt_predictions)
    true_da = np.stack(da_targets)
    true_rt = np.stack(rt_targets)
    predicted_spread = da_pred.reshape(-1) - rt_da_pred.reshape(-1)
    true_spread = true_da.reshape(-1) - true_rt.reshape(-1)
    important = np.abs(true_spread) >= important_threshold
    correlation = float(np.corrcoef(predicted_spread, true_spread)[0, 1])
    return {
        "nodes": int(len(nodes)),
        "common_days": int(len(dates or [])),
        "da_mae": float(np.mean(np.abs(da_pred.reshape(-1) - true_da.reshape(-1)))),
        "rt_at_da_mae": float(
            np.mean(np.abs(rt_da_pred.reshape(-1) - true_rt.reshape(-1)))
        ),
        "rt_first_step_mae": float(
            np.mean(np.abs(rt_pred.reshape(-1) - true_rt.reshape(-1)))
        ),
        "predicted_spread_abs_mean": float(np.mean(np.abs(predicted_spread))),
        "predicted_spread_abs_q95": float(np.quantile(np.abs(predicted_spread), 0.95)),
        "predicted_spread_abs_q99": float(np.quantile(np.abs(predicted_spread), 0.99)),
        "true_spread_abs_mean": float(np.mean(np.abs(true_spread))),
        "true_spread_abs_q95": float(np.quantile(np.abs(true_spread), 0.95)),
        "true_spread_abs_q99": float(np.quantile(np.abs(true_spread), 0.99)),
        "spread_pearson_correlation": correlation,
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


def _revenue_distribution(system: dict) -> dict:
    daily = system["macro_daily_net_revenue"]
    values = np.asarray([daily[key] for key in sorted(daily)], dtype=np.float64)
    top = np.sort(values)[-10:]
    total = float(values.sum())
    return {
        "days": int(values.size),
        "total_usd_per_node": total,
        "mean_usd_per_node_day": float(values.mean()),
        "median_usd_per_node_day": float(np.median(values)),
        "top_10_days_total_usd_per_node": float(top.sum()),
        "top_10_days_share_of_total": float(top.sum() / total),
        "positive_day_rate": float(np.mean(values > 0.0)),
    }


def _training_curves(log_dir: Path) -> dict:
    reports = {}
    for path in sorted(log_dir.glob("global_ercot_huber*_inner_seed0.log")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "epoch" in value and "validation" in value:
                rows.append(value)
        maes = [float(row["validation"]["macro_mae"]) for row in rows]
        if not maes:
            continue
        best_index = int(np.argmin(maes))
        reports[path.stem] = {
            "epochs": [int(row["epoch"]) for row in rows],
            "validation_macro_mae": maes,
            "best_epoch": int(rows[best_index]["epoch"]),
            "best_at_last_budget_epoch": best_index == len(rows) - 1,
            "range_across_only_three_epochs": float(max(maes) - min(maes)),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir", default="data/results/global_f0_f1_predictions_v1"
    )
    parser.add_argument(
        "--backtest", default="data/results/global_continuous_mpc_backtest_v1.json"
    )
    parser.add_argument(
        "--log-dir", default="data/results/cloud_global_inner_seed0_screen_v1"
    )
    parser.add_argument(
        "--output", default="data/results/global_screen_failure_diagnosis_v1.json"
    )
    args = parser.parse_args()

    directory = ROOT / args.prediction_dir
    archives = {
        (family, task): _load_archive(directory, family, task)
        for family in ("f0", "f1") for task in ("da", "rt_at_da", "rt")
    }
    nodes = archives[("f0", "da")]["nodes"]
    archives.update(_seasonal_archives(archives, nodes))
    backtest = json.loads((ROOT / args.backtest).read_text(encoding="utf-8"))
    global_system = backtest["systems"][GLOBAL_LABEL]
    seasonal_system = backtest["systems"][SEASONAL_LABEL]
    global_daily = global_system["macro_daily_net_revenue"]
    seasonal_daily = seasonal_system["macro_daily_net_revenue"]
    ordered_days = sorted(global_daily)
    if ordered_days != sorted(seasonal_daily):
        raise ValueError("Global与Seasonal收益日期不一致")
    advantage = np.asarray([
        seasonal_daily[day] - global_daily[day] for day in ordered_days
    ], dtype=np.float64)
    largest_advantage = np.sort(advantage)[-10:]

    report = {
        "experiment": "global_seed0_three_epoch_failure_diagnosis_v1",
        "scope": "2023 inner validation; diagnostic only; no final selection",
        "global_candidate": CHAMPION,
        "forecast_diagnostics": {
            "global_champion": _system_forecast_diagnostics(
                archives, CHAMPION, nodes
            ),
            "seasonal": _system_forecast_diagnostics(
                archives,
                {"da": "seasonal", "rt_at_da": "seasonal", "rt": "seasonal"},
                nodes,
            ),
        },
        "revenue_distribution": {
            "global_champion": _revenue_distribution(global_system),
            "seasonal": _revenue_distribution(seasonal_system),
            "seasonal_minus_global": {
                "mean_usd_per_node_day": float(advantage.mean()),
                "median_usd_per_node_day": float(np.median(advantage)),
                "seasonal_wins_day_rate": float(np.mean(advantage > 0.0)),
                "top_10_advantage_days_share_of_total_gap": float(
                    largest_advantage.sum() / advantage.sum()
                ),
            },
        },
        "training_curves": _training_curves(ROOT / args.log_dir),
        "diagnosis": [
            "The global model has lower individual price MAE than Seasonal but nearly random DA-minus-RT spread direction.",
            "The three-epoch budget is not a convergence run; validation MAE oscillates strongly across adjacent epochs.",
            "A few scarcity days dominate the annual mean-revenue gap, while Seasonal has materially worse tail loss and drawdown.",
            "Next runs must retain multiple forecast checkpoints and choose the final three-task triplet using the frozen full settlement.",
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
