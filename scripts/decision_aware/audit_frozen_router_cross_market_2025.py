#!/usr/bin/env python
"""Apply the frozen ERCOT expert router to all 2025 markets without refitting.

This is a transfer audit, not a new model-selection stage.  The router and its
probability threshold are loaded from the ERCOT 2023+2024 follow-up artifact.
Only forecast and tail metrics are reported because non-ERCOT settlement
contracts have not been frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.tail_diagnostics import (  # noqa: E402
    daily_peak_report,
    event_detection_report,
    fit_tail_thresholds,
    price_slice_report,
    spread_slice_report,
)
from decision_aware.tail_expert_router import (  # noqa: E402
    build_feature_cube,
    router_probability,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _target_history,
)


MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=1e-5, equal_nan=True
    ):
        raise ValueError(f"{label}不一致")


def _validate(reference: dict[str, np.ndarray], item: dict[str, np.ndarray], label: str) -> None:
    for key in ("nodes", "dates"):
        if not np.array_equal(reference[key], item[key]):
            raise ValueError(f"{label}/{key}不一致")
    _same(reference["target"], item["target"], f"{label}/target")


def _system_report(
    tasks: dict[str, dict[str, np.ndarray]],
    thresholds,
) -> dict:
    report: dict[str, object] = {}
    for task in TASKS:
        lower = thresholds.da_lower_1pct if task == "da" else thresholds.rt_lower_1pct
        upper = thresholds.da_upper_1pct if task == "da" else thresholds.rt_upper_1pct
        report[task] = price_slice_report(
            tasks[task]["prediction"], tasks[task]["target"],
            lower_1pct=lower, upper_1pct=upper,
        )
    report["da_rt_at_da_spread"] = spread_slice_report(
        tasks["da"]["prediction"], tasks["rt_at_da"]["prediction"],
        tasks["da"]["target"], tasks["rt_at_da"]["target"], thresholds,
    )
    report["da_positive_spike_event"] = event_detection_report(
        tasks["da"]["prediction"], tasks["da"]["target"],
        threshold=thresholds.da_upper_1pct, direction="above", tolerance_hours=1,
    )
    report["da_negative_price_event"] = event_detection_report(
        tasks["da"]["prediction"], tasks["da"]["target"],
        threshold=0.0, direction="below", tolerance_hours=1,
    )
    report["da_daily_peak"] = daily_peak_report(
        tasks["da"]["prediction"], tasks["da"]["target"],
        positive_threshold=thresholds.da_upper_1pct,
    )
    return report


def _mixed(
    primary: dict[str, dict[str, np.ndarray]],
    alternate: dict[str, dict[str, np.ndarray]],
    select_alternate: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS:
        source = primary[task]
        prediction = source["prediction"]
        if task != "rt":
            prediction = np.where(
                select_alternate[..., None],
                alternate[task]["prediction"],
                source["prediction"],
            )
        result[task] = {**source, "prediction": prediction.astype(np.float32)}
    return result


def _write_profit_archives(
    output_dir: Path,
    systems: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    """Write evaluator-compatible archives under a common research contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for family, tasks in systems.items():
        for task, item in tasks.items():
            arrays: dict[str, np.ndarray] = {
                "nodes": item["nodes"],
                "dates": item["dates"],
                "prediction": item["prediction"],
                "target_rt": (
                    tasks["rt_at_da"]["target"] if task != "rt" else item["target"]
                ),
                "metadata": np.asarray(json.dumps({
                    "protocol": "frozen_ercot_router_cross_market_transfer_2025_v1",
                    "family": family,
                    "task": task,
                    "phase": "final",
                    "selection_allowed": False,
                    "economic_interpretation": "standardized_research_contract_only",
                }, ensure_ascii=False, sort_keys=True)),
            }
            if task == "da":
                arrays["target_da"] = item["target"]
            if task == "rt":
                arrays["executed_timestamps"] = item["executed_timestamps"]
                arrays["timezone"] = item["timezone"]
            np.savez_compressed(
                output_dir / f"{family}_{task}_final_seed0.npz", **arrays
            )


def _summary_rows(market: str, systems: dict[str, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, report in systems.items():
        peak = report["da_daily_peak"]
        positive = report["da_positive_spike_event"]
        negative = report["da_negative_price_event"]
        spread = report["da_rt_at_da_spread"]
        rows.append({
            "market": market,
            "system": name,
            "da_mae": report["da"]["all"]["mae"],
            "rt_at_da_mae": report["rt_at_da"]["all"]["mae"],
            "rolling_rt_mae": report["rt"]["all"]["mae"],
            "da_highest_1pct_mae": report["da"]["highest_1pct"]["mae"],
            "positive_spike_recall": positive["exact"]["recall"],
            "positive_spike_recall_within_1h": positive["within_1h"]["recall"],
            "positive_spike_magnitude_ratio": positive["magnitude_ratio_at_true_events"],
            "daily_peak_amplitude_ratio": peak["predicted_daily_peak_to_true_peak_ratio"],
            "daily_peak_within_1h": peak["within_1h_rate"],
            "negative_price_recall": negative["exact"]["recall"],
            "negative_price_precision": negative["exact"]["precision"],
            "spread_direction_accuracy": spread["all"]["direction_accuracy"],
            "extreme_spread_direction_accuracy": spread["abs_spread_top_1pct"]["direction_accuracy"],
            "extreme_spread_mae": spread["abs_spread_top_1pct"]["spread_mae"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ensemble-dir", default="data/results/allmarket_locked_2025_ensemble_v1"
    )
    parser.add_argument(
        "--dlinear-dir", default="data/results/allmarket_locked_2025_baselines_v1/dlinear"
    )
    parser.add_argument(
        "--router", default="data/results/tail_expert_router_v1/router_refit_2023_2024.joblib"
    )
    parser.add_argument(
        "--router-spec", default="data/results/tail_expert_router_v1/followup_router.json"
    )
    parser.add_argument(
        "--output", default="data/results/tail_expert_router_cross_market_2025_v1.json"
    )
    parser.add_argument(
        "--csv", default="data/results/tail_expert_router_cross_market_2025_v1.csv"
    )
    parser.add_argument(
        "--profit-root",
        default="data/results/tail_expert_router_cross_market_profit_inputs_v1",
    )
    args = parser.parse_args()

    ensemble_dir = ROOT / args.ensemble_dir
    dlinear_dir = ROOT / args.dlinear_dir
    model = joblib.load(ROOT / args.router)
    spec = json.loads((ROOT / args.router_spec).read_text(encoding="utf-8"))
    threshold = float(spec["threshold"])
    report = {
        "protocol": "frozen_ercot_router_cross_market_transfer_2025_v1",
        "status": "fixed_posthoc_transfer_audit_no_refit",
        "year": 2025,
        "selection_allowed": False,
        "router_training_markets": ["ERCOT"],
        "router_training_years": [2023, 2024],
        "router_threshold": threshold,
        "non_ercot_profit_claim_allowed": False,
        "markets": {},
    }
    rows: list[dict[str, object]] = []
    for market in MARKETS:
        lower = market.lower()
        transformer = {
            task: _load(ensemble_dir / f"{lower}_{task}_ensemble.npz") for task in TASKS
        }
        dlinear = {
            task: _load(dlinear_dir / f"{lower}_{task}.npz") for task in TASKS
        }
        for task in TASKS:
            _validate(transformer[task], dlinear[task], f"{market}/{task}/DLinear")
        _same(
            transformer["da"]["target"], transformer["rt_at_da"]["target_da"],
            f"{market}/DA与RT-at-DA的DA真值",
        )
        features, feature_names = build_feature_cube(
            transformer["da"]["prediction"],
            transformer["rt_at_da"]["prediction"],
            dlinear["da"]["prediction"],
            dlinear["rt_at_da"]["prediction"],
            transformer["da"]["dates"],
        )
        probability = router_probability(model, features.reshape(-1, features.shape[-1]))
        probability = probability.reshape(features.shape[:2])
        select_dlinear = probability >= threshold
        static_dlinear = np.ones_like(select_dlinear, dtype=bool)
        systems_raw = {
            "transformer": transformer,
            "dlinear": dlinear,
            "static_hybrid": _mixed(transformer, dlinear, static_dlinear),
            "frozen_router": _mixed(transformer, dlinear, select_dlinear),
        }
        _write_profit_archives(ROOT / args.profit_root / lower, systems_raw)
        train_da, train_rt = _target_history(
            market, transformer["da"]["nodes"].astype(str), 2024
        )
        thresholds = fit_tail_thresholds(train_da, train_rt)
        system_reports = {
            name: _system_report(tasks, thresholds) for name, tasks in systems_raw.items()
        }
        rows.extend(_summary_rows(market, system_reports))
        report["markets"][market] = {
            "nodes": transformer["da"]["nodes"].astype(str).tolist(),
            "dates": int(len(transformer["da"]["dates"])),
            "thresholds": thresholds.to_dict(),
            "route": {
                "dlinear_node_days": int(select_dlinear.sum()),
                "node_days": int(select_dlinear.size),
                "dlinear_share": float(select_dlinear.mean()),
                "probability_mean": float(probability.mean()),
                "probability_q10": float(np.quantile(probability, 0.10)),
                "probability_q90": float(np.quantile(probability, 0.90)),
                "features": list(feature_names),
            },
            "systems": system_reports,
        }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = ROOT / args.csv
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "event": "frozen_router_cross_market_audit_complete",
        "output": output.relative_to(ROOT).as_posix(),
        "csv": csv_path.relative_to(ROOT).as_posix(),
        "profit_root": args.profit_root,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
