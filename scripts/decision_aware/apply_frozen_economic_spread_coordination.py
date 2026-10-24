#!/usr/bin/env python
"""Apply the already-frozen economic spread coordinator to prediction archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.build_economic_spread_coordination import (  # noqa: E402
    apply_coordinator,
    economic_state,
    load_archive,
    raw_features,
    save_archive,
    validate_triplet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-family", default="ensemble")
    parser.add_argument("--candidate-family", default="econcoord")
    parser.add_argument("--phase", default="final")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    suffix = f"{args.phase}_seed{args.seed}.npz"
    tasks = {
        task: load_archive(input_dir / f"{args.source_family}_{task}_{suffix}")
        for task in ("da", "rt_at_da", "rt")
    }
    validate_triplet(tasks)
    artifact = json.loads((ROOT / args.model).read_text(encoding="utf-8"))
    model = artifact["model"]
    da = tasks["da"]["prediction"].astype(np.float64)
    rt_at_da = tasks["rt_at_da"]["prediction"].astype(np.float64)
    features = raw_features(da, rt_at_da)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    z = (features - mean) / std
    calibrated, diagnostics = apply_coordinator(da, rt_at_da, model)

    output_dir.mkdir(parents=True, exist_ok=True)
    for family in (args.source_family, args.candidate_family):
        for task, source in tasks.items():
            values = dict(source)
            if family == args.candidate_family and task == "rt_at_da":
                values["prediction"] = calibrated
            metadata = {
                "family": family,
                "source_family": args.source_family,
                "task": task,
                "phase": args.phase,
                "seed": args.seed,
                "method": (
                    "frozen_economic_spread_coordination_v1"
                    if family == args.candidate_family
                    else "unchanged_final_ensemble"
                ),
                "fit_phase": "ercot_inner_2023",
                "confirmation_phase": "ercot_outer_2024",
                "selection_allowed": False,
                "post_original_2025_followup": args.phase == "final",
                "realized_target_used_at_inference": False,
            }
            values["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
            save_archive(output_dir / f"{family}_{task}_{suffix}", values)

    predicted_state = diagnostics["predicted_state"].reshape(-1)
    threshold = float(model["threshold"])
    report = {
        "experiment": "apply_frozen_economic_spread_coordination_v1",
        "selection_allowed": False,
        "phase": args.phase,
        "source_family": args.source_family,
        "candidate_family": args.candidate_family,
        "compatibility": {
            "feature_standardized_mean": z.mean(axis=0).tolist(),
            "feature_standardized_std": z.std(axis=0).tolist(),
            "absolute_z_above_5_rate": float(np.mean(np.abs(z) > 5.0)),
            "maximum_absolute_z": float(np.max(np.abs(z))),
            "nan_or_infinite_count": int(np.sum(~np.isfinite(z))),
        },
        "predicted_state_counts": {
            str(state): int(np.sum(predicted_state == state)) for state in (-1, 0, 1)
        },
        "original_mean_absolute_spread": float(
            np.mean(np.abs(diagnostics["original_spread"]))
        ),
        "coordinated_mean_absolute_spread": float(
            np.mean(np.abs(diagnostics["coordinated_spread"]))
        ),
        "original_economic_crossing_rate": float(
            np.mean(np.abs(diagnostics["original_spread"]) > threshold)
        ),
        "coordinated_economic_crossing_rate": float(
            np.mean(np.abs(diagnostics["coordinated_spread"]) > threshold)
        ),
    }
    if "target_da" in tasks["da"] and "target_rt" in tasks["rt_at_da"]:
        realized = (
            tasks["da"]["target_da"].astype(np.float64)
            - tasks["rt_at_da"]["target_rt"].astype(np.float64)
        )
        state = economic_state(realized, threshold)
        report["posthoc_truth_metrics"] = {
            "state_accuracy": float(np.mean(diagnostics["predicted_state"] == state)),
            "original_spread_mae": float(
                np.mean(np.abs(diagnostics["original_spread"] - realized))
            ),
            "coordinated_spread_mae": float(
                np.mean(np.abs(diagnostics["coordinated_spread"] - realized))
            ),
        }
    report_path = output_dir / "application_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
