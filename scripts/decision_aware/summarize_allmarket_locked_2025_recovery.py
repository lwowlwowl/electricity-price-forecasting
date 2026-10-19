#!/usr/bin/env python
"""Recovery wrapper for locked-2025 summary truth canonicalization.

The frozen summary remains unchanged.  Pooled XGBoost reconstructs targets via
an independent float32 inverse-transform path, which can differ from the
Transformer archive by a few ten-thousandths of one USD/MWh.  This wrapper keeps
all structural checks exact, rejects differences above 0.0003 USD/MWh, and then
uses the reference truth for every metric so systems are compared identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.decision_aware import summarize_allmarket_locked_2025 as frozen


TRUTH_RECOVERY_ATOL = 3e-4
AUDIT: list[dict[str, float | int | str]] = []


def canonicalize_numeric_truth(
    reference: np.ndarray,
    item: np.ndarray,
    label: str,
) -> np.ndarray:
    left = np.asarray(reference)
    right = np.asarray(item)
    if left.shape != right.shape:
        raise ValueError(f"{label}形状不一致")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    finite_or_matching_nan = np.isfinite(difference) | (np.isnan(left) & np.isnan(right))
    if not np.all(finite_or_matching_nan):
        raise ValueError(f"{label}包含不匹配的非有限值")
    finite_difference = difference[np.isfinite(difference)]
    maximum = float(np.max(finite_difference, initial=0.0))
    if maximum > TRUTH_RECOVERY_ATOL:
        raise ValueError(
            f"{label}差异超过float32恢复容差: max={maximum}, "
            f"atol={TRUTH_RECOVERY_ATOL}"
        )
    changed = int(np.count_nonzero(finite_difference))
    if changed:
        AUDIT.append(
            {
                "label": label,
                "changed_values": changed,
                "max_abs_difference": maximum,
            }
        )
    return left.copy()


def validate_and_canonicalize(
    reference: dict,
    item: dict,
    *,
    system: str,
    market: str,
    task: str,
) -> None:
    label = f"{system}/{market}/{task}"
    required = {"nodes", "dates", "prediction", "target"}
    missing = sorted(required.difference(item))
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")
    if item["prediction"].shape != item["target"].shape:
        raise ValueError(f"{label}预测与真值形状不一致")
    if not np.array_equal(reference["nodes"], item["nodes"]):
        raise ValueError(f"{label}节点不一致")
    if not np.array_equal(reference["dates"], item["dates"]):
        raise ValueError(f"{label}日期不一致")
    item["target"] = canonicalize_numeric_truth(
        reference["target"], item["target"], f"{label}真值"
    )
    for key in ("target_da", "executed_timestamps", "timezone"):
        if key not in reference:
            continue
        if key not in item:
            raise ValueError(f"{label}缺少对齐字段{key}")
        if key == "target_da":
            item[key] = canonicalize_numeric_truth(
                reference[key], item[key], f"{label}/{key}"
            )
        elif not np.array_equal(reference[key], item[key]):
            raise ValueError(f"{label}/{key}不一致")


def output_path_from_argv() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output", default="data/results/allmarket_locked_2025_summary_v2.json"
    )
    args, _ = parser.parse_known_args()
    return frozen.ROOT / args.output


def main() -> None:
    frozen._validate_archive_alignment = validate_and_canonicalize
    frozen.main()
    output = output_path_from_argv()
    report = json.loads(output.read_text(encoding="utf-8"))
    report["truth_alignment_recovery"] = {
        "protocol": "locked_2025_float32_truth_canonicalization_v1",
        "atol_usd_per_mwh": TRUTH_RECOVERY_ATOL,
        "canonical_truth_source": "transformer_ensemble_archive",
        "affected_comparisons": AUDIT,
        "maximum_abs_difference": max(
            (float(item["max_abs_difference"]) for item in AUDIT), default=0.0
        ),
        "prediction_arrays_modified": False,
        "selection_allowed": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "locked_2025_summary_truth_recovery_complete",
                "affected_comparisons": len(AUDIT),
                "maximum_abs_difference": report["truth_alignment_recovery"][
                    "maximum_abs_difference"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
