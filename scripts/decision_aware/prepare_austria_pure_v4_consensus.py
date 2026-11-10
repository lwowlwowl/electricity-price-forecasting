#!/usr/bin/env python
"""Create a fixed, label-free consensus of three complementary pure-v4 heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERS = (
    "data/results/pure_v4_local_meta_ridge_v1/predictions.npz",
    "data/results/pure_v4_context_kernel_v1/predictions.npz",
    "data/results/austria_pure_v4_multiscale_5y_v1/predictions.npz",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument("--output-dir", default="data/results/austria_pure_v4_consensus_v1")
    args = parser.parse_args()
    arrays = []
    dates = None
    for raw in args.members:
        path = ROOT / raw
        with np.load(path, allow_pickle=False) as payload:
            value = payload["da_usd_per_mwh"].astype(np.float64)
            if value.shape != (365, 24) or not np.isfinite(value).all():
                raise ValueError(f"invalid pure-v4 member {path}: {value.shape}")
            arrays.append(value)
            if "delivery_dates" in payload.files:
                candidate = payload["delivery_dates"].astype(str)
                if dates is None:
                    dates = candidate
                elif not np.array_equal(dates, candidate):
                    raise ValueError("consensus delivery dates differ")
    consensus = np.mean(np.stack(arrays), axis=0)
    if dates is None:
        dates = np.load(ROOT / DEFAULT_MEMBERS[-1])["delivery_dates"].astype(str)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz", delivery_dates=dates,
        da_usd_per_mwh=consensus.astype(np.float32),
    )
    summary = {
        "method": "pure-v4-fixed-three-branch-consensus",
        "weights": [1.0 / len(arrays)] * len(arrays),
        "members": list(args.members),
        "uses_exaa": False,
        "uses_external_model_predictions": False,
        "uses_austria_labels_for_training_or_selection": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
