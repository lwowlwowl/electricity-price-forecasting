#!/usr/bin/env python
"""Build a preregistered equal-weight ensemble from complete profit archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_ensemble(
    directory: Path,
    members: tuple[str, ...],
    output_family: str,
    *,
    phase: str = "inner",
    weights: tuple[float, ...] | None = None,
    selection_by_profit: bool = False,
) -> None:
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("members must contain at least two unique families")
    if not output_family or output_family in members:
        raise ValueError("output family must be non-empty and distinct from members")
    if weights is None:
        normalized_weights = np.full(len(members), 1.0 / len(members))
    else:
        if len(weights) != len(members):
            raise ValueError("weights must have one value per member")
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(normalized_weights).all() or np.any(normalized_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        if float(normalized_weights.sum()) <= 0.0:
            raise ValueError("at least one weight must be positive")
        normalized_weights = normalized_weights / normalized_weights.sum()
    for value in (*members, output_family):
        if any(character.isspace() for character in value):
            raise ValueError("family names cannot contain whitespace")

    directory.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        records = [
            _load(directory / f"{family}_{task}_{phase}_seed0.npz")
            for family in members
        ]
        reference = records[0]
        for family, record in zip(members[1:], records[1:]):
            for key in ("nodes", "dates", "executed_timestamps", "timezone"):
                if key in reference or key in record:
                    if key not in reference or key not in record or not np.array_equal(
                        reference[key], record[key]
                    ):
                        raise ValueError(f"{task}/{family}: {key} mismatch")
            for key in ("target_da", "target_rt"):
                if key in reference or key in record:
                    if key not in reference or key not in record or not np.allclose(
                        reference[key], record[key], rtol=0.0, atol=5e-4
                    ):
                        raise ValueError(f"{task}/{family}: {key} mismatch")
            if record["prediction"].shape != reference["prediction"].shape:
                raise ValueError(f"{task}/{family}: prediction shape mismatch")

        prediction = np.average(
            np.stack([item["prediction"].astype(np.float64) for item in records]),
            axis=0,
            weights=normalized_weights,
        ).astype(np.float32)
        arrays = {
            key: value
            for key, value in reference.items()
            if key not in {"prediction", "metadata"}
        }
        arrays["prediction"] = prediction
        arrays["metadata"] = json.dumps(
            {
                "family": output_family,
                "task": task,
                "phase": phase,
                "protocol": (
                    "fixed_equal_weight_seed_ensemble"
                    if weights is None else "fixed_convex_weight_family_ensemble"
                ),
                "members": list(members),
                "weights": normalized_weights.tolist(),
                "seed_selection_by_profit": bool(selection_by_profit),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        np.savez_compressed(
            directory / f"{output_family}_{task}_{phase}_seed0.npz", **arrays
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default="data/results/global_implicit_qp_profit_predictions_v1",
    )
    parser.add_argument("--members", required=True)
    parser.add_argument(
        "--weights", help="optional comma-separated nonnegative member weights"
    )
    parser.add_argument(
        "--selection-by-profit", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--output-family", default="implicitqp_decoder_balanced_ensemble"
    )
    parser.add_argument("--phase", choices=("inner", "outer", "final"), default="inner")
    args = parser.parse_args()
    members = tuple(value.strip() for value in args.members.split(",") if value.strip())
    weights = (
        tuple(float(value.strip()) for value in args.weights.split(","))
        if args.weights else None
    )
    build_ensemble(
        ROOT / args.prediction_dir,
        members,
        args.output_family,
        phase=args.phase,
        weights=weights,
        selection_by_profit=args.selection_by_profit,
    )
    print(json.dumps({
        "event": "fixed_family_ensemble_ready",
        "family": args.output_family,
        "members": members,
        "weights": weights,
    }))


if __name__ == "__main__":
    main()
