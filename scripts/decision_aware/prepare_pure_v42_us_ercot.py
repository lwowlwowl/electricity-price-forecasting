#!/usr/bin/env python
"""Prepare an exact-v4 ERCOT archive with the frozen memory correction.

The memory rule was selected before the Austria test.  Here its correction,
expressed in USD/MWh, is added to the official v4 DA archive.  The official
v4 RT-at-DA and rolling-RT legs are copied byte-for-value at array level so a
controlled three-leg settlement can change DA only.
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

from scripts.decision_aware.evaluate_pure_v42_memory_us_markets import (  # noqa: E402
    _node_arrays,
)
from scripts.decision_aware.research_pure_v42_memory_residual import (  # noqa: E402
    _load,
    _memory_predict,
)


TASKS = ("da", "rt_at_da", "rt")
SOURCE_FAMILY = "implicitqp_robust_seed0_twothirds_v4"
BASE_FAMILY = "v4_official"
CANDIDATE_FAMILY = "pure_v42_memory"


def _official_path(directory: Path, task: str) -> Path:
    return directory / f"{SOURCE_FAMILY}_{task}_inner_seed0.npz"


def _save_archive(
    source: Path,
    destination: Path,
    family: str,
    task: str,
    prediction: np.ndarray | None = None,
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if prediction is not None:
        if prediction.shape != payload["prediction"].shape:
            raise ValueError(f"prediction shape mismatch for {task}")
        payload["prediction"] = prediction.astype(np.float32)
    metadata = json.loads(str(payload["metadata"].item()))
    metadata.update(
        {
            "family": family,
            "task": task,
            "phase": "inner",
            "source_family": SOURCE_FAMILY,
            "controlled_change": "da_only" if family == CANDIDATE_FAMILY else "none",
        }
    )
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", default="data/results/pure_v4_multiscale_cache_v1/ercot_2023.npz"
    )
    parser.add_argument(
        "--selected-summary",
        default="data/results/austria_pure_v42_hour_memory_residual_v1/summary.json",
    )
    parser.add_argument(
        "--official-dir", default="data/results/global_implicit_qp_profit_predictions_v1"
    )
    parser.add_argument(
        "--output-dir", default="data/results/pure_v42_memory_us_ercot_predictions_v1"
    )
    args = parser.parse_args()

    cache = _load(ROOT / args.cache)
    config = json.loads(
        (ROOT / args.selected_summary).read_text(encoding="utf-8")
    )["source_selected_config"]
    official_dir = ROOT / args.official_dir
    with np.load(_official_path(official_dir, "da"), allow_pickle=False) as archive:
        official_nodes = archive["nodes"].astype(str)
        official_dates = archive["dates"].astype(str)
        official_da = archive["prediction"].astype(np.float64)
        target_da = archive["target_da"].astype(np.float64)
    if not np.array_equal(cache["nodes"].astype(str), official_nodes):
        raise ValueError("cache and official v4 nodes differ")
    if not np.array_equal(cache["dates"].astype(str), official_dates):
        raise ValueError("cache and official v4 dates differ")
    if not np.allclose(cache["target"], target_da, rtol=0.0, atol=1e-6):
        raise ValueError("cache and official v4 DA targets differ")

    candidate_da = official_da.copy()
    node_rows = []
    for node, name in enumerate(official_nodes):
        arrays = _node_arrays(cache, node)
        strengthened_proxy, _, confidence = _memory_predict(**arrays, **config)
        std = np.maximum(cache["price_std"][node].astype(np.float64), 1.0)[:, None]
        correction = (strengthened_proxy - arrays["anchor"]) * std
        candidate_da[node] += correction
        base_mae = float(np.mean(np.abs(official_da[node] - target_da[node])))
        candidate_mae = float(np.mean(np.abs(candidate_da[node] - target_da[node])))
        node_rows.append(
            {
                "node": name,
                "base_mae_usd_per_mwh": base_mae,
                "candidate_mae_usd_per_mwh": candidate_mae,
                "relative_mae_change_pct": float(
                    100.0 * (candidate_mae / base_mae - 1.0)
                ),
                "mean_absolute_correction_usd_per_mwh": float(
                    np.mean(np.abs(correction))
                ),
                "mean_retrieval_confidence": float(confidence.mean()),
            }
        )

    output = ROOT / args.output_dir
    for family in (BASE_FAMILY, CANDIDATE_FAMILY):
        for task in TASKS:
            prediction = candidate_da if (family == CANDIDATE_FAMILY and task == "da") else None
            _save_archive(
                _official_path(official_dir, task),
                output / f"{family}_{task}_inner_seed0.npz",
                family,
                task,
                prediction,
            )

    base_mae = float(np.mean(np.abs(official_da - target_da)))
    candidate_mae = float(np.mean(np.abs(candidate_da - target_da)))
    report = {
        "experiment": "pure_v42_memory_us_ercot_predictions_v1",
        "status": "frozen_memory_correction_on_official_v4_da",
        "year": 2023,
        "selected_config": config,
        "rt_at_da_and_rt_arrays_identical_to_official_v4": True,
        "base_mae_usd_per_mwh": base_mae,
        "candidate_mae_usd_per_mwh": candidate_mae,
        "relative_mae_change_pct": float(100.0 * (candidate_mae / base_mae - 1.0)),
        "improved_nodes": int(
            sum(row["candidate_mae_usd_per_mwh"] < row["base_mae_usd_per_mwh"] for row in node_rows)
        ),
        "node_count": len(node_rows),
        "nodes": node_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
