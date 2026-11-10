#!/usr/bin/env python
"""Audit the frozen MemCast-style pure-v4 correction on U.S. markets.

This is an evaluation-only script.  It reuses the configuration selected
before the Austria test and never tunes on the market being reported.  The
cached source artifact contains the three frozen v4 seed predictions, known
fundamentals, calendar variables, and targets for every node.
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

from scripts.decision_aware.research_pure_v42_memory_residual import (  # noqa: E402
    _load,
    _memory_predict,
)


MARKETS = ("caiso", "ercot", "nyiso", "pjm")


def _node_arrays(payload: dict[str, np.ndarray], node: int) -> dict[str, np.ndarray]:
    mean = payload["price_mean"][node].astype(np.float64)[:, None]
    std = np.maximum(payload["price_std"][node].astype(np.float64)[:, None], 1.0)
    experts = (
        payload["experts"][node].astype(np.float64) - mean[:, :, None]
    ) / std[:, :, None]
    # This is the source-cache representation used to select the strengthened
    # v4 rule: 2/3 seed0 plus 1/3 robust three-seed centre (median proxy).
    anchor = (2.0 / 3.0) * experts[:, :, 0] + (1.0 / 3.0) * np.median(
        experts[:, :, :3], axis=2
    )
    truth = (payload["target"][node].astype(np.float64) - mean) / std
    return {
        "anchor": anchor,
        "truth": truth,
        "fundamentals": payload["fundamental_target"][node].astype(np.float64),
        "calendar": payload["calendar"][node, :, 0, 2:].astype(np.float64),
    }


def _metrics(
    arrays: dict[str, np.ndarray], prediction: np.ndarray, std: np.ndarray, start: int
) -> dict[str, float]:
    target = arrays["truth"][start:]
    anchor = arrays["anchor"][start:]
    candidate = prediction[start:]
    scale = std[start:, None]
    base_abs = np.abs(anchor - target) * scale
    candidate_abs = np.abs(candidate - target) * scale
    return {
        "samples": int(base_abs.size),
        "base_mae_usd_per_mwh": float(base_abs.mean()),
        "candidate_mae_usd_per_mwh": float(candidate_abs.mean()),
        "mae_change_usd_per_mwh": float(candidate_abs.mean() - base_abs.mean()),
        "relative_mae_change_pct": float(
            100.0 * (candidate_abs.mean() / base_abs.mean() - 1.0)
        ),
        "base_normalized_mae": float(np.abs(anchor - target).mean()),
        "candidate_normalized_mae": float(np.abs(candidate - target).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", default="data/results/pure_v4_multiscale_cache_v1"
    )
    parser.add_argument(
        "--selected-summary",
        default="data/results/austria_pure_v42_hour_memory_residual_v1/summary.json",
    )
    parser.add_argument(
        "--output",
        default="data/results/pure_v42_memory_us_markets_v1/summary.json",
    )
    args = parser.parse_args()

    selected_payload = json.loads(
        (ROOT / args.selected_summary).read_text(encoding="utf-8")
    )
    config = selected_payload["source_selected_config"]
    report: dict[str, object] = {
        "experiment": "pure_v42_memory_us_markets_v1",
        "status": "frozen_source_side_audit_no_target_market_tuning",
        "year": 2023,
        "anchor": "two_thirds_seed0_plus_one_third_three_seed_median_proxy",
        "selected_config": config,
        "markets": {},
    }
    pooled_base: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []

    for market in MARKETS:
        payload = _load(ROOT / args.cache_dir / f"{market}_2023.npz")
        rows = []
        for node, name in enumerate(payload["nodes"].astype(str)):
            arrays = _node_arrays(payload, node)
            prediction, _, confidence = _memory_predict(**arrays, **config)
            std = np.maximum(payload["price_std"][node].astype(np.float64), 1.0)
            full = _metrics(arrays, prediction, std, 0)
            last90 = _metrics(arrays, prediction, std, max(0, len(std) - 90))
            rows.append(
                {
                    "node": name,
                    "full_year": full,
                    "last_90_days": last90,
                    "mean_retrieval_confidence": float(confidence.mean()),
                }
            )
            pooled_base.append(
                np.abs(arrays["anchor"] - arrays["truth"]) * std[:, None]
            )
            pooled_candidate.append(
                np.abs(prediction - arrays["truth"]) * std[:, None]
            )

        base = np.asarray([row["full_year"]["base_mae_usd_per_mwh"] for row in rows])
        candidate = np.asarray(
            [row["full_year"]["candidate_mae_usd_per_mwh"] for row in rows]
        )
        report["markets"][market] = {
            "node_count": len(rows),
            "macro_base_mae_usd_per_mwh": float(base.mean()),
            "macro_candidate_mae_usd_per_mwh": float(candidate.mean()),
            "macro_relative_mae_change_pct": float(
                100.0 * (candidate.mean() / base.mean() - 1.0)
            ),
            "improved_nodes": int(np.sum(candidate < base)),
            "rows": rows,
        }

    pooled_base_values = np.concatenate([item.reshape(-1) for item in pooled_base])
    pooled_candidate_values = np.concatenate(
        [item.reshape(-1) for item in pooled_candidate]
    )
    report["pooled_micro"] = {
        "samples": int(pooled_base_values.size),
        "base_mae_usd_per_mwh": float(pooled_base_values.mean()),
        "candidate_mae_usd_per_mwh": float(pooled_candidate_values.mean()),
        "relative_mae_change_pct": float(
            100.0
            * (pooled_candidate_values.mean() / pooled_base_values.mean() - 1.0)
        ),
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
