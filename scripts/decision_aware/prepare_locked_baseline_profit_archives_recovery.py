#!/usr/bin/env python
"""Prepare locked-2025 baseline profit archives with strict trailing-date alignment.

The frozen rolling-RT contract needs four future hours and therefore ends one
delivery day before the 24-hour DA tasks at the read boundary.  This recovery
entrypoint permits only that single trailing-day difference and evaluates every
task on the exact common prefix.  It never fabricates a prediction or target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")
DA_TRUTH_ATOL = 1e-4
RT_TRUTH_ATOL = 2e-4


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=DA_TRUTH_ATOL
    ):
        raise ValueError(f"{label}不一致")


def canonicalize_rolling_first_step_truth(
    rt_at_da_truth: np.ndarray,
    rolling_truth_windows: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Unify duplicate RT truth only within a strict float32 inverse-transform tolerance.

    The locked evaluator deliberately requires bitwise-equivalent truth after its
    float32 conversion.  Some baseline archives reach the same ERCOT truth via
    independent float32 inverse-transform/serialization paths and can therefore
    differ by tiny roundoff residuals.  This function rejects any discrepancy
    above 0.0002 USD/MWh before copying the RT-at-DA truth into the rolling
    first-step truth slot.
    """
    left = np.asarray(rt_at_da_truth, dtype=np.float32)
    windows = np.asarray(rolling_truth_windows)
    if windows.ndim != left.ndim + 1 or windows.shape[:-1] != left.shape:
        raise ValueError("滚动RT窗口与RT-at-DA真值形状不一致")
    right = np.asarray(windows[..., 0], dtype=np.float32)
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    if np.any(difference > RT_TRUTH_ATOL):
        index = np.unravel_index(int(np.argmax(difference)), difference.shape)
        raise ValueError(
            "RT-at-DA与滚动RT第一步真值差异超过float32恢复容差: "
            f"index={index}, left={left[index]}, right={right[index]}, "
            f"diff={difference[index]}, atol={RT_TRUTH_ATOL}"
        )
    canonical = windows.copy()
    canonical[..., 0] = np.asarray(rt_at_da_truth, dtype=canonical.dtype)
    stats: dict[str, float | int] = {
        "rt_truth_canonicalized_values": int(np.count_nonzero(difference)),
        "rt_truth_max_abs_difference": float(np.max(difference, initial=0.0)),
        "rt_truth_atol": RT_TRUTH_ATOL,
    }
    return canonical, stats


def align_trailing_rt_boundary(
    source: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    da_dates = source["da"]["dates"].astype(str)
    rt_at_da_dates = source["rt_at_da"]["dates"].astype(str)
    rt_dates = source["rt"]["dates"].astype(str)
    if not np.array_equal(da_dates, rt_at_da_dates):
        raise ValueError("RT-at-DA日期与DA不一致")
    if len(da_dates) - len(rt_dates) not in {0, 1}:
        raise ValueError("滚动RT与DA只能相差零个或一个尾端交付日")
    if not np.array_equal(da_dates[: len(rt_dates)], rt_dates):
        raise ValueError("滚动RT日期必须是DA日期的完整前缀")

    dropped_dates = da_dates[len(rt_dates) :].tolist()
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for task, item in source.items():
        day_count = len(rt_dates)
        current = dict(item)
        current["dates"] = item["dates"][:day_count]
        for key in ("prediction", "target", "target_da"):
            if key in item:
                current[key] = item[key][:, :day_count]
        if "executed_timestamps" in item:
            current["executed_timestamps"] = item["executed_timestamps"][:day_count]
        aligned[task] = current
    return aligned, dropped_dates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--source-da", required=True)
    parser.add_argument("--source-rt-at-da", required=True)
    parser.add_argument("--source-rt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perfect-oracle", action="store_true")
    args = parser.parse_args()

    paths = {
        "da": ROOT / args.source_da,
        "rt_at_da": ROOT / args.source_rt_at_da,
        "rt": ROOT / args.source_rt,
    }
    source = {task: load(path) for task, path in paths.items()}
    for task in TASKS:
        if not np.array_equal(source["da"]["nodes"], source[task]["nodes"]):
            raise ValueError(f"{task}节点顺序与DA不一致")
    source, dropped_dates = align_trailing_rt_boundary(source)
    if "target_da" not in source["rt_at_da"]:
        raise ValueError("RT-at-DA档案缺少同日起报时可核验的DA真值")
    da_truth_max_abs_difference = float(
        np.max(np.abs(source["da"]["target"] - source["rt_at_da"]["target_da"]))
    )
    same(source["da"]["target"], source["rt_at_da"]["target_da"], "DA真值")
    if "executed_timestamps" not in source["rt"] or "timezone" not in source["rt"]:
        raise ValueError("滚动RT档案缺少执行时间戳或时区")
    source["rt"]["target"], rt_truth_stats = canonicalize_rolling_first_step_truth(
        source["rt_at_da"]["target"], source["rt"]["target"]
    )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": "allmarket_locked_2025_baseline_profit_archive_v1_recovery1",
        "family": args.family,
        "phase": "final",
        "selection_allowed": False,
        "date_alignment": "strict_common_prefix_due_to_rolling_rt_horizon",
        "dropped_trailing_dates": dropped_dates,
        "da_truth_atol": DA_TRUTH_ATOL,
        "da_truth_max_abs_difference": da_truth_max_abs_difference,
        "rt_truth_canonicalization": "rt_at_da_truth_copied_to_rolling_first_step_after_float32_roundoff_check",
        **rt_truth_stats,
    }
    for task in TASKS:
        item = source[task]
        arrays: dict[str, np.ndarray] = {
            "nodes": item["nodes"],
            "dates": item["dates"],
            "prediction": item["target"] if args.perfect_oracle else item["prediction"],
            "target_rt": source["rt_at_da"]["target"] if task != "rt" else item["target"],
            "metadata": np.asarray(
                json.dumps({**common, "task": task}, ensure_ascii=False, sort_keys=True)
            ),
        }
        if task == "da":
            arrays["target_da"] = item["target"]
        if task == "rt":
            arrays["executed_timestamps"] = item["executed_timestamps"]
            arrays["timezone"] = item["timezone"]
        np.savez_compressed(output_dir / f"{args.family}_{task}_final_seed0.npz", **arrays)
    print(
        json.dumps(
            {
                "event": "baseline_profit_archives_ready",
                "family": args.family,
                "perfect_oracle": args.perfect_oracle,
                "common_days": len(source["rt"]["dates"]),
                "dropped_trailing_dates": dropped_dates,
            }
        )
    )


if __name__ == "__main__":
    main()
