#!/usr/bin/env python
"""Convert cutoff-safe historical baseline predictions to LP/MPC archives."""
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


def _same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label}不一致")


def prepare(
    family: str,
    phase: str,
    source_paths: dict[str, Path],
    output_dir: Path,
) -> None:
    if phase not in ("inner", "outer"):
        raise ValueError("phase必须为inner或outer")
    source = {task: _load(source_paths[task]) for task in TASKS}
    for task in TASKS:
        if not np.array_equal(source["da"]["nodes"], source[task]["nodes"]):
            raise ValueError(f"{task}节点顺序与DA不一致")
    if not np.array_equal(source["da"]["dates"], source["rt_at_da"]["dates"]):
        raise ValueError("DA与RT-at-DA日期不一致")
    if "target_da" not in source["rt_at_da"]:
        raise ValueError("RT-at-DA档案缺少target_da")
    _same(source["da"]["target"], source["rt_at_da"]["target_da"], "DA真值")
    if "executed_timestamps" not in source["rt"] or "timezone" not in source["rt"]:
        raise ValueError("滚动RT档案缺少时间戳或时区")

    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": "historical_baseline_profit_archive_v1",
        "family": family,
        "phase": phase,
        "selection_allowed": phase == "inner",
        "locked_2025_read": False,
    }
    for task in TASKS:
        item = source[task]
        arrays: dict[str, np.ndarray] = {
            "nodes": item["nodes"],
            "dates": item["dates"],
            "prediction": item["prediction"],
            "target_rt": (
                source["rt_at_da"]["target"]
                if task != "rt"
                else item["target"] if "target" in item else item["target_rt"]
            ),
            "metadata": np.asarray(
                json.dumps({**common, "task": task}, ensure_ascii=False, sort_keys=True)
            ),
        }
        if task == "da":
            arrays["target_da"] = item["target"]
        if task == "rt":
            arrays["executed_timestamps"] = item["executed_timestamps"]
            arrays["timezone"] = item["timezone"]
        np.savez_compressed(output_dir / f"{family}_{task}_{phase}_seed0.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--phase", required=True, choices=("inner", "outer"))
    parser.add_argument("--source-da", required=True)
    parser.add_argument("--source-rt-at-da", required=True)
    parser.add_argument("--source-rt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prepare(
        args.family,
        args.phase,
        {
            "da": ROOT / args.source_da,
            "rt_at_da": ROOT / args.source_rt_at_da,
            "rt": ROOT / args.source_rt,
        },
        ROOT / args.output_dir,
    )
    print(json.dumps({"event": "historical_profit_archives_ready", "phase": args.phase}))


if __name__ == "__main__":
    main()
