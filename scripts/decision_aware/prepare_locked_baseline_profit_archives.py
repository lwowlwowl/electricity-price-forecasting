#!/usr/bin/env python
"""把一个ERCOT三任务锁定预测族转换为统一LP/MPC结算档案。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label}不一致")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--source-da", required=True)
    parser.add_argument("--source-rt-at-da", required=True)
    parser.add_argument("--source-rt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--perfect-oracle",
        action="store_true",
        help="将每个任务预测替换为其真实价格，仅用于同电池合同的受限先知上界",
    )
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
        if not np.array_equal(source["da"]["dates"], source[task]["dates"]):
            raise ValueError(f"{task}日期与DA不一致")
    if "target_da" not in source["rt_at_da"]:
        raise ValueError("RT-at-DA档案缺少同日起报时可核验的DA真值")
    same(source["da"]["target"], source["rt_at_da"]["target_da"], "DA真值")
    if "executed_timestamps" not in source["rt"] or "timezone" not in source["rt"]:
        raise ValueError("滚动RT档案缺少执行时间戳或时区")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": "allmarket_locked_2025_baseline_profit_archive_v1",
        "family": args.family,
        "phase": "final",
        "selection_allowed": False,
    }
    for task in TASKS:
        item = source[task]
        arrays: dict[str, np.ndarray] = {
            "nodes": item["nodes"],
            "dates": item["dates"],
            "prediction": item["target"] if args.perfect_oracle else item["prediction"],
            "target_rt": source["rt_at_da"]["target"] if task != "rt" else item["target"],
            "metadata": np.asarray(json.dumps({**common, "task": task}, ensure_ascii=False, sort_keys=True)),
        }
        if task == "da":
            arrays["target_da"] = item["target"]
        if task == "rt":
            arrays["executed_timestamps"] = item["executed_timestamps"]
            arrays["timezone"] = item["timezone"]
        np.savez_compressed(output_dir / f"{args.family}_{task}_final_seed0.npz", **arrays)
    print(json.dumps({
        "event": "baseline_profit_archives_ready",
        "family": args.family,
        "perfect_oracle": args.perfect_oracle,
    }))


if __name__ == "__main__":
    main()
