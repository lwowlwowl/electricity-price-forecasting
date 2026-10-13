#!/usr/bin/env python
"""从稳定F0/F1训练中为每个任务冻结总体macro-MAE前2个候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")
DEFAULT_ROOTS = {
    "f0": "data/checkpoints/global_ercot_huber_f0_stable_v2",
    "f1": "data/checkpoints/global_ercot_huber_f1_stable_v2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_candidates(roots: dict[str, str], keep: int = 2) -> dict:
    if keep < 1:
        raise ValueError("keep必须至少为1")
    selected_by_task = {}
    for task in TASKS:
        candidates = []
        for architecture, root_value in roots.items():
            run_dir = ROOT / root_value / f"{task}_inner_seed0"
            manifest_path = run_dir / "candidate_manifest.json"
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in entries:
                checkpoint = run_dir / entry["checkpoint"]
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                candidates.append({
                    "architecture": architecture,
                    "task": task,
                    "epoch": int(entry["epoch"]),
                    "macro_mae": float(entry["macro_mae"]),
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint),
                })
        candidates.sort(
            key=lambda item: (
                item["macro_mae"], item["architecture"], item["epoch"]
            )
        )
        selected_by_task[task] = candidates[:keep]
    families = {
        f"c{rank}": {
            task: selected_by_task[task][rank]["checkpoint"] for task in TASKS
        }
        for rank in range(keep)
    }
    return {
        "experiment": "global_f0_f1_stable_v2_forecast_candidate_filter",
        "phase": "inner",
        "seed": 0,
        "selection_rule": (
            "pool retained F0/F1 epoch candidates; keep overall top-2 per task "
            "by 2023 node-equal macro MAE; use full settlement only to select "
            "among the resulting 2x2x2 triplets"
        ),
        "selected_by_task": selected_by_task,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument(
        "--output",
        default="data/results/global_f0_f1_stable_v2_candidate_manifest.json",
    )
    args = parser.parse_args()
    report = select_candidates(DEFAULT_ROOTS, args.keep)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
