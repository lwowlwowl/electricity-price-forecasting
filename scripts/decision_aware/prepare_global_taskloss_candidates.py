#!/usr/bin/env python
"""冻结无锚点任务专属损失的候选，并生成完整三元组清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")
DEFAULT_RUN_DIRS = {
    "da": "data/checkpoints/global_ercot_taskloss_da_mae_v1/da_inner_seed0",
    "rt_at_da": (
        "data/checkpoints/global_ercot_taskloss_rt_at_da_pseudohuber_v1/"
        "rt_at_da_inner_seed0"
    ),
    "rt": "data/checkpoints/global_ercot_taskloss_rt_mae_v1/rt_inner_seed0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    *, root: Path = ROOT, run_dirs: dict[str, str] | None = None, keep: int = 2,
    family_prefix: str = "taskloss",
) -> dict:
    if keep < 1:
        raise ValueError("keep必须至少为1")
    configured = DEFAULT_RUN_DIRS if run_dirs is None else run_dirs
    if set(configured) != set(TASKS):
        raise ValueError(f"run_dirs必须且只能包含{TASKS}")
    if not family_prefix or "_" in family_prefix:
        raise ValueError("family_prefix必须是非空且不含下划线的短标识")

    selected_by_task: dict[str, list[dict]] = {}
    for task in TASKS:
        run_dir = root / configured[task]
        manifest_path = run_dir / "candidate_manifest.json"
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        ranked = sorted(
            entries,
            key=lambda item: (
                float(item.get("selection_score", item["macro_mae"])),
                int(item["epoch"]),
            ),
        )[:keep]
        if len(ranked) < keep:
            raise ValueError(f"{task}候选不足{keep}个")
        selected = []
        for entry in ranked:
            checkpoint = run_dir / entry["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            selected.append({
                "task": task,
                "epoch": int(entry["epoch"]),
                "macro_mae": float(entry["macro_mae"]),
                "selection_score": float(
                    entry.get("selection_score", entry["macro_mae"])
                ),
                "checkpoint": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": _sha256(checkpoint),
            })
        selected_by_task[task] = selected

    families = {
        f"{family_prefix}{rank}": {
            task: selected_by_task[task][rank]["checkpoint"] for task in TASKS
        }
        for rank in range(keep)
    }
    return {
        "experiment": f"global_ercot_{family_prefix}_v1_candidates",
        "phase": "inner",
        "seed": 0,
        "2024_used": False,
        "2025_used": False,
        "selection_rule": (
            "train each task from scratch without lag168; retain top-K epochs by "
            "2023 node-equal macro MAE; select only among the resulting Cartesian "
            "triplets using the frozen full settlement"
        ),
        "losses": {"da": "mae", "rt_at_da": "pseudo_huber", "rt": "mae"},
        "selected_by_task": selected_by_task,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--family-prefix", default="taskloss")
    parser.add_argument("--da-run-dir", default=DEFAULT_RUN_DIRS["da"])
    parser.add_argument(
        "--rt-at-da-run-dir", default=DEFAULT_RUN_DIRS["rt_at_da"]
    )
    parser.add_argument("--rt-run-dir", default=DEFAULT_RUN_DIRS["rt"])
    parser.add_argument(
        "--output",
        default="data/results/global_taskloss_candidate_manifest_v1.json",
    )
    args = parser.parse_args()
    report = prepare(
        keep=args.keep,
        family_prefix=args.family_prefix,
        run_dirs={
            "da": args.da_run_dir,
            "rt_at_da": args.rt_at_da_run_dir,
            "rt": args.rt_run_dir,
        },
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
