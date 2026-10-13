#!/usr/bin/env python
"""合并多组联合训练 checkpoint manifest，供统一预测导出。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = {"da", "rt_at_da", "rt"}


def combine(root: Path, family_dirs: list[Path]) -> dict:
    families = {}
    for directory in family_dirs:
        report = json.loads(
            (directory / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        for family, task_paths in report["families"].items():
            if family in families:
                raise ValueError(f"重复family: {family}")
            if set(task_paths) != TASKS:
                raise ValueError(f"{family}没有完整三个任务: {sorted(task_paths)}")
            for path in task_paths.values():
                if not (root / path).is_file():
                    raise FileNotFoundError(root / path)
            families[family] = task_paths
    if not families:
        raise ValueError("至少需要一个family目录")
    return {
        "experiment": "global_joint_spread_grid_candidate_manifest_v1",
        "phase": "inner",
        "selection_rule": "full dual-settlement comparison on 2023 inner validation",
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-dirs", required=True, help="逗号分隔的family目录")
    parser.add_argument(
        "--output", default="data/results/global_joint_spread_grid_manifest_v1.json"
    )
    args = parser.parse_args()
    directories = [ROOT / value.strip() for value in args.family_dirs.split(",") if value.strip()]
    report = combine(ROOT, directories)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "joint_grid_manifest_saved",
        "path": output.relative_to(ROOT).as_posix(),
        "families": sorted(report["families"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
