#!/usr/bin/env python
"""合并任务专属损失与旧稳定Global候选，供小规模任务替换诊断。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def prepare(taskloss: dict, stable: dict) -> dict:
    for report, required in ((taskloss, ("taskloss0", "taskloss1")), (stable, ("c0",))):
        families = report.get("families", {})
        for family in required:
            if set(families.get(family, {})) != set(TASKS):
                raise ValueError(f"候选清单缺少完整{family}三任务")
    return {
        "experiment": "global_ercot_taskloss_small_mixed_diagnosis_v1",
        "phase": "inner",
        "2024_used": False,
        "2025_used": False,
        "selection_scope": (
            "three full families only: two task-specific-loss ranks and the "
            "single best pre-existing stable Global family; 3^3=27 triplets"
        ),
        "families": {
            "new0": dict(taskloss["families"]["taskloss0"]),
            "new1": dict(taskloss["families"]["taskloss1"]),
            "stable0": dict(stable["families"]["c0"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taskloss",
        default="data/results/global_taskloss_candidate_manifest_v1.json",
    )
    parser.add_argument(
        "--stable",
        default="data/results/global_f0_f1_stable_v2_candidate_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="data/results/global_taskloss_mixed_manifest_v1.json",
    )
    args = parser.parse_args()
    taskloss = json.loads((ROOT / args.taskloss).read_text(encoding="utf-8"))
    stable = json.loads((ROOT / args.stable).read_text(encoding="utf-8"))
    report = prepare(taskloss, stable)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": output.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
