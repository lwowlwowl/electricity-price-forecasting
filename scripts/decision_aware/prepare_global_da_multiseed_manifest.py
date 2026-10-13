#!/usr/bin/env python
"""为已完成的Global DA seed1/2生成可审计预测导出清单。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="data/checkpoints/global_ercot_huber_f1_stable_v2",
    )
    parser.add_argument(
        "--output", default="data/results/global_da_multiseed_candidate_manifest.json"
    )
    parser.add_argument("--phase", choices=("inner", "outer"), default="inner")
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()
    families = {}
    selected = []
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds必须非空且不能重复")
    for seed in seeds:
        run_dir = ROOT / args.checkpoint_root / f"da_{args.phase}_seed{seed}"
        summary_path = run_dir / "summary.json"
        best_path = run_dir / "best.pt"
        if not summary_path.is_file() or not best_path.is_file():
            raise FileNotFoundError(f"seed{seed}训练未完成: {run_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        family = f"globalseed{seed}"
        relative = best_path.relative_to(ROOT).as_posix()
        families[family] = {"da": relative}
        # 严格外层目录使用独立输出名；同时保留旧family名，兼容已启动的云脚本。
        if "outer_frozen" in args.checkpoint_root:
            families[f"globalfrozenseed{seed}"] = {"da": relative}
        selected.append({
            "family": family,
            "seed": seed,
            "checkpoint": relative,
            "best_epoch": int(summary["best_epoch"]),
            "best_macro_mae": float(summary["best_macro_mae"]),
        })
    report = {
        "experiment": "global_da_f1_multiseed_export",
        "status": (
            "2023_inner_validation_candidate_not_final"
            if args.phase == "inner"
            else "2024_outer_validation_frozen_candidate"
        ),
        "phase": args.phase,
        "families": families,
        "selected": selected,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "global_da_multiseed_manifest_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "selected": selected,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
