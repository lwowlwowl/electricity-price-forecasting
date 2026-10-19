#!/usr/bin/env python
"""Replace one forecast leg in an already aligned three-task archive family."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--base-family", required=True)
    parser.add_argument("--new-family", required=True)
    parser.add_argument("--phase", choices=("inner", "outer", "final"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_dir = ROOT / args.base_dir
    output_dir = ROOT / args.output_dir
    suffix = f"{args.phase}_seed{args.seed}.npz"
    base = {
        task: _load(base_dir / f"{args.base_family}_{task}_{suffix}")
        for task in TASKS
    }
    replacement = _load(ROOT / args.replacement)
    reference = base[args.task]
    for key in ("nodes", "dates"):
        if key not in replacement or not np.array_equal(replacement[key], reference[key]):
            raise ValueError(f"替换预测的{key}与基础档案不一致")
    if replacement["prediction"].shape != reference["prediction"].shape:
        raise ValueError("替换预测形状与基础档案不一致")
    target_key = "target_da" if args.task == "da" else "target_rt"
    replacement_target_key = "target" if "target" in replacement else target_key
    if replacement_target_key in replacement and target_key in reference:
        if not np.allclose(
            replacement[replacement_target_key], reference[target_key], rtol=0.0, atol=1e-5
        ):
            raise ValueError("替换预测与基础档案的真值不一致")

    output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        values = dict(base[task])
        if task == args.task:
            values["prediction"] = replacement["prediction"].astype(
                reference["prediction"].dtype
            )
        metadata = {
            "family": args.new_family,
            "base_family": args.base_family,
            "replaced_task": args.task,
            "task": task,
            "phase": args.phase,
            "seed": args.seed,
            "replacement": args.replacement if task == args.task else None,
        }
        values["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(output_dir / f"{args.new_family}_{task}_{suffix}", **values)

    print(json.dumps({
        "event": "prediction_leg_variant_ready",
        "family": args.new_family,
        "replaced_task": args.task,
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
