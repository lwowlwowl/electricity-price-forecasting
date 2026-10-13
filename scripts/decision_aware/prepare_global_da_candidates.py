#!/usr/bin/env python
"""汇总多组DA训练候选，并为完整三元组评价准备统一预测目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clone_archive(source: Path, destination: Path, family: str, task: str) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["family"] = family
    metadata["task"] = task
    metadata["cloned_template_source"] = source.relative_to(ROOT).as_posix()
    arrays["metadata"] = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", required=True, help="逗号分隔的checkpoint run目录")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--output-manifest",
        default="data/results/global_da_robust_candidate_manifest.json",
    )
    parser.add_argument(
        "--prediction-dir",
        default="data/results/global_da_robust_predictions",
    )
    parser.add_argument(
        "--template-dir",
        default="data/results/global_f0_f1_stable_v2_predictions",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k必须至少为1")

    candidates = []
    for raw in args.run_dirs.split(","):
        run_dir = ROOT / raw.strip()
        manifest = json.loads(
            (run_dir / "candidate_manifest.json").read_text(encoding="utf-8")
        )
        for item in manifest:
            checkpoint = run_dir / item["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            candidates.append({
                "macro_mae": float(item["macro_mae"]),
                "epoch": int(item["epoch"]),
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": _sha256(checkpoint),
                "source_run": run_dir.relative_to(ROOT).as_posix(),
            })
    unique = {}
    for item in candidates:
        # 三组实验都从同一个冻结checkpoint开始；它们的epoch 0模型权重相同，
        # 但optimizer/元数据不同会导致整个.pt哈希不同。只保留一个epoch 0，
        # 避免重复基线挤占真正训练候选名额。
        key = "shared_epoch0" if item["epoch"] == 0 else item["checkpoint_sha256"]
        unique.setdefault(key, item)
    selected = sorted(
        unique.values(), key=lambda item: (item["macro_mae"], item["source_run"], item["epoch"])
    )[:args.top_k]

    families = {}
    prediction_dir = ROOT / args.prediction_dir
    template_dir = ROOT / args.template_dir
    for index, item in enumerate(selected):
        family = f"daopt{index}"
        families[family] = {"da": item["checkpoint"]}
        # evaluator会统一加载family的三个任务；实际候选组合只使用该family的DA。
        for task, template_family in (("rt_at_da", "c1"), ("rt", "c0")):
            _clone_archive(
                template_dir / f"{template_family}_{task}_inner_seed0.npz",
                prediction_dir / f"{family}_{task}_inner_seed0.npz",
                family,
                task,
            )
    report = {
        "experiment": "global_da_robust_candidate_pool",
        "selection": "top unique candidates by 2023 inner-validation node-macro MAE",
        "top_k": args.top_k,
        "selected": selected,
        "families": families,
        "prediction_dir": args.prediction_dir,
        "template_only_tasks": {
            "rt_at_da": "c1; only satisfies archive schema and is not used in requested triplets",
            "rt": "c0; only satisfies archive schema and is not used in requested triplets",
        },
    }
    output = ROOT / args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "candidate_manifest_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "selected": selected,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
