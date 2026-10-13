#!/usr/bin/env python
"""对多个Global DA seed的已导出预测做等权平均。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", required=True, help="逗号分隔的DA npz路径")
    parser.add_argument("--family", default="globalensemble")
    parser.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    parser.add_argument("--phase", choices=("inner", "outer"), default="inner")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = [ROOT / value.strip() for value in args.archives.split(",") if value.strip()]
    if len(paths) < 2:
        raise ValueError("多seed平均至少需要2份预测档案")

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            records.append({key: archive[key] for key in archive.files})
    reference = records[0]
    for index, record in enumerate(records[1:], start=1):
        for key in ("nodes", "dates", "target_da", "target_rt"):
            if key not in reference or key not in record or not np.array_equal(
                reference[key], record[key]
            ):
                raise ValueError(f"多seed档案字段不一致: index={index}, key={key}")
        if record["prediction"].shape != reference["prediction"].shape:
            raise ValueError(f"多seed预测形状不一致: index={index}")

    stacked = np.stack([
        record["prediction"].astype(np.float64) for record in records
    ])
    prediction = (
        np.mean(stacked, axis=0)
        if args.aggregation == "mean"
        else np.median(stacked, axis=0)
    ).astype(np.float32)
    metadata_items = [json.loads(str(record["metadata"].item())) for record in records]
    payload = dict(reference)
    payload["prediction"] = prediction
    payload["metadata"] = json.dumps({
        "archive_version": 2,
        "family": args.family,
        "task": "da",
        "phase": args.phase,
        "model": f"global_da_seed_{args.aggregation}_ensemble",
        "aggregation": args.aggregation,
        "members": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "family": metadata.get("family"),
                "seed": metadata.get("seed"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            }
            for path, metadata in zip(paths, metadata_items)
        ],
        "selection_stage": (
            "2023_inner_validation_candidate_not_final"
            if args.phase == "inner"
            else "2024_outer_validation_frozen_candidate"
        ),
    }, ensure_ascii=False, sort_keys=True)
    output = ROOT / args.output_dir / f"{args.family}_da_{args.phase}_seed0.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(json.dumps({
        "event": "global_da_seed_ensemble_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "members": len(records),
        "shape": list(prediction.shape),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
