#!/usr/bin/env python
"""生成“稳健XGBoost DA + Global DA”的固定权重融合候选。

这个脚本只融合已有预测，不读取收益、不修改原始数据。权重候选随后
必须通过同一个2023内部验证收益合同选择。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _save(path: Path, arrays: dict, metadata: dict) -> None:
    payload = dict(arrays)
    payload["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _family(weight: float) -> str:
    exact_percent = 100 * weight
    if not np.isclose(exact_percent, round(exact_percent), rtol=0.0, atol=1e-9):
        basis_points = int(round(10_000 * abs(weight)))
        sign = "m" if weight < 0 else "p"
        return f"blendg{sign}bp{basis_points:04d}"
    percent = int(round(100 * weight))
    return (
        f"blendg{percent:02d}" if percent > 0
        else f"blendgm{abs(percent):02d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-family", default="daopt2")
    parser.add_argument(
        "--source-dir", default="data/results/global_da_robust_predictions"
    )
    parser.add_argument(
        "--template-dir",
        default=None,
        help="RT-at-DA/RT schema模板目录；默认与source-dir相同",
    )
    parser.add_argument(
        "--template-family",
        default=None,
        help="RT-at-DA/RT schema模板family；默认与source-family相同",
    )
    parser.add_argument(
        "--weights", default="0.02,0.05,0.10,0.15,0.20,0.30,0.40"
    )
    parser.add_argument(
        "--baseline-dir", default="data/results/global_baselines_v1/da_inner"
    )
    parser.add_argument("--phase", choices=("inner", "outer"), default="inner")
    parser.add_argument(
        "--output-dir", default="data/results/global_da_blend_predictions"
    )
    parser.add_argument(
        "--output-manifest", default="data/results/global_da_blend_manifest.json"
    )
    args = parser.parse_args()

    weights = tuple(float(value) for value in args.weights.split(",") if value.strip())
    if not weights or any(value == 0.0 or not -1.0 < value < 1.0 for value in weights):
        raise ValueError("weights必须是(-1,1)内且非0的逗号分隔数值")
    if len(set(weights)) != len(weights):
        raise ValueError("weights不能重复")
    families = [_family(value) for value in weights]
    if len(set(families)) != len(families):
        raise ValueError("权重四舍五入后的family名称重复")

    source_dir = ROOT / args.source_dir
    template_dir = ROOT / (args.template_dir or args.source_dir)
    template_family = args.template_family or args.source_family
    output_dir = ROOT / args.output_dir
    source_da = _load(
        source_dir / f"{args.source_family}_da_{args.phase}_seed0.npz"
    )
    source_metadata = json.loads(str(source_da["metadata"].item()))
    nodes = source_da["nodes"].astype(str)
    neural_prediction = source_da["prediction"].astype(np.float32)
    target = source_da["target_da"].astype(np.float32)
    baseline_predictions = []
    for index, node in enumerate(nodes):
        path = (
            ROOT / args.baseline_dir / f"xgboost_absolute_{node}_seed0.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["prediction"].astype(np.float32)
            actual = archive["target"].astype(np.float32)
        if actual.shape != target[index].shape or not np.allclose(
            actual, target[index], rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"XGBoost与统一档案真值不一致: {node}")
        baseline_predictions.append(prediction)
    baseline_prediction = np.stack(baseline_predictions)
    if baseline_prediction.shape != neural_prediction.shape:
        raise ValueError("XGBoost与Global DA预测形状不一致")

    manifest = []
    for family, neural_weight in zip(families, weights):
        arrays = dict(source_da)
        arrays["prediction"] = (
            (1.0 - neural_weight) * baseline_prediction
            + neural_weight * neural_prediction
        ).astype(np.float32)
        metadata = dict(source_metadata)
        metadata.update({
            "family": family,
            "task": "da",
            "model": "fixed_weight_xgboost_global_ensemble",
            "xgboost_weight": 1.0 - neural_weight,
            "global_weight": neural_weight,
            "global_source_family": args.source_family,
            "phase": args.phase,
            "selection_stage": (
                "2023_inner_validation_candidate"
                if args.phase == "inner"
                else "2024_outer_validation_frozen_candidate"
            ),
        })
        _save(output_dir / f"{family}_da_{args.phase}_seed0.npz", arrays, metadata)

        # 评价器会加载family的三个任务。这两份只是schema模板；
        # 请求的三元组仍显式使用稳健XGBoost RT-at-DA和RT。
        for task in ("rt_at_da", "rt"):
            template = _load(
                template_dir / f"{template_family}_{task}_{args.phase}_seed0.npz"
            )
            template_metadata = json.loads(str(template["metadata"].item()))
            template_metadata.update({
                "family": family,
                "task": task,
                "phase": args.phase,
                "template_only": True,
            })
            _save(
                output_dir / f"{family}_{task}_{args.phase}_seed0.npz",
                template,
                template_metadata,
            )
        manifest.append({
            "family": family,
            "xgboost_weight": 1.0 - neural_weight,
            "global_weight": neural_weight,
            "global_source_family": args.source_family,
        })

    output = ROOT / args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": "global_da_fixed_weight_blend_screen",
        "status": (
            "2023_inner_validation_candidate_not_final"
            if args.phase == "inner"
            else "2024_outer_validation_frozen_candidate"
        ),
        "phase": args.phase,
        "selection_target": "full continuous MPC net revenue",
        "families": manifest,
        "prediction_dir": args.output_dir,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "global_da_blends_saved",
        "families": families,
        "output": output.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
