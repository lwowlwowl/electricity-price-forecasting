#!/usr/bin/env python
"""用三个Global DA seed的一致性门控XGBoost残差校正。

只处理已导出的2023内部验证预测，不读写原始市场数据。
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


def _same_contract(left: dict, right: dict, label: str) -> None:
    for key in ("nodes", "dates", "target_da", "target_rt"):
        if key not in left or key not in right or not np.array_equal(left[key], right[key]):
            raise ValueError(f"Global seed档案合同不一致: {label}/{key}")


def _direction_accuracy(spread: np.ndarray, realized: np.ndarray) -> float:
    valid = (spread != 0.0) & (realized != 0.0)
    return float(np.mean(np.sign(spread[valid]) == np.sign(realized[valid])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-archives",
        default=(
            "data/results/global_da_robust_predictions/daopt2_da_inner_seed0.npz,"
            "data/results/global_da_multiseed_predictions/globalseed1_da_inner_seed0.npz,"
            "data/results/global_da_multiseed_predictions/globalseed2_da_inner_seed0.npz"
        ),
    )
    parser.add_argument("--weight", type=float, default=0.20)
    parser.add_argument(
        "--baseline-dir", default="data/results/global_baselines_v1"
    )
    parser.add_argument(
        "--template-dir", default="data/results/global_da_robust_predictions"
    )
    parser.add_argument("--template-family", default="daopt2")
    parser.add_argument(
        "--output-dir", default="data/results/global_da_consensus_predictions"
    )
    parser.add_argument(
        "--output-manifest", default="data/results/global_da_consensus_manifest.json"
    )
    args = parser.parse_args()
    if not 0.0 < args.weight < 1.0:
        raise ValueError("weight必须位于(0,1)")

    paths = [ROOT / value.strip() for value in args.seed_archives.split(",") if value.strip()]
    if len(paths) != 3:
        raise ValueError("当前一致性门控要求恰好3个Global seed")
    seeds = [_load(path) for path in paths]
    for index, record in enumerate(seeds[1:], start=1):
        _same_contract(seeds[0], record, f"seed{index}")
    nodes = seeds[0]["nodes"].astype(str)
    global_prediction = np.stack([
        record["prediction"].astype(np.float32) for record in seeds
    ])
    target_da = seeds[0]["target_da"].astype(np.float32)
    target_rt = seeds[0]["target_rt"].astype(np.float32)

    xgb_da = []
    xgb_rt_at_da = []
    for node_index, node in enumerate(nodes):
        da_path = (
            ROOT / args.baseline_dir / "da_inner"
            / f"xgboost_absolute_{node}_seed0.npz"
        )
        rt_path = (
            ROOT / args.baseline_dir / "rt_at_da_inner"
            / f"xgboost_pseudohuber_{node}_seed0.npz"
        )
        with np.load(da_path, allow_pickle=False) as archive:
            da_prediction = archive["prediction"].astype(np.float32)
            da_target = archive["target"].astype(np.float32)
        with np.load(rt_path, allow_pickle=False) as archive:
            rt_prediction = archive["prediction"].astype(np.float32)
            rt_target = archive["target"].astype(np.float32)
        if not np.allclose(da_target, target_da[node_index], rtol=0.0, atol=1e-5):
            raise ValueError(f"XGBoost DA真值不一致: {node}")
        if not np.allclose(rt_target, target_rt[node_index], rtol=0.0, atol=1e-5):
            raise ValueError(f"XGBoost RT-at-DA真值不一致: {node}")
        xgb_da.append(da_prediction)
        xgb_rt_at_da.append(rt_prediction)
    xgb_da = np.stack(xgb_da)
    xgb_rt_at_da = np.stack(xgb_rt_at_da)

    residual = xgb_da[None, ...] - global_prediction
    residual_mean = residual.mean(axis=0)
    residual_std = residual.std(axis=0)
    mean_corrected = xgb_da + float(args.weight) * residual_mean
    per_seed_corrected = xgb_da[None, ...] + float(args.weight) * residual
    per_seed_direction = np.sign(per_seed_corrected - xgb_rt_at_da[None, ...])
    direction_consensus = np.all(
        per_seed_direction == per_seed_direction[0:1], axis=0
    )
    residual_direction = np.sign(residual)
    residual_consensus = np.all(
        residual_direction == residual_direction[0:1], axis=0
    )
    base_spread = xgb_da - xgb_rt_at_da
    realized_spread = target_da - target_rt
    epsilon = 1e-6
    cv100 = residual_std <= np.maximum(np.abs(residual_mean), epsilon)
    cv050 = residual_std <= 0.5 * np.maximum(np.abs(residual_mean), epsilon)

    candidates = {
        "consdir": (direction_consensus, None),
        "consres": (residual_consensus, None),
        "consboth": (direction_consensus & residual_consensus, None),
        "conscv100": (direction_consensus & cv100, None),
        "conscv050": (direction_consensus & cv050, None),
        "consclip10": (direction_consensus, 10.0),
        "consclip25": (direction_consensus, 25.0),
    }
    output_dir = ROOT / args.output_dir
    template_dir = ROOT / args.template_dir
    source_metadata = [json.loads(str(record["metadata"].item())) for record in seeds]
    reports = {}
    for family, (mask, clip) in candidates.items():
        correction = float(args.weight) * residual_mean
        if clip is not None:
            correction = np.clip(correction, -clip, clip)
        prediction = np.where(mask, xgb_da + correction, xgb_da).astype(np.float32)
        spread = prediction - xgb_rt_at_da
        reports[family] = {
            "gate_fraction": float(mask.mean()),
            "sign_flip_fraction_vs_xgboost": float(
                np.mean(np.sign(spread) != np.sign(base_spread))
            ),
            "direction_accuracy_vs_realized_spread": _direction_accuracy(
                spread, realized_spread
            ),
            "clip_usd": clip,
        }
        arrays = dict(seeds[0])
        arrays["prediction"] = prediction
        metadata = {
            "archive_version": 2,
            "family": family,
            "task": "da",
            "model": "xgboost_global_multiseed_consensus_residual_gate",
            "xgboost_weight_when_active": 1.0 + float(args.weight),
            "global_seed_mean_weight_when_active": -float(args.weight),
            "gate": family,
            "clip_usd": clip,
            "global_members": [
                {"path": path.relative_to(ROOT).as_posix(), "seed": meta.get("seed")}
                for path, meta in zip(paths, source_metadata)
            ],
            "selection_stage": "2023_inner_validation_candidate_not_final",
        }
        _save(output_dir / f"{family}_da_inner_seed0.npz", arrays, metadata)
        for task in ("rt_at_da", "rt"):
            template = _load(
                template_dir / f"{args.template_family}_{task}_inner_seed0.npz"
            )
            template_metadata = json.loads(str(template["metadata"].item()))
            template_metadata.update({
                "family": family, "task": task, "template_only": True
            })
            _save(
                output_dir / f"{family}_{task}_inner_seed0.npz",
                template,
                template_metadata,
            )

    report = {
        "experiment": "global_da_multiseed_consensus_residual_gate",
        "status": "2023_inner_validation_candidate_not_final",
        "weight": float(args.weight),
        "baseline_direction_accuracy_vs_realized_spread": _direction_accuracy(
            base_spread, realized_spread
        ),
        "ungated_seed_mean_direction_accuracy_vs_realized_spread": _direction_accuracy(
            mean_corrected - xgb_rt_at_da, realized_spread
        ),
        "diagnostics": {
            "per_seed_corrected_direction_consensus_fraction": float(
                direction_consensus.mean()
            ),
            "residual_direction_consensus_fraction": float(
                residual_consensus.mean()
            ),
            "ungated_sign_flip_fraction_vs_xgboost": float(np.mean(
                np.sign(mean_corrected - xgb_rt_at_da) != np.sign(base_spread)
            )),
        },
        "candidates": reports,
        "output_dir": args.output_dir,
    }
    output = ROOT / args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "global_da_consensus_candidates_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "diagnostics": report["diagnostics"],
        "candidates": reports,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
