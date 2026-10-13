#!/usr/bin/env python
"""准备经济状态头的逐seed系统与无权重多数方向候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aligned(reference: dict, candidate: dict, label: str) -> None:
    for key in ("nodes", "dates", "target_rt"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{label}的{key}与固定DA档案不一致")


def _metadata(source: dict, family: str, task: str, provenance: dict) -> np.ndarray:
    value = json.loads(str(source["metadata"].item()))
    value.update({
        "family": family,
        "task": task,
        "selection": "fixed three-seed economic-regime robustness evaluation",
        "multiseed_provenance": provenance,
        "xgboost_used": False,
        "tuned_thresholds_or_weights": False,
    })
    return np.asarray(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _unweighted_majority(signs: np.ndarray) -> np.ndarray:
    """三个{-1,+1}方向的无权重多数票。"""
    if signs.shape[0] != 3 or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("多数票固定要求三个{-1,+1}方向数组")
    return np.where(signs.sum(axis=0) >= 0, 1.0, -1.0)


def _unanimous_or_base(
    signs: np.ndarray, base_sign: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """仅在三个seed方向一致时采用共识，否则回退冻结基础方向。"""
    if signs.shape[0] != 3 or signs.shape[1:] != base_sign.shape:
        raise ValueError("一致性弃权的seed/base形状不匹配")
    unanimous = np.all(signs == signs[0:1], axis=0)
    selected = np.where(unanimous, signs[0], base_sign)
    return selected, unanimous


def _save_task(
    source: dict,
    output_dir: Path,
    family: str,
    task: str,
    provenance: dict,
) -> None:
    arrays = dict(source)
    arrays["metadata"] = _metadata(source, family, task, provenance)
    np.savez_compressed(
        output_dir / f"{family}_{task}_inner_seed0.npz", **arrays
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-prediction-dir",
        default="data/results/global_economic_regime_multiseed_raw_v1",
    )
    parser.add_argument(
        "--fixed-prediction-dir",
        default="data/results/global_entitypre_local_predictions_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/global_economic_regime_multiseed_predictions_v1",
    )
    parser.add_argument(
        "--seed-families", default="regimeseed0,regimeseed1,regimeseed2"
    )
    parser.add_argument("--vote-family", default="regimevote")
    parser.add_argument("--unanimous-family", default="regimeunanimous")
    parser.add_argument(
        "--config",
        default="configs/decision_aware/global_ercot_economic_regime_rt_at_da_v1.yaml",
        help="合同参数来源；关键价差阈值严格由kappa/eta派生。",
    )
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(str(ROOT / args.config))
    economic_threshold = float(cfg.resolved_spread_threshold)

    seed_families = tuple(
        value.strip().lower()
        for value in args.seed_families.split(",")
        if value.strip()
    )
    if len(seed_families) != 3 or len(set(seed_families)) != 3:
        raise ValueError("多数方向候选固定要求三个不同seed family")
    seed_dir = ROOT / args.seed_prediction_dir
    fixed_dir = ROOT / args.fixed_prediction_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_paths = {
        "da": fixed_dir / "entitylocal1_da_inner_seed0.npz",
        "base_rt_at_da": fixed_dir / "entitylocal0_rt_at_da_inner_seed0.npz",
        "rt": fixed_dir / "entitylocal0_rt_inner_seed0.npz",
    }
    seed_paths = {
        family: seed_dir / f"{family}_rt_at_da_inner_seed0.npz"
        for family in seed_families
    }
    for path in [*fixed_paths.values(), *seed_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    fixed = {name: _load(path) for name, path in fixed_paths.items()}
    seeds = {family: _load(path) for family, path in seed_paths.items()}
    for family, archive in seeds.items():
        _aligned(fixed["da"], archive, family)

    provenance = {
        "fixed": {name: _sha256(path) for name, path in fixed_paths.items()},
        "seed_rt_at_da": {
            family: _sha256(path) for family, path in seed_paths.items()
        },
    }
    for family, archive in seeds.items():
        _save_task(fixed["da"], output_dir, family, "da", provenance)
        _save_task(archive, output_dir, family, "rt_at_da", provenance)
        _save_task(fixed["rt"], output_dir, family, "rt", provenance)

    da_prediction = fixed["da"]["prediction"].astype(np.float64)
    base_rt = fixed["base_rt_at_da"]["prediction"].astype(np.float64)
    base_magnitude = np.abs(da_prediction - base_rt)
    seed_signs = np.stack([
        np.where(
            da_prediction - seeds[family]["prediction"].astype(np.float64) >= 0,
            1.0,
            -1.0,
        )
        for family in seed_families
    ])
    vote_sign = _unweighted_majority(seed_signs)
    vote_rt = da_prediction - vote_sign * base_magnitude
    vote_archive = dict(seeds[seed_families[0]])
    vote_archive["prediction"] = vote_rt.astype(np.float32)
    vote_archive["metadata"] = _metadata(
        vote_archive, args.vote_family, "rt_at_da", provenance
    )
    vote_meta = json.loads(str(vote_archive["metadata"].item()))
    vote_meta["multiseed_direction_vote"] = {
        "seed_families": list(seed_families),
        "rule": "unweighted majority of final DA-minus-RT-at-DA direction",
        "magnitude": "absolute spread from frozen pre-encoder base model",
    }
    vote_archive["metadata"] = np.asarray(
        json.dumps(vote_meta, ensure_ascii=False, sort_keys=True)
    )
    _save_task(fixed["da"], output_dir, args.vote_family, "da", provenance)
    np.savez_compressed(
        output_dir / f"{args.vote_family}_rt_at_da_inner_seed0.npz",
        **vote_archive,
    )
    _save_task(fixed["rt"], output_dir, args.vote_family, "rt", provenance)

    base_sign = np.where(da_prediction - base_rt >= 0, 1.0, -1.0)
    unanimous_sign, unanimous_mask = _unanimous_or_base(
        seed_signs, base_sign
    )
    unanimous_rt = da_prediction - unanimous_sign * base_magnitude
    unanimous_archive = dict(seeds[seed_families[0]])
    unanimous_archive["prediction"] = unanimous_rt.astype(np.float32)
    unanimous_archive["metadata"] = _metadata(
        unanimous_archive, args.unanimous_family, "rt_at_da", provenance
    )
    unanimous_meta = json.loads(str(unanimous_archive["metadata"].item()))
    unanimous_meta["multiseed_direction_unanimity"] = {
        "seed_families": list(seed_families),
        "rule": "use seed direction only when all three final directions agree; otherwise preserve frozen base direction",
        "magnitude": "absolute spread from frozen pre-encoder base model",
        "tuned_thresholds_or_weights": False,
    }
    unanimous_archive["metadata"] = np.asarray(
        json.dumps(unanimous_meta, ensure_ascii=False, sort_keys=True)
    )
    _save_task(
        fixed["da"], output_dir, args.unanimous_family, "da", provenance
    )
    np.savez_compressed(
        output_dir / f"{args.unanimous_family}_rt_at_da_inner_seed0.npz",
        **unanimous_archive,
    )
    _save_task(
        fixed["rt"], output_dir, args.unanimous_family, "rt", provenance
    )

    true_spread = (
        fixed["da"]["target_da"].astype(np.float64)
        - fixed["da"]["target_rt"].astype(np.float64)
    )
    important = np.abs(true_spread) >= economic_threshold
    report = {
        "experiment": "global_economic_regime_three_seed_direction_vote_v1",
        "2024_used": False,
        "2025_used": False,
        "xgboost_used": False,
        "tuned_thresholds_or_weights": False,
        "economic_spread_threshold_usd_per_mwh": economic_threshold,
        "economic_spread_threshold_source": "battery contract kappa/eta",
        "seed_families": list(seed_families),
        "vote_family": args.vote_family,
        "unanimous_family": args.unanimous_family,
        "unanimous_seed_direction_rate": float(unanimous_mask.mean()),
        "unanimous_intervention_rate": float(
            (unanimous_mask & (unanimous_sign != base_sign)).mean()
        ),
        "all_spread_direction_accuracy": float(
            ((vote_sign > 0) == (true_spread > 0)).mean()
        ),
        "important_spread_direction_accuracy": float(
            (((vote_sign > 0) == (true_spread > 0)) & important).sum()
            / important.sum()
        ),
        "provenance_sha256": provenance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
