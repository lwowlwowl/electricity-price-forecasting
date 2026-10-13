#!/usr/bin/env python
"""把自有RT-at-DA价差的方向与幅度做可审计因子分解。"""
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


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _assert_aligned(left: dict, right: dict, label: str) -> None:
    for key in ("nodes", "dates", "target_rt"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"{label}的{key}不一致")


def _derived_metadata(
    source: dict,
    family: str,
    task: str,
    provenance: dict[str, str],
    formula: str,
    mode: str,
) -> np.ndarray:
    metadata = json.loads(str(source["metadata"].item()))
    metadata.update({
        "family": family,
        "task": task,
        "selection": "fixed interpretable direction-magnitude factorization",
        "derived_prediction": {
            "formula": formula,
            "mode": mode,
            "direction_source": "self-trained explicit-spread RT-at-DA model",
            "magnitude_source": "self-trained pre-encoder RT-at-DA model",
            "center_source": "self-trained DA model",
            "xgboost_used": False,
            "tuned_thresholds_or_weights": False,
            "provenance_sha256": provenance,
        },
    })
    # 派生档案不冒充单一checkpoint；完整来源哈希记录在derived_prediction中。
    metadata.pop("checkpoint", None)
    metadata.pop("checkpoint_sha256", None)
    return np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


def _save_copy(
    source: dict,
    path: Path,
    family: str,
    task: str,
    provenance: dict[str, str],
    formula: str,
    mode: str,
) -> None:
    arrays = dict(source)
    arrays["metadata"] = _derived_metadata(
        source, family, task, provenance, formula, mode
    )
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", default="data/results/global_entitypre_local_predictions_v1"
    )
    parser.add_argument(
        "--direction-dir", default="data/results/global_spreadfactor_predictions_v1"
    )
    parser.add_argument(
        "--output-dir", default="data/results/global_direction_magnitude_predictions_v1"
    )
    parser.add_argument("--family", default="dirfactor")
    parser.add_argument(
        "--mode",
        choices=("hard_direction", "economic_gate"),
        default="hard_direction",
    )
    args = parser.parse_args()

    base_dir = ROOT / args.base_dir
    direction_dir = ROOT / args.direction_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "da": base_dir / "entitylocal1_da_inner_seed0.npz",
        "base_rt_at_da": base_dir / "entitylocal0_rt_at_da_inner_seed0.npz",
        "rt": base_dir / "entitylocal0_rt_inner_seed0.npz",
        "direction_rt_at_da": (
            direction_dir / "spreadfactor0_rt_at_da_inner_seed0.npz"
        ),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少{name}预测档案: {path}")
    archives = {name: _load(path) for name, path in paths.items()}
    da = archives["da"]
    base_rt_at_da = archives["base_rt_at_da"]
    direction_rt_at_da = archives["direction_rt_at_da"]
    _assert_aligned(da, base_rt_at_da, "DA与基础RT-at-DA")
    _assert_aligned(da, direction_rt_at_da, "DA与方向RT-at-DA")

    da_center = da["prediction"].astype(np.float64)
    base_rt = base_rt_at_da["prediction"].astype(np.float64)
    direction_rt = direction_rt_at_da["prediction"].astype(np.float64)
    direction_spread = da_center - direction_rt
    base_spread = da_center - base_rt
    base_magnitude = np.abs(da_center - base_rt)
    direction_sign = np.where(direction_spread >= 0.0, 1.0, -1.0)
    base_sign = np.where(base_spread >= 0.0, 1.0, -1.0)
    break_even_spread = 5.7 / 0.95
    if args.mode == "hard_direction":
        spread_sign = direction_sign
        override = np.ones_like(direction_spread, dtype=bool)
        formula = (
            "RT = DA_center - sign(explicit_spread) * abs(DA_center - base_RT)"
        )
    else:
        # 盈亏平衡线来自冻结BESS合同，不由ERCOT验证结果选择。
        override = np.abs(direction_spread) >= break_even_spread
        spread_sign = np.where(override, direction_sign, base_sign)
        formula = (
            "RT = DA_center - selected_sign * abs(DA_center - base_RT); "
            "selected_sign uses explicit-spread direction only when "
            "abs(explicit_spread) >= bess_kappa / bess_eta, otherwise base direction"
        )
    factored_rt = da_center - spread_sign * base_magnitude

    provenance = {
        name: _sha256(path)
        for name, path in paths.items()
    }
    _save_copy(
        da,
        output_dir / f"{args.family}_da_inner_seed0.npz",
        args.family,
        "da",
        provenance,
        formula,
        args.mode,
    )
    _save_copy(
        archives["rt"],
        output_dir / f"{args.family}_rt_inner_seed0.npz",
        args.family,
        "rt",
        provenance,
        formula,
        args.mode,
    )
    derived_rt_at_da = dict(base_rt_at_da)
    derived_rt_at_da["prediction"] = factored_rt.astype(np.float32)
    derived_rt_at_da["metadata"] = _derived_metadata(
        base_rt_at_da, args.family, "rt_at_da", provenance, formula, args.mode
    )
    np.savez_compressed(
        output_dir / f"{args.family}_rt_at_da_inner_seed0.npz",
        **derived_rt_at_da,
    )

    true_spread = da["target_da"].astype(np.float64) - da["target_rt"].astype(
        np.float64
    )
    report = {
        "experiment": "self_model_direction_magnitude_factorization_v1",
        "2024_used": False,
        "2025_used": False,
        "mode": args.mode,
        "formula": formula,
        "economic_break_even_spread": break_even_spread,
        "direction_override_fraction": float(override.mean()),
        "xgboost_used": False,
        "tuned_thresholds_or_weights": False,
        "all_spread_direction_accuracy": float(
            ((da_center - factored_rt > 0) == (true_spread > 0)).mean()
        ),
        "base_all_spread_direction_accuracy": float(
            ((base_spread > 0) == (true_spread > 0)).mean()
        ),
        "important_spread_direction_accuracy": float(
            (
                ((da_center - factored_rt > 0) == (true_spread > 0))
                & (np.abs(true_spread) >= 5.7)
            ).sum()
            / (np.abs(true_spread) >= 5.7).sum()
        ),
        "base_important_spread_direction_accuracy": float(
            (
                ((base_spread > 0) == (true_spread > 0))
                & (np.abs(true_spread) >= 5.7)
            ).sum()
            / (np.abs(true_spread) >= 5.7).sum()
        ),
        "predicted_abs_spread_mean": float(np.abs(da_center - factored_rt).mean()),
        "base_predicted_abs_spread_mean": float(
            np.abs(da_center - base_rt).mean()
        ),
        "true_abs_spread_mean": float(np.abs(true_spread).mean()),
        "provenance_sha256": provenance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
