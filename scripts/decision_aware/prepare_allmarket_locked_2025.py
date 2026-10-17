#!/usr/bin/env python
"""校验四市场最终预测、构造固定三seed集成及ERCOT结算档案。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
TASKS = ("da", "rt_at_da", "rt")
SEEDS = (0, 1, 2)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label}不一致")


def _validate_seed_group(items: list[dict[str, np.ndarray]], market: str, task: str) -> None:
    reference = items[0]
    for seed, item in enumerate(items):
        metadata = json.loads(str(item["metadata"].item()))
        if metadata.get("protocol") != "all_market_locked_2025":
            raise ValueError(f"{market}/{task}/seed{seed}不是锁定2025协议")
        if metadata.get("selection_allowed") is not False:
            raise ValueError(f"{market}/{task}/seed{seed}错误允许2025选模")
        if metadata.get("checkpoint_phase") != "final":
            raise ValueError(f"{market}/{task}/seed{seed}不是final checkpoint")
        if metadata.get("evaluation_market") != market:
            raise ValueError(f"{market}/{task}/seed{seed}市场元数据错误")
        if seed:
            if not np.array_equal(reference["nodes"], item["nodes"]):
                raise ValueError(f"{market}/{task}三seed节点不一致")
            if not np.array_equal(reference["dates"], item["dates"]):
                raise ValueError(f"{market}/{task}三seed日期不一致")
            _same(reference["target"], item["target"], f"{market}/{task}三seed真值")


def _source_path(input_dir: Path, market: str, task: str, seed: int) -> Path:
    return input_dir / f"{market.lower()}_{task}_seed{seed}.npz"


def _ensemble_path(output_dir: Path, market: str, task: str) -> Path:
    return output_dir / f"{market.lower()}_{task}_ensemble.npz"


def _write_ensemble(
    input_dir: Path, output_dir: Path, market: str, task: str
) -> tuple[dict, list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    items = [_load(_source_path(input_dir, market, task, seed)) for seed in SEEDS]
    _validate_seed_group(items, market, task)
    prediction = np.mean(
        np.stack([item["prediction"].astype(np.float64) for item in items]), axis=0
    ).astype(np.float32)
    target = items[0]["target"].astype(np.float32)
    error = prediction.astype(np.float64) - target.astype(np.float64)
    metadata = {
        "protocol": "all_market_locked_2025",
        "evaluation_market": market,
        "task": task,
        "family": "fixed_seed012_mean",
        "seeds": list(SEEDS),
        "selection_allowed": False,
        "weights": [1 / 3, 1 / 3, 1 / 3],
    }
    arrays: dict[str, np.ndarray] = {
        "nodes": items[0]["nodes"],
        "dates": items[0]["dates"],
        "prediction": prediction,
        "target": target,
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    for key in ("target_da", "executed_timestamps", "timezone"):
        if key in items[0]:
            arrays[key] = items[0][key]
    output = _ensemble_path(output_dir, market, task)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return ({
        "market": market,
        "task": task,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "elements": int(error.size),
        "output": output.relative_to(ROOT).as_posix(),
    }, items, arrays)


def _write_ercot_profit_archives(
    input_dir: Path, ensemble_dir: Path, output_dir: Path
) -> None:
    families = {f"seed{seed}": seed for seed in SEEDS}
    families["ensemble"] = None
    for family, seed in families.items():
        source = {}
        for task in TASKS:
            path = (
                _source_path(input_dir, "ERCOT", task, seed)
                if seed is not None
                else _ensemble_path(ensemble_dir, "ERCOT", task)
            )
            source[task] = _load(path)
        if not np.array_equal(source["da"]["nodes"], source["rt_at_da"]["nodes"]):
            raise ValueError(f"{family}的DA与RT-at-DA节点不一致")
        if not np.array_equal(source["da"]["dates"], source["rt_at_da"]["dates"]):
            raise ValueError(f"{family}的DA与RT-at-DA日期不一致")
        _same(source["da"]["target"], source["rt_at_da"]["target_da"], f"{family} DA真值")
        output_dir.mkdir(parents=True, exist_ok=True)
        common_meta = {
            "protocol": "all_market_locked_2025",
            "family": family,
            "phase": "final",
            "selection_allowed": False,
            "seed": seed,
        }
        for task in TASKS:
            item = source[task]
            arrays: dict[str, np.ndarray] = {
                "nodes": item["nodes"],
                "dates": item["dates"],
                "prediction": item["prediction"],
                "target_rt": (
                    source["rt_at_da"]["target"] if task != "rt" else item["target"]
                ),
                "metadata": np.asarray(json.dumps(
                    {**common_meta, "task": task}, ensure_ascii=False, sort_keys=True
                )),
            }
            if task == "da":
                arrays["target_da"] = item["target"]
            if task == "rt":
                arrays["executed_timestamps"] = item["executed_timestamps"]
                arrays["timezone"] = item["timezone"]
            np.savez_compressed(
                output_dir / f"{family}_{task}_final_seed0.npz", **arrays
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="data/results/allmarket_locked_2025_predictions_v1"
    )
    parser.add_argument(
        "--ensemble-dir", default="data/results/allmarket_locked_2025_ensemble_v1"
    )
    parser.add_argument(
        "--profit-dir", default="data/results/allmarket_locked_2025_ercot_profit_v1"
    )
    args = parser.parse_args()
    input_dir = ROOT / args.input_dir
    ensemble_dir = ROOT / args.ensemble_dir
    rows = []
    for market in MARKETS:
        for task in TASKS:
            row, _, _ = _write_ensemble(input_dir, ensemble_dir, market, task)
            rows.append(row)
    _write_ercot_profit_archives(input_dir, ensemble_dir, ROOT / args.profit_dir)
    report = {
        "status": "locked_2025_complete_no_model_selection_allowed",
        "selection_allowed": False,
        "markets": list(MARKETS),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "rows": rows,
    }
    (ensemble_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "locked_2025_ensemble_ready", "rows": len(rows)}))


if __name__ == "__main__":
    main()
