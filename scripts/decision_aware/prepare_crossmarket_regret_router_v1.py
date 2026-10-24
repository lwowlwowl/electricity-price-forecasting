#!/usr/bin/env python
"""Build strict 2023 LOMO Transformer/DLinear archives for regret routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.global_data import MARKET_ORDER  # noqa: E402
from decision_aware.loader_v2 import market_timezone  # noqa: E402


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _timestamps(dates: np.ndarray, market: str) -> np.ndarray:
    timezone = market_timezone(market)
    rows: list[list[str]] = []
    for value in dates.astype(str):
        naive_start = pd.Timestamp(str(value))
        start = naive_start.tz_localize(timezone)
        end = (naive_start + pd.Timedelta(days=1)).tz_localize(timezone)
        stamps = pd.date_range(start, end, inclusive="left", freq="h")
        if len(stamps) != 24:
            raise ValueError(f"{market} {value}不是24小时交付日；不能合成执行时间戳")
        rows.append([stamp.isoformat() for stamp in stamps])
    return np.asarray(rows)


def _same(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label}不一致")


def _source_paths(market: str, dlinear_root: Path) -> dict[str, dict[str, Path]]:
    lower = market.lower()
    return {
        "transformer": {
            "da": ROOT / "data/results/universal_lomo_da_v1" / f"{lower}_da_v1.npz",
            "rt_at_da": ROOT / "data/results/universal_lomo_rt_at_da_v1" / f"{lower}_rt_at_da_v1.npz",
            "rt": ROOT / "data/results/universal_lomo_rt_v1" / f"{lower}_rt_v1.npz",
        },
        "static_hybrid": {
            "da": dlinear_root / f"{lower}_da_inner.npz",
            "rt_at_da": dlinear_root / f"{lower}_rt_at_da_inner.npz",
            "rt": ROOT / "data/results/universal_lomo_rt_v1" / f"{lower}_rt_v1.npz",
        },
    }


def prepare_market(market: str, dlinear_root: Path, output_root: Path) -> dict:
    paths = _source_paths(market, dlinear_root)
    sources = {
        family: {task: _load(path) for task, path in task_paths.items()}
        for family, task_paths in paths.items()
    }
    transformer = sources["transformer"]
    dlinear = sources["static_hybrid"]
    for family, tasks in sources.items():
        for task, item in tasks.items():
            if not np.array_equal(transformer["da"]["nodes"], item["nodes"]):
                raise ValueError(f"{market} {family}/{task}节点顺序不一致")
        if not np.array_equal(tasks["da"]["dates"], tasks["rt_at_da"]["dates"]):
            raise ValueError(f"{market} {family} DA与RT-at-DA日期不一致")
    _same(transformer["da"]["target"], transformer["rt_at_da"]["target_da"], f"{market} Transformer DA真值")
    _same(dlinear["da"]["target"], transformer["da"]["target"], f"{market} DLinear DA真值")
    _same(dlinear["rt_at_da"]["target"], transformer["rt_at_da"]["target"], f"{market} DLinear RT-at-DA真值")

    output_dir = output_root / market.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = (
        transformer["rt"].get("executed_timestamps")
        if "executed_timestamps" in transformer["rt"]
        else _timestamps(transformer["rt"]["dates"], market)
    )
    timezone = np.asarray(market_timezone(market))
    for family, tasks in sources.items():
        for task, item in tasks.items():
            arrays: dict[str, np.ndarray] = {
                "nodes": item["nodes"],
                "dates": item["dates"],
                "prediction": item["prediction"].astype(np.float32),
                "target_rt": (
                    transformer["rt_at_da"]["target"].astype(np.float32)
                    if task != "rt"
                    else transformer["rt"]["target"].astype(np.float32)
                ),
                "metadata": np.asarray(json.dumps({
                    "protocol": "crossmarket_regret_router_lomo_2023_v1",
                    "market": market,
                    "family": family,
                    "task": task,
                    "training_markets": [value for value in MARKET_ORDER if value != market],
                    "target_market_labels_used_for_training": False,
                    "selection_allowed": True,
                    "locked_2025_read": False,
                }, ensure_ascii=False, sort_keys=True)),
            }
            if task == "da":
                arrays["target_da"] = transformer["da"]["target"].astype(np.float32)
            if task == "rt":
                arrays["executed_timestamps"] = timestamps
                arrays["timezone"] = timezone
            np.savez_compressed(output_dir / f"{family}_{task}_inner_seed0.npz", **arrays)
    return {
        "market": market,
        "nodes": int(len(transformer["da"]["nodes"])),
        "da_days": int(len(transformer["da"]["dates"])),
        "rt_days": int(len(transformer["rt"]["dates"])),
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlinear-root", default="data/results/crossmarket_regret_router_lomo_dlinear_v1")
    parser.add_argument("--output-root", default="data/results/crossmarket_regret_router_profit_inputs_v1")
    parser.add_argument("--markets", default=",".join(MARKET_ORDER))
    args = parser.parse_args()
    markets = [value.strip().upper() for value in args.markets.split(",") if value.strip()]
    unknown = sorted(set(markets) - set(MARKET_ORDER))
    if unknown or not markets:
        raise ValueError(f"markets无效: {unknown}")
    reports = [
        prepare_market(market, ROOT / args.dlinear_root, ROOT / args.output_root)
        for market in markets
    ]
    manifest = {
        "experiment": "crossmarket_regret_router_profit_inputs_v1",
        "year": 2023,
        "markets": reports,
        "locked_2025_read": False,
    }
    output = ROOT / args.output_root / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
