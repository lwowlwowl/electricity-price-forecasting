#!/usr/bin/env python
"""Fit source-market regret trees and emit strict held-market 2023 routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.global_data import MARKET_ORDER  # noqa: E402
from decision_aware.tail_expert_router import (  # noqa: E402
    build_feature_cube,
    fit_regret_tree,
    predict_regret,
    regret_tree_explanation,
)


FAMILIES = ("economic_regret_tree", "mae_regret_tree", "economic_no_tail_tree")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _triplet(root: Path, market: str, family: str) -> dict[str, dict[str, np.ndarray]]:
    directory = root / market.lower()
    return {
        task: _load(directory / f"{family}_{task}_inner_seed0.npz")
        for task in ("da", "rt_at_da", "rt")
    }


def _feature_record(root: Path, market: str) -> dict:
    transformer = _triplet(root, market, "transformer")
    dlinear = _triplet(root, market, "static_hybrid")
    cube, names = build_feature_cube(
        transformer["da"]["prediction"],
        transformer["rt_at_da"]["prediction"],
        dlinear["da"]["prediction"],
        dlinear["rt_at_da"]["prediction"],
        transformer["da"]["dates"],
    )
    return {
        "transformer": transformer,
        "dlinear": dlinear,
        "features": cube,
        "names": names,
        "dates": transformer["da"]["dates"].astype(str),
        "nodes": transformer["da"]["nodes"].astype(str),
    }


def _system_label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def _economic_advantage(record: dict, settlement: dict) -> np.ndarray:
    transformer_nodes = settlement["systems"][_system_label("transformer")]["nodes"]
    dlinear_nodes = settlement["systems"][_system_label("static_hybrid")]["nodes"]
    result = np.full(record["features"].shape[:2], np.nan, dtype=np.float64)
    date_to_index = {value: index for index, value in enumerate(record["dates"])}
    for node_index, node in enumerate(record["nodes"]):
        left = transformer_nodes[node]["daily_net_revenue"]
        right = dlinear_nodes[node]["daily_net_revenue"]
        if set(left) != set(right):
            raise ValueError(f"{node}两静态系统结算日期不一致")
        for date in left:
            result[node_index, date_to_index[date]] = float(right[date]) - float(left[date])
    return result


def _mae_advantage(record: dict) -> np.ndarray:
    transformer = record["transformer"]
    dlinear = record["dlinear"]
    t_error = (
        np.abs(transformer["da"]["prediction"] - transformer["da"]["target_da"]).mean(axis=-1)
        + np.abs(transformer["rt_at_da"]["prediction"] - transformer["rt_at_da"]["target_rt"]).mean(axis=-1)
    ) / 2.0
    d_error = (
        np.abs(dlinear["da"]["prediction"] - dlinear["da"]["target_da"]).mean(axis=-1)
        + np.abs(dlinear["rt_at_da"]["prediction"] - dlinear["rt_at_da"]["target_rt"]).mean(axis=-1)
    ) / 2.0
    return t_error.astype(np.float64) - d_error.astype(np.float64)


def _training_rows(records: dict[str, dict], settlements: dict[str, dict], sources: list[str]):
    features: list[np.ndarray] = []
    economic: list[np.ndarray] = []
    mae: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for market in sources:
        record = records[market]
        advantage = _economic_advantage(record, settlements[market])
        valid = np.isfinite(advantage)
        x = record["features"][valid]
        features.append(x)
        economic.append(advantage[valid])
        mae.append(_mae_advantage(record)[valid])
        # Each source market has equal total fitting weight; nodes/days remain equal within market.
        weights.append(np.full(len(x), 1.0 / len(x), dtype=np.float64))
    scale = sum(len(value) for value in weights) / len(weights)
    return (
        np.concatenate(features),
        np.concatenate(economic),
        np.concatenate(mae),
        np.concatenate([value * scale for value in weights]),
    )


def _copy_family(record: dict, output: Path, family: str, mask: np.ndarray | None, metadata: dict) -> None:
    for task in ("da", "rt_at_da", "rt"):
        transformer = record["transformer"][task]
        dlinear = record["dlinear"][task]
        arrays = {key: value for key, value in transformer.items() if key != "metadata"}
        if task != "rt" and mask is not None:
            arrays["prediction"] = np.where(
                mask[:, :, None], dlinear["prediction"], transformer["prediction"]
            ).astype(np.float32)
        arrays["metadata"] = np.asarray(json.dumps({
            **metadata,
            "family": family,
            "task": task,
            "dlinear_selected_node_day_rate": float(mask.mean()) if mask is not None else (1.0 if family == "static_hybrid" and task != "rt" else 0.0),
        }, ensure_ascii=False, sort_keys=True))
        np.savez_compressed(output / f"{family}_{task}_inner_seed0.npz", **arrays)


def build_target(input_root: Path, settlement_root: Path, output_root: Path, target: str) -> dict:
    records = {market: _feature_record(input_root, market) for market in MARKET_ORDER}
    names = records[target]["names"]
    if any(record["names"] != names for record in records.values()):
        raise ValueError("市场间路由特征定义不一致")
    settlements = {
        market: json.loads((settlement_root / f"{market.lower()}.json").read_text(encoding="utf-8"))
        for market in MARKET_ORDER
    }
    sources = [market for market in MARKET_ORDER if market != target]
    x, economic_y, mae_y, sample_weight = _training_rows(records, settlements, sources)
    full_models = {
        "economic_regret_tree": fit_regret_tree(x, economic_y, sample_weight=sample_weight),
        "mae_regret_tree": fit_regret_tree(x, mae_y, sample_weight=sample_weight),
    }
    no_tail_indices = np.asarray([
        index for index, name in enumerate(names)
        if "disagreement" in name or name in {"month_sin", "month_cos", "dow_sin", "dow_cos", "weekend"}
    ], dtype=np.int64)
    full_models["economic_no_tail_tree"] = fit_regret_tree(
        x[:, no_tail_indices], economic_y, sample_weight=sample_weight
    )
    target_features = records[target]["features"]
    flat_target = target_features.reshape(-1, target_features.shape[-1])
    predictions = {
        "economic_regret_tree": predict_regret(full_models["economic_regret_tree"], flat_target),
        "mae_regret_tree": predict_regret(full_models["mae_regret_tree"], flat_target),
        "economic_no_tail_tree": predict_regret(full_models["economic_no_tail_tree"], flat_target[:, no_tail_indices]),
    }
    output = output_root / target.lower()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "protocol": "crossmarket_history_assisted_economic_regret_router_v1",
        "evaluation_year": 2023,
        "router_training_markets": sources,
        "dlinear_training_markets": list(MARKET_ORDER),
        "transformer_training_markets": sources,
        "holdout_market": target,
        "target_market_2020_2022_history_used_for_dlinear": True,
        "target_market_2023_labels_used_for_router_training": False,
        "uses_market_or_node_id": False,
        "rolling_rt_expert": "transformer",
        "selection_rule": "choose DLinear for both DA and RT-at-DA iff predicted advantage > 0",
        "locked_2025_read": False,
    }
    _copy_family(records[target], output, "transformer", None, common)
    _copy_family(records[target], output, "static_hybrid", np.ones(target_features.shape[:2], dtype=bool), common)
    explanations = {}
    for family in FAMILIES:
        mask = predictions[family].reshape(target_features.shape[:2]) > 0.0
        _copy_family(records[target], output, family, mask, common)
        if family == "economic_no_tail_tree":
            feature_names = tuple(names[index] for index in no_tail_indices)
        else:
            feature_names = names
        explanations[family] = {
            **regret_tree_explanation(full_models[family], feature_names),
            "dlinear_selected_node_day_rate": float(mask.mean()),
            "predicted_advantage_mean": float(predictions[family].mean()),
            "predicted_advantage_positive_mean": float(predictions[family][predictions[family] > 0].mean()) if mask.any() else None,
        }
    manifest = {
        **common,
        "sample_unit": "node_delivery_day",
        "source_training_rows": int(len(x)),
        "source_market_equal_sample_weight": True,
        "target_rows": int(len(flat_target)),
        "feature_count": int(x.shape[1]),
        "no_tail_feature_names": [names[index] for index in no_tail_indices],
        "models": explanations,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/results/crossmarket_regret_router_profit_inputs_v1")
    parser.add_argument("--settlement-root", default="data/results/crossmarket_regret_router_base_settlement_v1")
    parser.add_argument("--output-root", default="data/results/crossmarket_regret_router_candidates_v1")
    parser.add_argument("--markets", default=",".join(MARKET_ORDER))
    args = parser.parse_args()
    markets = [value.strip().upper() for value in args.markets.split(",") if value.strip()]
    reports = [
        build_target(ROOT / args.input_root, ROOT / args.settlement_root, ROOT / args.output_root, market)
        for market in markets
    ]
    print(json.dumps({"event": "crossmarket_regret_routes_ready", "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
