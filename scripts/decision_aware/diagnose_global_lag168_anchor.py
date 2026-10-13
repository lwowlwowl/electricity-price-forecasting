#!/usr/bin/env python
"""评价168小时历史锚点本身，不训练模型、不读取2024/2025。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)


def _mae_by_node(dataset, target_key: str, anchor_key: str) -> dict:
    nodes = {}
    absolute_total = 0.0
    element_total = 0
    for entity, node_dataset in zip(dataset.entities, dataset.datasets):
        errors = []
        for index in range(len(node_dataset)):
            sample = node_dataset[index]
            error = (
                sample[anchor_key].float() - sample[target_key].float()
            ).abs().numpy()
            errors.append(error.reshape(-1))
        values = np.concatenate(errors)
        nodes[entity[1]] = float(values.mean())
        absolute_total += float(values.sum())
        element_total += int(values.size)
    return {
        "macro_mae": float(np.mean(list(nodes.values()))),
        "micro_mae": absolute_total / element_total,
        "node_mae": nodes,
        "elements": element_total,
    }


def _da_rt_da_and_spread(dataset, threshold: float) -> dict:
    da = _mae_by_node(dataset, "price_da_tgt", "price_da_lag168")
    rt_at_da = _mae_by_node(dataset, "price_rt_tgt", "price_rt_lag168")
    node_direction = {}
    node_magnitude = {}
    for entity, node_dataset in zip(dataset.entities, dataset.datasets):
        correct = []
        errors = []
        for index in range(len(node_dataset)):
            sample = node_dataset[index]
            true_spread = (
                sample["price_da_tgt"] - sample["price_rt_tgt"]
            ).numpy()
            anchor_spread = (
                sample["price_da_lag168"] - sample["price_rt_lag168"]
            ).numpy()
            important = np.abs(true_spread) >= threshold
            if important.any():
                correct.append(
                    (np.signbit(anchor_spread[important]) == np.signbit(true_spread[important]))
                )
            errors.append(np.abs(anchor_spread - true_spread))
        direction = np.concatenate(correct) if correct else np.asarray([], dtype=bool)
        node_direction[entity[1]] = (
            float(direction.mean()) if direction.size else None
        )
        node_magnitude[entity[1]] = float(np.concatenate(errors).mean())
    valid_direction = [value for value in node_direction.values() if value is not None]
    return {
        "da": da,
        "rt_at_da": rt_at_da,
        "spread": {
            "important_abs_true_spread_threshold": threshold,
            "macro_important_direction_accuracy": float(np.mean(valid_direction)),
            "macro_spread_mae": float(np.mean(list(node_magnitude.values()))),
            "node_important_direction_accuracy": node_direction,
            "node_spread_mae": node_magnitude,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-config",
        default="configs/decision_aware/global_ercot_lag168_da_mae_v1.yaml",
    )
    parser.add_argument(
        "--rt-config",
        default="configs/decision_aware/global_ercot_lag168_rt_mae_v1.yaml",
    )
    parser.add_argument("--direction-threshold", type=float, default=6.0)
    parser.add_argument(
        "--output", default="data/results/global_lag168_anchor_diagnosis_v1.json"
    )
    args = parser.parse_args()
    if args.direction_threshold < 0:
        raise ValueError("direction-threshold不能为负")

    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        ["ERCOT"]
    )
    entities = [("ERCOT", node) for node in mappings.node_to_id]
    da_cfg = PilotConfig.from_yaml(ROOT / args.da_config)
    rt_cfg = PilotConfig.from_yaml(ROOT / args.rt_config)
    _, da_validation = build_global_task_datasets(
        da_cfg, "da", entities, mappings,
        train_stage="inner_train", validation_stage="inner_validation",
    )
    _, rt_validation = build_global_task_datasets(
        rt_cfg, "rt", entities, mappings,
        train_stage="inner_train", validation_stage="inner_validation",
    )
    report = {
        "experiment": "global_lag168_anchor_diagnosis_v1",
        "status": "diagnostic_only_not_model_selection",
        "split": "2023 inner validation only",
        "nodes": len(entities),
        "2024_used": False,
        "2025_used": False,
        **_da_rt_da_and_spread(da_validation, args.direction_threshold),
        "rt": _mae_by_node(rt_validation, "price_rt_tgt", "price_rt_lag168"),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "da_mae": report["da"]["macro_mae"],
        "rt_at_da_mae": report["rt_at_da"]["macro_mae"],
        "rt_mae": report["rt"]["macro_mae"],
        "spread_direction_accuracy": report["spread"]["macro_important_direction_accuracy"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
