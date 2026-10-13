#!/usr/bin/env python
"""四市场代表节点的三任务Dataset与模型前向只读烟雾测试。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    build_global_task_datasets,
    flatten_rt_delivery_day_batch,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


TASKS = {
    "da": (DecisionAwareDAForecaster, "p_da", (24,)),
    "rt_at_da": (DecisionAwareRTAtDAForecaster, "p_rt_at_da", (24,)),
    "rt": (DecisionAwareRTForecaster, "p_rt", (4,)),
}


def _source_hashes() -> dict[str, str]:
    audit = json.loads(
        (ROOT / "data/results/multimarket_global_data_audit.json").read_text(
            encoding="utf-8"
        )
    )
    verified = {}
    for market in MARKET_ORDER:
        record = audit["markets"][market]
        path = ROOT / Path(str(record["path"]).replace("\\", "/"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"{market}源数据哈希与冻结审计不一致")
        verified[market] = digest
    return verified


def run(config_path: Path, device: torch.device, nodes_per_market: int) -> dict:
    if nodes_per_market < 1:
        raise ValueError("nodes_per_market必须至少为1")
    mappings = load_formal_entity_mappings(ROOT / "configs/nodes.yaml")
    cfg = PilotConfig.from_yaml(str(config_path))
    cfg = replace(
        cfg,
        global_num_nodes=len(mappings.node_to_id),
        global_num_markets=len(mappings.market_to_id),
        global_use_critical_event_weighting=False,
        use_amp=False,
    )
    raw_nodes = yaml.safe_load(
        (ROOT / "configs/nodes.yaml").read_text(encoding="utf-8")
    )
    report = {
        "scope": "read-only four-market representative-node three-task smoke",
        "split_usage": {
            "train": "2020-2022 inner train",
            "validation": "2023 inner validation",
            "2024_used_for_selection": False,
            "2025_used": False,
        },
        "device": str(device),
        "source_sha256": _source_hashes(),
        "markets": {},
    }
    for market in MARKET_ORDER:
        nodes = [str(node) for node in raw_nodes[market]["ablation"]][
            :nodes_per_market
        ]
        entities = [(market, node) for node in nodes]
        market_record = {"nodes": nodes, "tasks": {}}
        for task_name, (model_cls, prediction_key, expected_tail) in TASKS.items():
            started = perf_counter()
            train_dataset, validation_dataset = build_global_task_datasets(
                cfg,
                task_name,
                entities,
                mappings,
                train_stage=AnnualStage.INNER_TRAIN,
                validation_stage=AnnualStage.INNER_VALIDATION,
            )
            batch = next(iter(DataLoader(validation_dataset, batch_size=2)))
            if task_name == "rt":
                batch = flatten_rt_delivery_day_batch(batch)
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            model = model_cls(cfg).to(device).eval()
            with torch.no_grad():
                output = model(batch)[prediction_key]
            if tuple(output.shape[1:]) != expected_tail:
                raise ValueError(
                    f"{market}/{task_name}输出形状异常: {tuple(output.shape)}"
                )
            if not bool(torch.isfinite(output).all()):
                raise ValueError(f"{market}/{task_name}输出含NaN或Inf")
            market_record["tasks"][task_name] = {
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "forward_shape": list(output.shape),
                "finite": True,
                "elapsed_seconds": perf_counter() - started,
            }
            del model, batch, output, train_dataset, validation_dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["markets"][market] = market_record
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/decision_aware/global_multimarket_da_mae_v1.yaml",
    )
    parser.add_argument("--nodes-per-market", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", default="data/results/multimarket_three_task_smoke_v1.json"
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    report = run(ROOT / args.config, device, args.nodes_per_market)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved": output.relative_to(ROOT).as_posix(),
        "markets": list(report["markets"]),
        "all_finite": all(
            task["finite"]
            for market in report["markets"].values()
            for task in market["tasks"].values()
        ),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
