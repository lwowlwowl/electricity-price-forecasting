#!/usr/bin/env python
"""在训练从未包含的整个市场上一次性评价无身份Global checkpoint。"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    flatten_rt_delivery_day_batch,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.loader_v2 import market_timezone  # noqa: E402


TASKS = {
    "da": (DecisionAwareDAForecaster, "p_da", "price_da_tgt"),
    "rt_at_da": (
        DecisionAwareRTAtDAForecaster,
        "p_rt_at_da",
        "price_rt_at_da_tgt",
    ),
    "rt": (DecisionAwareRTForecaster, "p_rt", "price_rt_tgt"),
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_config(checkpoint: dict) -> PilotConfig:
    allowed = {field.name for field in fields(PilotConfig)}
    return PilotConfig(**{
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed
    })


def _dates(node_dataset, task: str) -> np.ndarray:
    if task == "rt":
        return np.asarray([str(day) for day, _ in node_dataset.days])
    return np.asarray([
        str(node_dataset.sample_time_contract(index)["delivery_date"])
        for index in range(len(node_dataset))
    ])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument(
        "--holdout-market", required=True, choices=["CAISO", "ERCOT", "NYISO", "PJM"]
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-stage", choices=("inner", "outer_frozen"), default="inner")
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
    if checkpoint.get("task") != args.task:
        raise ValueError("checkpoint任务与--task不一致")
    cfg = _checkpoint_config(checkpoint)
    if args.evaluation_stage == "outer_frozen" and checkpoint.get("phase") != "inner":
        raise ValueError("冻结2024检查必须使用原inner checkpoint，不得替换为重训模型")
    training_markets = tuple(checkpoint.get("global_training_markets", ()))
    if args.holdout_market in training_markets:
        raise ValueError("留出市场出现在checkpoint训练市场中，不能称为零样本验证")
    if cfg.global_use_entity_embeddings:
        raise ValueError("通用留出市场评价禁止node_id/market_id embedding")
    if str(cfg.global_normalization_mode).lower() != "context_robust":
        raise ValueError("通用留出市场评价要求context_robust归一化")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model_class, prediction_key, target_key = TASKS[args.task]
    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    mappings = load_formal_entity_mappings(
        ROOT / "configs" / "nodes.yaml"
    ).subset([args.holdout_market])
    entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    # context_robust下train_dataset不会给验证集提供任何拟合统计量；这里只
    # 复用统一builder的窗口合同，实际预测只读取每个起报点以前的历史。
    evaluation_cfg = replace(cfg, global_use_critical_event_weighting=True)
    validation_stage = (
        AnnualStage.OUTER_VALIDATION if args.evaluation_stage == "outer_frozen"
        else AnnualStage.INNER_VALIDATION
    )
    evaluation_end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(validation_stage).end
    read_end_utc = (
        pd.Timestamp(evaluation_end, tz=market_timezone(args.holdout_market))
        + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    _, validation_dataset = build_global_task_datasets(
        evaluation_cfg,
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=validation_stage,
        read_end_utc=read_end_utc,
    )

    node_predictions: list[np.ndarray] = []
    node_targets: list[np.ndarray] = []
    node_target_da: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    per_node_mae: dict[str, float] = {}
    per_node_rmse: dict[str, float] = {}
    for entity_index, ((market, node), node_dataset) in enumerate(
        zip(validation_dataset.entities, validation_dataset.datasets)
    ):
        start = validation_dataset.offsets[entity_index]
        end = validation_dataset.offsets[entity_index + 1]
        loader = DataLoader(
            Subset(validation_dataset, range(start, end)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        predictions = []
        targets = []
        targets_da = []
        for raw_batch in loader:
            targets.append(raw_batch[target_key].cpu().numpy())
            if args.task == "rt_at_da":
                targets_da.append(
                    raw_batch["price_da_tgt_weighting_only"].cpu().numpy()
                )
            batch = (
                flatten_rt_delivery_day_batch(raw_batch)
                if args.task == "rt" else dict(raw_batch)
            )
            # 这是部署合同的直接测试：模型前向中根本不提供两个身份字段。
            batch.pop("node_id", None)
            batch.pop("market_id", None)
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            prediction = model(batch)[prediction_key].detach().float().cpu().numpy()
            if args.task == "rt":
                prediction = prediction.reshape(-1, 24, prediction.shape[-1])
            predictions.append(prediction)
        node_prediction = np.concatenate(predictions).astype(np.float32)
        node_target = np.concatenate(targets).astype(np.float32)
        dates = _dates(node_dataset, args.task)
        if common_dates is None:
            common_dates = dates
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"留出市场各节点日期不一致，首次冲突节点={node}")
        error = node_prediction - node_target
        per_node_mae[node] = float(np.mean(np.abs(error)))
        per_node_rmse[node] = float(np.sqrt(np.mean(np.square(error))))
        node_predictions.append(node_prediction)
        node_targets.append(node_target)
        if targets_da:
            node_target_da.append(np.concatenate(targets_da).astype(np.float32))
        print(json.dumps({
            "event": "unseen_node_complete",
            "market": market,
            "node": node,
            "mae": per_node_mae[node],
        }, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "leave_one_market_out_zero_shot",
        "evaluation_stage": args.evaluation_stage,
        "checkpoint_phase": checkpoint.get("phase"),
        "retrained": False,
        "selection_allowed": args.evaluation_stage == "inner",
        "data_read_end_utc_exclusive": read_end_utc,
        "task": args.task,
        "holdout_market": args.holdout_market,
        "training_markets": list(training_markets),
        "uses_node_id_for_prediction": False,
        "uses_market_id_for_prediction": False,
        "normalization_mode": cfg.global_normalization_mode,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "node_macro_mae": float(np.mean(list(per_node_mae.values()))),
        "node_macro_rmse": float(np.mean(list(per_node_rmse.values()))),
        "per_node_mae": per_node_mae,
        "per_node_rmse": per_node_rmse,
        "final_profit_claim_allowed": False,
        "note": "单任务留出市场筛选；完整盈利必须等待通用DA/RT-at-DA/RT三模型。",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "nodes": np.asarray([node for _, node in validation_dataset.entities]),
        "dates": common_dates,
        "prediction": np.stack(node_predictions),
        "target": np.stack(node_targets),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if node_target_da:
        arrays["target_da"] = np.stack(node_target_da)
    np.savez_compressed(output, **arrays)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
