#!/usr/bin/env python
"""TimesFM-2.5最终LoRA：2020—2024训练与锁定2025推理分离。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    MarketNodeBalancedSampler,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)
from scripts.decision_aware.preflight_timesfm25_lora import MODEL_ID  # noqa: E402
from scripts.decision_aware.run_universal_lomo_timesfm25_lora import (  # noqa: E402
    HORIZONS,
    RawTargetWindowDataset,
    _entities,
    _flatten_batch,
    _mean_prediction,
    _predict_node,
    set_seed,
    sha256,
    task_loss,
)


TASKS = ("da", "rt_at_da", "rt")
FINAL_READ_END_UTC = {
    "CAISO": "2025-01-01T08:00:00+00:00",
    "ERCOT": "2025-01-01T06:00:00+00:00",
    "NYISO": "2025-01-01T05:00:00+00:00",
    "PJM": "2025-01-01T05:00:00+00:00",
}


def _load_protocol(path: Path, task: str) -> dict:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if protocol.get("protocol_version") != "allmarket_final_baselines_v1":
        raise ValueError("不是冻结的最终基线协议")
    if protocol.get("training_years") != [2020, 2021, 2022, 2023, 2024]:
        raise ValueError("TimesFM-2.5最终训练年份必须为2020—2024")
    section = protocol.get("timesfm25", {})
    if task not in section.get("tasks", []):
        raise ValueError(f"TimesFM-2.5协议未登记任务: {task}")
    if int(section.get("seed", -1)) != 0:
        raise ValueError("TimesFM-2.5最终基线固定seed 0")
    if int(section.get("fixed_epochs", {}).get(task, 0)) < 1:
        raise ValueError(f"TimesFM-2.5缺少{task}固定epoch")
    return protocol


def _validate_data_config(cfg: PilotConfig) -> None:
    if set(str(item).upper() for item in cfg.global_training_markets) != set(MARKET_ORDER):
        raise ValueError("TimesFM-2.5最终基线必须使用四市场配置")
    if cfg.context_len != 168 or cfg.global_use_entity_embeddings:
        raise ValueError("TimesFM-2.5要求168小时且不使用实体ID")


def _load_base_and_lora(args: argparse.Namespace, section: dict, *, trainable: bool):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_path, dtype=dtype
    ).to(device)
    if trainable:
        model = get_peft_model(base, LoraConfig(
            r=int(section["rank"]),
            lora_alpha=int(section["alpha"]),
            target_modules="all-linear",
            lora_dropout=float(section["dropout"]),
            bias="none",
        ))
    else:
        model = PeftModel.from_pretrained(base, ROOT / args.adapter_dir).to(device)
    return model, device


def _train(args: argparse.Namespace, cfg: PilotConfig, protocol: dict) -> None:
    adapter_dir = ROOT / args.adapter_dir
    output = ROOT / args.output
    if (adapter_dir.exists() or output.exists()) and not args.force:
        raise FileExistsError("最终adapter或summary已存在")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    entities = _entities(mappings, MARKET_ORDER)
    if args.smoke:
        entities = [
            next(entity for entity in entities if entity[0] == market)
            for market in MARKET_ORDER
        ]
    train, _ = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.FINAL_TRAIN,
        validation_stage=AnnualStage.OUTER_VALIDATION,
        read_end_utc=FINAL_READ_END_UTC,
    )
    section = protocol["timesfm25"]
    model, device = _load_base_and_lora(args, section, trainable=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    epochs = 1 if args.smoke else int(section["fixed_epochs"][args.task])
    samples_per_market = 2 if args.smoke else int(section["samples_per_market"])
    batch_size = 2 if args.smoke else int(section["batch_size"])
    sampler = MarketNodeBalancedSampler(
        train, samples_per_market=samples_per_market, seed=args.seed
    )
    loader = DataLoader(
        RawTargetWindowDataset(train, args.task, seed=args.seed),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(loader))
    )
    started = perf_counter()
    history: list[dict] = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for context, target in loader:
            context, target = _flatten_batch(context, target)
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _mean_prediction(model, context, HORIZONS[args.task])
            loss = task_loss(prediction, target, args.task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record), flush=True)
    model.save_pretrained(adapter_dir)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    summary = {
        "protocol": "all_market_timesfm25_lora_final_train",
        "model_id": MODEL_ID,
        "model_source": str(args.model_path),
        "task": args.task,
        "phase": "final",
        "seed": args.seed,
        "training_markets": list(MARKET_ORDER),
        "training_years": [2020, 2021, 2022, 2023, 2024],
        "locked_2025_read": False,
        "read_end_utc_exclusive_by_market": FINAL_READ_END_UTC,
        "selection_allowed": False,
        "fixed_epochs": epochs,
        "fixed_epoch_source": section["epoch_source"],
        "history": history,
        "samples_per_market_per_epoch": samples_per_market,
        "input_features": ["raw_target_price_history"],
        "uses_node_or_market_id_for_prediction": False,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "adapter_dir": adapter_dir.relative_to(ROOT).as_posix(),
        "adapter_sha256": sha256(adapter_file),
        "protocol_config": Path(args.protocol_config).as_posix(),
        "protocol_config_sha256": sha256(ROOT / args.protocol_config),
        "wall_seconds": float(perf_counter() - started),
        "smoke_only": bool(args.smoke),
    }
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace, cfg: PilotConfig, protocol: dict) -> None:
    adapter_dir = ROOT / args.adapter_dir
    output = ROOT / args.output
    summary_path = adapter_dir / "final_training_summary.json"
    # 云端runner会把训练summary复制进adapter目录，令评价入口可独立审计。
    if not summary_path.exists():
        raise FileNotFoundError(f"adapter缺少final_training_summary.json: {adapter_dir}")
    frozen = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": "all_market_timesfm25_lora_final_train",
        "task": args.task,
        "phase": "final",
        "seed": args.seed,
        "locked_2025_read": False,
        "selection_allowed": False,
    }
    mismatches = {
        key: (frozen.get(key), value)
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError(f"TimesFM-2.5最终adapter元数据不匹配: {mismatches}")
    if sha256(adapter_dir / "adapter_model.safetensors") != frozen["adapter_sha256"]:
        raise ValueError("TimesFM-2.5 adapter哈希不匹配")
    if output.exists() and not args.force:
        raise FileExistsError(f"输出已存在: {output}")
    market = args.market.upper()
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset([market])
    entities = _entities(mappings, [market])
    if args.smoke:
        entities = entities[:1]
    end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        AnnualStage.LOCKED_RETROSPECTIVE_TEST
    ).end
    read_end_utc = (
        pd.Timestamp(end, tz=market_timezone(market)) + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    _, validation = build_global_task_datasets(
        replace(cfg, global_use_critical_event_weighting=True),
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.FINAL_TRAIN,
        validation_stage=AnnualStage.LOCKED_RETROSPECTIVE_TEST,
        read_end_utc=read_end_utc,
    )
    model, device = _load_base_and_lora(args, protocol["timesfm25"], trainable=False)
    inference_started = perf_counter()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_da_values: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    common_timestamps: np.ndarray | None = None
    per_node: dict[str, dict[str, float]] = {}
    for (_, node), node_dataset in zip(validation.entities, validation.datasets):
        prediction, target, dates, target_da = _predict_node(
            model,
            node_dataset,
            args.task,
            device,
            batch_size=int(protocol["timesfm25"]["batch_size"]),
            max_days=2 if args.smoke else 0,
        )
        _, timestamps = _dates_and_timestamps(node_dataset, args.task)
        if args.smoke and timestamps is not None:
            timestamps = timestamps[:2]
        if common_dates is None:
            common_dates, common_timestamps = dates, timestamps
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"节点日期不一致: {node}")
        error = prediction.astype(np.float64) - target.astype(np.float64)
        per_node[node] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(np.mean(error)),
        }
        predictions.append(prediction)
        targets.append(target)
        if target_da is not None:
            target_da_values.append(target_da)
    inference_seconds = float(perf_counter() - inference_started)
    prediction_elements = int(sum(item.size for item in predictions))
    summary = {
        "protocol": "all_market_timesfm25_lora_locked_2025",
        "model_id": MODEL_ID,
        "task": args.task,
        "evaluation_market": market,
        "evaluation_stage": "locked_2025",
        "selection_allowed": False,
        "adapter_dir": adapter_dir.relative_to(ROOT).as_posix(),
        "adapter_sha256": frozen["adapter_sha256"],
        "data_read_end_utc_exclusive": read_end_utc,
        "nodes": len(entities),
        "node_macro_mae": float(np.mean([value["mae"] for value in per_node.values()])),
        "node_macro_rmse": float(np.mean([value["rmse"] for value in per_node.values()])),
        "node_macro_bias": float(np.mean([value["bias"] for value in per_node.values()])),
        "per_node": per_node,
        "inference_seconds": inference_seconds,
        "prediction_elements": prediction_elements,
        "seconds_per_prediction_element": inference_seconds / prediction_elements,
        "smoke_only": bool(args.smoke),
        "final_profit_claim_allowed": False,
    }
    arrays = {
        "nodes": np.asarray([node for _, node in entities]),
        "dates": common_dates,
        "prediction": np.stack(predictions).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if target_da_values:
        arrays["target_da"] = np.stack(target_da_values).astype(np.float32)
    if args.task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(market_timezone(market))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("train", "locked_2025"))
    parser.add_argument("--protocol-config", default="configs/decision_aware/allmarket_final_baselines_v1.yaml")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--market", choices=tuple(MARKET_ORDER))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("TIMESFM25_MODEL_PATH", MODEL_ID),
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("最终TimesFM-2.5基线固定seed 0")
    protocol = _load_protocol(ROOT / args.protocol_config, args.task)
    cfg = PilotConfig.from_yaml(ROOT / args.data_config)
    _validate_data_config(cfg)
    if args.mode == "locked_2025" and not args.market:
        raise ValueError("locked_2025必须指定--market")
    if args.mode == "train" and args.market:
        raise ValueError("train模式不接受--market")
    if args.preflight_only:
        print(json.dumps({
            "event": "allmarket_timesfm25_preflight_ok",
            "mode": args.mode,
            "task": args.task,
            "test_2025": args.mode == "locked_2025",
        }))
        return
    if args.mode == "train":
        _train(args, cfg, protocol)
        summary_path = ROOT / args.output
        adapter_summary = ROOT / args.adapter_dir / "final_training_summary.json"
        adapter_summary.write_bytes(summary_path.read_bytes())
    else:
        _evaluate(args, cfg, protocol)


if __name__ == "__main__":
    main()
