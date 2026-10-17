#!/usr/bin/env python
"""Source-market supervised TimesFM-2.5 LoRA under the frozen LOMO contract.

Each fold trains only on the other three markets (2020--2022), selects the
adapter on those source markets' 2023 windows, and evaluates the untouched
holdout market in 2023.  The official Transformers path is deliberately
univariate: only the raw 168-hour history of the task's target price is used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MARKET_ORDER,
    MarketNodeBalancedSampler,
    MultiNodeDataset,
    RTDeliveryDayDataset,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.loader_v2 import market_timezone  # noqa: E402
from scripts.decision_aware.preflight_timesfm25_lora import (  # noqa: E402
    MODEL_ID,
    validate_protocol,
)


TASKS = ("da", "rt_at_da", "rt")
HORIZONS = {"da": 24, "rt_at_da": 24, "rt": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_loss(prediction: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    error = prediction.float() - target.float()
    if task == "rt_at_da":
        delta = 1.0
        return (torch.sqrt(error.square() + delta * delta) - delta).mean()
    return error.abs().mean()


def _raw_window(node_dataset: Any, task: str, local_index: int) -> tuple[np.ndarray, np.ndarray]:
    horizon = HORIZONS[task]
    if task == "rt":
        if not isinstance(node_dataset, RTDeliveryDayDataset):
            raise TypeError("rolling RT requires RTDeliveryDayDataset")
        _, hourly_indices = node_dataset.days[local_index]
        hourly = node_dataset.hourly_dataset
        contexts: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index in hourly_indices:
            window = hourly.windows[index]
            context = hourly.price_rt[
                int(window.context_start):int(window.issue_position)
            ]
            target = hourly.price_rt[np.asarray(window.target_positions, dtype=np.int64)]
            contexts.append(np.asarray(context, dtype=np.float32))
            targets.append(np.asarray(target, dtype=np.float32))
        context_array = np.stack(contexts)
        target_array = np.stack(targets)
    else:
        window = node_dataset.windows[local_index]
        price = node_dataset.price_da if task == "da" else node_dataset.price_rt
        context_array = np.asarray(
            price[int(window.context_start):int(window.context_end)], dtype=np.float32
        )
        target_array = np.asarray(
            price[np.asarray(window.target_positions, dtype=np.int64)], dtype=np.float32
        )
    if context_array.shape[-1] != 168:
        raise ValueError(f"TimesFM-2.5 context must be 168, got {context_array.shape}")
    if target_array.shape[-1] != horizon:
        raise ValueError(f"{task} horizon must be {horizon}, got {target_array.shape}")
    if not np.isfinite(context_array).all() or not np.isfinite(target_array).all():
        raise ValueError("TimesFM-2.5 raw window contains non-finite values")
    return context_array, target_array


class RawTargetWindowDataset(Dataset):
    """Expose raw target-price windows while preserving the balanced sampler indices."""

    def __init__(self, dataset: MultiNodeDataset, task: str, seed: int = 0) -> None:
        self.dataset = dataset
        self.task = task
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        entity_index, local_index = self.dataset._locate(index)
        context, target = _raw_window(
            self.dataset.datasets[entity_index], self.task, local_index
        )
        if self.task == "rt":
            # A balanced draw is one issue hour, matching the pooled XGBoost
            # budget instead of silently expanding every draw by 24x.
            hour = (int(index) * 17 + self.seed * 13) % 24
            context = context[hour]
            target = target[hour]
        return torch.from_numpy(context), torch.from_numpy(target)


def _flatten_batch(
    context: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim == 3:
        context = context.reshape(-1, context.shape[-1])
        target = target.reshape(-1, target.shape[-1])
    if context.ndim != 2 or target.ndim != 2:
        raise ValueError(f"unexpected batch shapes {tuple(context.shape)}, {tuple(target.shape)}")
    return context, target


def _mean_prediction(model: Any, context: torch.Tensor, horizon: int) -> torch.Tensor:
    # TimesFM-2.5 tokenizes the context in non-overlapping 32-hour patches.
    # Keep the experiment's 168 observed hours, but ask the official wrapper to
    # left-pad them to 192 (= 6 patches).  Its padding mask ensures that the
    # added 24 zeros are not treated as observed history.
    output = model(
        past_values=context,
        forecast_context_len=192,
        truncate_negative=False,
        force_flip_invariance=True,
        return_dict=True,
    )
    prediction = output.mean_predictions[..., :horizon]
    if prediction.shape[-1] != horizon:
        raise ValueError("TimesFM-2.5 returned the wrong horizon")
    return prediction


def _source_validation_loss(
    model: Any,
    dataset: MultiNodeDataset,
    task: str,
    device: torch.device,
    windows_per_node: int,
) -> float:
    model.eval()
    per_node: list[float] = []
    with torch.no_grad():
        for node_dataset in dataset.datasets:
            indices = range(max(0, len(node_dataset) - windows_per_node), len(node_dataset))
            losses: list[float] = []
            for index in indices:
                context, target = _raw_window(node_dataset, task, index)
                context_t = torch.from_numpy(context).to(device)
                target_t = torch.from_numpy(target).to(device)
                if context_t.ndim == 1:
                    context_t = context_t.unsqueeze(0)
                    target_t = target_t.unsqueeze(0)
                prediction = _mean_prediction(model, context_t, HORIZONS[task])
                losses.append(float(task_loss(prediction, target_t, task).item()))
            per_node.append(float(np.mean(losses)))
    if not per_node:
        raise ValueError("source validation is empty")
    return float(np.mean(per_node))


def _dates(node_dataset: Any, task: str, count: int) -> np.ndarray:
    if task == "rt":
        return np.asarray([str(day) for day, _ in node_dataset.days[:count]])
    return np.asarray([
        str(node_dataset.sample_time_contract(index)["delivery_date"])
        for index in range(count)
    ])


def _predict_node(
    model: Any,
    node_dataset: Any,
    task: str,
    device: torch.device,
    batch_size: int,
    max_days: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    count = min(len(node_dataset), max_days) if max_days else len(node_dataset)
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_da: list[np.ndarray] = []
    for index in range(count):
        context, target = _raw_window(node_dataset, task, index)
        contexts.append(context)
        targets.append(target)
        if task == "rt_at_da":
            window = node_dataset.windows[index]
            target_da.append(np.asarray(
                node_dataset.price_da[np.asarray(window.target_positions, dtype=np.int64)],
                dtype=np.float32,
            ))
    context_array = np.stack(contexts)
    target_array = np.stack(targets)
    flat_context = context_array.reshape(-1, context_array.shape[-1])
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_context), batch_size):
            context_t = torch.from_numpy(flat_context[start:start + batch_size]).to(device)
            prediction = _mean_prediction(model, context_t, HORIZONS[task])
            predictions.append(prediction.float().cpu().numpy())
    prediction_array = np.concatenate(predictions).reshape(target_array.shape)
    return (
        prediction_array.astype(np.float32),
        target_array.astype(np.float32),
        _dates(node_dataset, task, count),
        np.stack(target_da).astype(np.float32) if target_da else None,
    )


def _entities(mappings: Any, markets: Iterable[str]) -> list[tuple[str, str]]:
    market_set = set(markets)
    return [
        (mappings.node_to_market[node], node)
        for node in mappings.node_to_id
        if mappings.node_to_market[node] in market_set
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", default="configs/decision_aware/timesfm25_lora_source_lomo_v1.yaml")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--holdout-market", required=True, choices=tuple(MARKET_ORDER))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("TIMESFM25_MODEL_PATH", MODEL_ID),
        help="Official HF model id or a verified local snapshot directory",
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol_path = ROOT / args.protocol_config
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    audit = validate_protocol(protocol)
    data_cfg = PilotConfig.from_yaml(ROOT / args.data_config)
    holdout = args.holdout_market
    training_markets = tuple(str(item).upper() for item in data_cfg.global_training_markets)
    expected_sources = tuple(market for market in MARKET_ORDER if market != holdout)
    if set(training_markets) != set(expected_sources) or holdout in training_markets:
        raise ValueError("data config does not implement the requested LOMO fold")
    if data_cfg.context_len != 168 or data_cfg.global_use_entity_embeddings:
        raise ValueError("TimesFM-2.5 requires 168 hours and identity-free data contract")

    preflight = {
        **audit,
        "task": args.task,
        "holdout_market": holdout,
        "training_markets": list(training_markets),
        "seed": args.seed,
        "uses_holdout_labels_for_training": False,
        "uses_node_or_market_id_for_prediction": False,
        "uses_future_covariates": False,
        "test_2025": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    output = ROOT / args.output
    adapter_dir = ROOT / args.adapter_dir
    if (output.exists() or adapter_dir.exists()) and not args.force:
        raise FileExistsError("output or adapter already exists; pass --force to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    full_mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    source_mappings = full_mappings.subset(training_markets)
    source_entities = _entities(source_mappings, training_markets)
    if args.smoke:
        source_entities = [
            next(entity for entity in source_entities if entity[0] == market)
            for market in training_markets
        ]
    source_train, source_validation = build_global_task_datasets(
        data_cfg,
        args.task,
        source_entities,
        source_mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
        read_end_utc="2024-01-02T08:00:00+00:00",
    )

    holdout_mappings = full_mappings.subset([holdout])
    holdout_entities = _entities(holdout_mappings, [holdout])
    if args.smoke:
        holdout_entities = holdout_entities[:1]
    evaluation_end = FORMAL_EXPERIMENT_CONTRACT.annual_split.range_for(
        AnnualStage.INNER_VALIDATION
    ).end
    read_end_utc = (
        pd.Timestamp(evaluation_end, tz=market_timezone(holdout))
        + pd.Timedelta(days=1)
    ).tz_convert("UTC").isoformat()
    _, holdout_validation = build_global_task_datasets(
        replace(data_cfg, global_use_critical_event_weighting=True),
        args.task,
        holdout_entities,
        holdout_mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
        read_end_utc=read_end_utc,
    )

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_path, dtype=dtype
    ).to(device)
    adapter_cfg = protocol["adapter"]
    model = get_peft_model(model, LoraConfig(
        r=int(adapter_cfg["rank"]),
        lora_alpha=int(adapter_cfg["alpha"]),
        target_modules=adapter_cfg["target_modules"],
        lora_dropout=float(adapter_cfg["dropout"]),
        bias=str(adapter_cfg["bias"]),
    ))
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    training = protocol["training"]
    batch_size = 2 if args.smoke else int(training["batch_size"])
    epochs = 1 if args.smoke else int(training["max_epochs"])
    samples_per_market = 2 if args.smoke else int(training.get("samples_per_market", 2048))
    validation_windows = 1 if args.smoke else int(training.get("validation_windows_per_node", 2))
    sampler = MarketNodeBalancedSampler(
        source_train, samples_per_market=samples_per_market, seed=args.seed
    )
    loader = DataLoader(
        RawTargetWindowDataset(source_train, args.task, seed=args.seed),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(loader))
    )
    started = perf_counter()
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = _source_validation_loss(
            model, source_validation, args.task, device, validation_windows
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "source_node_macro_validation_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            model.save_pretrained(adapter_dir)
        else:
            stale += 1
        if not args.smoke and stale >= int(training["early_stopping_patience"]):
            break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_path, dtype=dtype
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_da_values: list[np.ndarray] = []
    common_dates: np.ndarray | None = None
    per_node: dict[str, dict[str, float]] = {}
    nodes = [node for _, node in holdout_validation.entities]
    for (_, node), node_dataset in zip(
        holdout_validation.entities, holdout_validation.datasets
    ):
        prediction, target, dates, target_da = _predict_node(
            model,
            node_dataset,
            args.task,
            device,
            batch_size=max(1, batch_size),
            max_days=2 if args.smoke else 0,
        )
        if common_dates is None:
            common_dates = dates
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"holdout node dates differ at {node}")
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
        print(json.dumps({"event": "holdout_node_complete", "node": node, **per_node[node]}), flush=True)

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    summary = {
        **preflight,
        "protocol": "timesfm25_source_market_supervised_lora_target_market_zero_shot",
        "model_id": MODEL_ID,
        "model_source": str(args.model_path),
        "input_features": ["raw_target_price_history"],
        "context_hours": 168,
        "horizon": HORIZONS[args.task],
        "source_train_years": [2020, 2021, 2022],
        "source_validation_year": 2023,
        "holdout_test_year": 2023,
        "selection_uses_holdout": False,
        "loss": "pseudo_huber_delta_1" if args.task == "rt_at_da" else "mae",
        "samples_per_source_market_per_epoch": samples_per_market,
        "rt_sampling_unit": (
            "one_deterministic_issue_hour_per_balanced_draw"
            if args.task == "rt" else "one_delivery_day_window"
        ),
        "batch_size": batch_size,
        "epochs_completed": len(history),
        "best_source_node_macro_validation_loss": best_val,
        "history": history,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "adapter_dir": adapter_dir.relative_to(ROOT).as_posix(),
        "adapter_sha256": sha256(adapter_file),
        "protocol_config_sha256": sha256(protocol_path),
        "data_config": Path(args.data_config).as_posix(),
        "data_read_end_utc_exclusive": read_end_utc,
        "nodes": len(nodes),
        "days": int(prediction_array.shape[1]),
        "node_macro_mae": float(np.mean([value["mae"] for value in per_node.values()])),
        "node_macro_rmse": float(np.mean([value["rmse"] for value in per_node.values()])),
        "node_macro_bias": float(np.mean([value["bias"] for value in per_node.values()])),
        "per_node": per_node,
        "wall_seconds": float(perf_counter() - started),
        "smoke_only": bool(args.smoke),
        "final_profit_claim_allowed": False,
    }
    arrays: dict[str, Any] = {
        "nodes": np.asarray(nodes),
        "dates": common_dates,
        "prediction": prediction_array.astype(np.float32),
        "target": target_array.astype(np.float32),
        "metadata": json.dumps(summary, ensure_ascii=False, sort_keys=True),
    }
    if target_da_values:
        arrays["target_da"] = np.stack(target_da_values).astype(np.float32)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
