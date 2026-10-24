#!/usr/bin/env python
"""Source-node screening smoke for proper tail post-training of Chronos-2.

This script is deliberately narrower than a paper experiment.  It compares
frozen Chronos-2, ordinary output-head LoRA, and a proposed fixed-threshold
proper-score LoRA on one real CAISO node.  Training uses 2020--2022 and the
screen uses 2023.  No ERCOT or 2025 data are read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.tail_proper_scores import (  # noqa: E402
    TailChainingSpec,
    chained_price,
    chained_quantile_score,
    daily_extreme_shape_loss,
    daily_hurdle_tail_loss,
    event_brier_score,
    event_probability_from_quantiles,
    nested_tail_price,
    nested_tail_quantile_score,
    quantile_score,
)


MODEL_SNAPSHOT = (
    ROOT
    / "hf_cache"
    / "hub"
    / "models--amazon--chronos-2"
    / "snapshots"
    / "29ec3766d36d6f73f0696f85560a422f50e8498c"
)
DEFAULT_CONFIG = (
    "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml"
)


def _configure_chronos_imports() -> None:
    source = ROOT / "external" / "chronos-forecasting" / "src"
    site_packages = (
        ROOT
        / "external"
        / "chronos-forecasting"
        / ".venv"
        / "Lib"
        / "site-packages"
    )
    if not source.exists() or not site_packages.exists():
        raise FileNotFoundError("local Chronos-2 source or dependency environment is missing")
    # torch is imported before this function, so the project CUDA build remains active.
    sys.path.insert(0, str(site_packages))
    sys.path.insert(0, str(source))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.up(self.down(values)) * self.scale

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "down.weight": self.down.weight.detach().cpu().clone(),
            "up.weight": self.up.weight.detach().cpu().clone(),
        }


def _delivery_offset(window: Any, dataset: Any) -> int:
    """Return the target start offset from Chronos' first forecast hour."""
    target_positions = np.asarray(window.target_positions, dtype=np.int64)
    if target_positions.shape != (24,):
        raise ValueError(f"unexpected target positions shape {target_positions.shape}")
    offset = int(target_positions[0]) - int(window.context_end)
    if offset < 0:
        raise ValueError("delivery target starts before the Chronos forecast origin")
    forecast_start = dataset.index[int(window.context_end)]
    target_start = dataset.index[int(target_positions[0])]
    expected = pd.Timedelta(hours=offset)
    if target_start - forecast_start != expected:
        raise ValueError(
            "row offset does not match the timestamp gap between forecast origin "
            "and delivery target"
        )
    return offset


def _dataset_arrays(
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    offsets: list[int] = []
    calendar_contexts: list[np.ndarray] = []
    calendar_futures: list[np.ndarray] = []
    calendar_values = np.asarray(dataset.streams["cal"], dtype=np.float32)
    if calendar_values.ndim != 2 or calendar_values.shape[1] != 8:
        raise ValueError(f"unexpected calendar shape {calendar_values.shape}")
    for window in dataset.windows:
        context = dataset.price_da[int(window.context_start) : int(window.context_end)]
        positions = np.asarray(window.target_positions, dtype=np.int64)
        target = dataset.price_da[positions]
        if context.shape != (168,) or target.shape != (24,):
            raise ValueError(f"unexpected DA window shapes {context.shape}, {target.shape}")
        contexts.append(np.asarray(context, dtype=np.float32))
        targets.append(np.asarray(target, dtype=np.float32))
        offset = _delivery_offset(window, dataset)
        offsets.append(offset)
        required_hours = math.ceil((offset + 24) / 16) * 16
        future_end = int(window.context_end) + required_hours
        if future_end > len(dataset.index):
            raise ValueError("calendar future does not cover the delivery horizon")
        calendar_contexts.append(
            calendar_values[int(window.context_start) : int(window.context_end)].T
        )
        calendar_futures.append(calendar_values[int(window.context_end) : future_end].T)
    offset_array = np.asarray(offsets, dtype=np.int64)
    first = dataset.windows[0]
    last = dataset.windows[-1]
    contract = {
        "forecast_origin_definition": "first hour not included in context",
        "first_issue_time": str(first.issue_time_utc),
        "first_forecast_start": str(dataset.index[int(first.context_end)]),
        "first_target_start": str(dataset.index[int(first.target_positions[0])]),
        "last_issue_time": str(last.issue_time_utc),
        "delivery_offset_hours_unique": sorted(
            int(value) for value in np.unique(offset_array)
        ),
    }
    return (
        np.stack(contexts),
        np.stack(targets),
        offset_array,
        np.stack(calendar_contexts),
        np.stack(calendar_futures),
        contract,
    )


def load_source_node(
    config_path: Path,
    market: str,
    node: str | None,
    *,
    all_market_nodes: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    str,
]:
    cfg = PilotConfig.from_yaml(config_path)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml")
    nodes = sorted(
        name for name, mapped_market in mappings.node_to_market.items()
        if mapped_market == market
    )
    if not nodes:
        raise ValueError(f"no configured nodes for {market}")
    if node is not None and all_market_nodes:
        raise ValueError("--node and --all-market-nodes are mutually exclusive")
    selected_nodes = (
        nodes
        if all_market_nodes
        else [item.strip() for item in node.split(",") if item.strip()]
        if node is not None
        else [nodes[0]]
    )
    if any(selected not in nodes for selected in selected_nodes):
        raise ValueError(f"one or more selected nodes are not configured for {market}")
    train, validation = build_global_task_datasets(
        cfg,
        "da",
        [(market, selected) for selected in selected_nodes],
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    train_parts = [_dataset_arrays(dataset) for dataset in train.datasets]
    validation_parts = [_dataset_arrays(dataset) for dataset in validation.datasets]

    def stack_parts(parts: list[tuple[Any, ...]], index: int) -> np.ndarray:
        return np.concatenate([part[index] for part in parts], axis=0)

    train_context = stack_parts(train_parts, 0)
    train_target = stack_parts(train_parts, 1)
    train_offsets = stack_parts(train_parts, 2)
    train_calendar_context = stack_parts(train_parts, 3)
    train_calendar_future = stack_parts(train_parts, 4)
    val_context = stack_parts(validation_parts, 0)
    val_target = stack_parts(validation_parts, 1)
    val_offsets = stack_parts(validation_parts, 2)
    val_calendar_context = stack_parts(validation_parts, 3)
    val_calendar_future = stack_parts(validation_parts, 4)
    contracts = {
        selected: {
            "train": train_part[5],
            "screen": validation_part[5],
        }
        for selected, train_part, validation_part in zip(
            selected_nodes, train_parts, validation_parts
        )
    }
    return (
        train_context,
        train_target,
        train_offsets,
        train_calendar_context,
        train_calendar_future,
        val_context,
        val_target,
        val_offsets,
        val_calendar_context,
        val_calendar_future,
        contracts,
        ",".join(selected_nodes),
    )


def _required_output_patches(offsets: np.ndarray, output_patch_size: int) -> int:
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("offsets must be a non-empty vector")
    required_hours = int(np.max(offsets)) + 24
    return math.ceil(required_hours / int(output_patch_size))


def _slice_delivery_predictions(
    predictions: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Select each sample's delivery-day hours from a direct forecast."""
    if predictions.ndim != 3 or offsets.ndim != 1:
        raise ValueError("predictions must be [batch, quantile, horizon] and offsets [batch]")
    if predictions.shape[0] != offsets.shape[0]:
        raise ValueError("prediction batch and offset batch differ")
    hours = offsets[:, None] + torch.arange(24, device=predictions.device)[None, :]
    if int(hours.max()) >= predictions.shape[-1]:
        raise ValueError("forecast horizon is too short for the delivery-day slice")
    return torch.gather(
        predictions,
        dim=2,
        index=hours[:, None, :].expand(-1, predictions.shape[1], -1),
    )


def _build_grouped_calendar_batch(
    contexts: torch.Tensor,
    calendar_context: torch.Tensor,
    calendar_future: torch.Tensor,
    *,
    offsets: torch.Tensor,
    required_hours: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Chronos-2's target-plus-known-covariates grouped input."""
    batch_size = contexts.shape[0]
    num_cal = calendar_context.shape[1]
    if calendar_context.shape[0] != batch_size or calendar_future.shape[0] != batch_size:
        raise ValueError("calendar and target batch sizes differ")
    if calendar_context.shape[-1] != contexts.shape[-1]:
        raise ValueError("calendar and target context lengths differ")
    if calendar_future.shape[-1] < required_hours:
        raise ValueError("calendar future is shorter than the direct forecast")
    nan_target = torch.full(
        (batch_size, 1, required_hours),
        float("nan"),
        dtype=contexts.dtype,
        device=contexts.device,
    )
    future = torch.cat(
        [nan_target, calendar_future[:, :, :required_hours]], dim=1
    )
    grouped_context = torch.cat([contexts[:, None, :], calendar_context], dim=1)
    grouped_context = grouped_context.reshape(batch_size * (num_cal + 1), -1)
    future = future.reshape(batch_size * (num_cal + 1), required_hours)
    group_ids = torch.arange(batch_size, device=contexts.device).repeat_interleave(
        num_cal + 1
    )
    return grouped_context, future, group_ids


def _select_target_group_rows(values: torch.Tensor, num_calendar: int) -> torch.Tensor:
    return values.reshape(-1, num_calendar + 1, *values.shape[1:])[:, 0]


def _replace_with_lora(
    model: nn.Module, rank: int, alpha: float, *, scope: str
) -> list[nn.Parameter]:
    if scope not in {"output", "official"}:
        raise ValueError("LoRA scope must be output or official")
    block = model.output_patch_embedding
    if not isinstance(block.output_layer, nn.Linear):
        raise TypeError("Chronos-2 output layer is not a Linear module")
    if not isinstance(block.residual_layer, nn.Linear):
        raise TypeError("Chronos-2 residual output layer is not a Linear module")
    block.output_layer = LoRALinear(block.output_layer, rank, alpha).to(model.device)
    if scope == "output":
        block.residual_layer = LoRALinear(block.residual_layer, rank, alpha).to(
            model.device
        )
    else:
        targets = ("self_attention.q", "self_attention.k", "self_attention.v", "self_attention.o")
        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear) or not name.endswith(targets):
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            parent._modules[child_name] = LoRALinear(module, rank, alpha).to(
                model.device
            )
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (".down." in name or ".up." in name) and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("LoRA adapters have no trainable parameters")
    return parameters


def _remove_lora_adapters(model: nn.Module) -> None:
    for name, module in reversed(list(model.named_modules())):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        parent._modules[child_name] = module.base


def _restore_output_block(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    from chronos.chronos2.layers import ResidualBlock

    config = model.config
    block = ResidualBlock(
        in_dim=config.d_model,
        h_dim=config.d_ff,
        out_dim=len(model.chronos_config.quantiles)
        * model.chronos_config.output_patch_size,
        act_fn_name=config.dense_act_fn,
        dropout_p=config.dropout_rate,
        use_layer_norm=False,
    ).to(model.device)
    block.load_state_dict(deepcopy(state))
    model.output_patch_embedding = block
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def predict_quantiles(
    model: nn.Module,
    contexts: np.ndarray,
    offsets: np.ndarray,
    calendar_contexts: np.ndarray | None = None,
    calendar_futures: np.ndarray | None = None,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    num_output_patches = _required_output_patches(
        offsets, model.chronos_config.output_patch_size
    )
    required_hours = num_output_patches * model.chronos_config.output_patch_size
    outputs: list[np.ndarray] = []
    for start in range(0, len(contexts), batch_size):
        context = torch.from_numpy(contexts[start : start + batch_size]).to(device)
        batch_offsets = torch.from_numpy(offsets[start : start + batch_size]).to(device)
        if calendar_contexts is None or calendar_futures is None:
            result = model(context=context, num_output_patches=num_output_patches)
            raw_predictions = result.quantile_preds
        else:
            cal_context = torch.from_numpy(
                calendar_contexts[start : start + batch_size]
            ).to(device)
            cal_future = torch.from_numpy(
                calendar_futures[start : start + batch_size]
            ).to(device)
            grouped_context, future_covariates, group_ids = _build_grouped_calendar_batch(
                context,
                cal_context,
                cal_future,
                offsets=batch_offsets,
                required_hours=required_hours,
            )
            result = model(
                context=grouped_context,
                future_covariates=future_covariates,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            )
            raw_predictions = _select_target_group_rows(result.quantile_preds, cal_context.shape[1])
        delivery = _slice_delivery_predictions(raw_predictions, batch_offsets)
        outputs.append(delivery.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def evaluate(
    predictions: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
    *,
    positive_threshold: float,
    event_temperature: float,
) -> dict[str, Any]:
    pred = torch.from_numpy(predictions)
    truth = torch.from_numpy(target)
    levels = torch.from_numpy(quantiles)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    point = predictions[:, median_index, :]
    ordinary = (target >= 0.0) & (target <= positive_threshold)
    positive = target > positive_threshold
    negative = target < 0.0
    abs_error = np.abs(point - target)

    positive_pred = point > positive_threshold
    negative_pred = point < 0.0
    pos_tp = int(np.sum(positive_pred & positive))
    pos_fp = int(np.sum(positive_pred & ~positive))
    neg_tp = int(np.sum(negative_pred & negative))
    neg_fp = int(np.sum(negative_pred & ~negative))

    p_positive = event_probability_from_quantiles(
        pred,
        levels,
        positive_threshold,
        side="above",
        temperature=event_temperature,
    ).numpy()
    p_negative = event_probability_from_quantiles(
        pred,
        levels,
        0.0,
        side="below",
        temperature=event_temperature,
    ).numpy()

    true_peak_hour = np.argmax(target, axis=1)
    pred_peak_hour = np.argmax(point, axis=1)
    circular_distance = np.minimum(
        np.abs(true_peak_hour - pred_peak_hour),
        24 - np.abs(true_peak_hour - pred_peak_hour),
    )
    spike_days = np.any(positive, axis=1)
    true_peak = np.max(target, axis=1)
    pred_peak = np.max(point, axis=1)
    peak_ratio = (
        float(np.mean(pred_peak[spike_days]) / np.mean(true_peak[spike_days]))
        if np.any(spike_days) else None
    )
    positive_mae = float(np.mean(abs_error[positive])) if np.any(positive) else None
    negative_mae = float(np.mean(abs_error[negative])) if np.any(negative) else None
    return {
        "samples": int(target.size),
        "days": int(target.shape[0]),
        "mae": float(np.mean(abs_error)),
        "ordinary_mae": float(np.mean(abs_error[ordinary])),
        "mean_pinball": float(quantile_score(pred, truth, levels)),
        "ramp_mae": float(np.mean(np.abs(np.diff(point, axis=1) - np.diff(target, axis=1)))),
        "positive": {
            "threshold": positive_threshold,
            "event_hours": int(np.sum(positive)),
            "recall": _safe_ratio(pos_tp, int(np.sum(positive))),
            "precision": _safe_ratio(pos_tp, pos_tp + pos_fp),
            "conditional_mae": positive_mae,
            "brier": float(np.mean((p_positive - positive.astype(np.float32)) ** 2)),
            "spike_days": int(np.sum(spike_days)),
            "peak_within_one_hour": (
                float(np.mean(circular_distance[spike_days] <= 1))
                if np.any(spike_days) else None
            ),
            "daily_peak_amplitude_ratio": peak_ratio,
        },
        "negative": {
            "event_hours": int(np.sum(negative)),
            "recall": _safe_ratio(neg_tp, int(np.sum(negative))),
            "precision": _safe_ratio(neg_tp, neg_tp + neg_fp),
            "conditional_mae": negative_mae,
            "brier": float(np.mean((p_negative - negative.astype(np.float32)) ** 2)),
        },
    }


def train_adapter(
    model: nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    offsets: np.ndarray,
    calendar_contexts: np.ndarray | None,
    calendar_futures: np.ndarray | None,
    quantiles: torch.Tensor,
    spec: TailChainingSpec,
    nested_positive_thresholds: tuple[float, ...],
    *,
    mode: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    rank: int,
    alpha: float,
    event_temperature: float,
    event_score_weight: float,
    extreme_temperature: float,
    tail_loss_weight: float,
    adapter_scope: str,
    seed: int,
    device: torch.device,
) -> tuple[list[float], dict[str, dict[str, torch.Tensor]], int]:
    if mode not in {
        "ordinary",
        "proper_tail",
        "nested_tail",
        "extreme_shape",
        "hurdle_tail",
    }:
        raise ValueError(
            "mode must be ordinary, proper_tail, extreme_shape or hurdle_tail"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parameters = _replace_with_lora(model, rank, alpha, scope=adapter_scope)
    model.eval()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    generator = np.random.default_rng(seed)
    price_scale = float(np.median(np.abs(targets - np.median(targets))))
    price_scale = max(price_scale, 1.0)
    transformed_targets = chained_price(torch.from_numpy(targets), spec).numpy()
    transformed_scale = float(
        np.median(np.abs(transformed_targets - np.median(transformed_targets)))
    )
    transformed_scale = max(transformed_scale, 1.0)
    nested_targets = nested_tail_price(
        torch.from_numpy(targets),
        nested_positive_thresholds,
        total_positive_slope=spec.positive_slope,
        negative_threshold=spec.negative_threshold,
        negative_slope=spec.negative_slope,
    ).numpy()
    nested_scale = float(np.median(np.abs(nested_targets - np.median(nested_targets))))
    nested_scale = max(nested_scale, 1.0)
    losses: list[float] = []
    num_output_patches = _required_output_patches(
        offsets, model.chronos_config.output_patch_size
    )
    for _ in range(steps):
        indices = generator.integers(0, len(contexts), size=batch_size)
        context = torch.from_numpy(contexts[indices]).to(device)
        target = torch.from_numpy(targets[indices]).to(device)
        batch_offsets = torch.from_numpy(offsets[indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        if calendar_contexts is None or calendar_futures is None:
            direct_predictions = model(
                context=context, num_output_patches=num_output_patches
            ).quantile_preds
        else:
            cal_context = torch.from_numpy(calendar_contexts[indices]).to(device)
            cal_future = torch.from_numpy(calendar_futures[indices]).to(device)
            grouped_context, future_covariates, group_ids = _build_grouped_calendar_batch(
                context,
                cal_context,
                cal_future,
                offsets=batch_offsets,
                required_hours=num_output_patches * model.chronos_config.output_patch_size,
            )
            grouped_predictions = model(
                context=grouped_context,
                future_covariates=future_covariates,
                group_ids=group_ids,
                num_output_patches=num_output_patches,
            ).quantile_preds
            direct_predictions = _select_target_group_rows(
                grouped_predictions, cal_context.shape[1]
            )
        predictions = _slice_delivery_predictions(direct_predictions, batch_offsets)
        base = quantile_score(predictions, target, quantiles) / price_scale
        if mode == "ordinary":
            loss = base
        elif mode == "proper_tail":
            tail = (
                chained_quantile_score(predictions, target, quantiles, spec)
                / transformed_scale
            )
            negative_brier = event_brier_score(
                predictions,
                target,
                quantiles,
                0.0,
                side="below",
                temperature=event_temperature,
            )
            positive_brier = event_brier_score(
                predictions,
                target,
                quantiles,
                spec.positive_threshold,
                side="above",
                temperature=event_temperature,
            )
            loss = 0.5 * base + 0.5 * tail + event_score_weight * (
                negative_brier + positive_brier
            )
        elif mode == "nested_tail":
            tail = nested_tail_quantile_score(
                predictions,
                target,
                quantiles,
                nested_positive_thresholds,
                total_positive_slope=spec.positive_slope,
                negative_threshold=spec.negative_threshold,
                negative_slope=spec.negative_slope,
            ) / nested_scale
            loss = 0.5 * base + 0.5 * tail
        elif mode == "extreme_shape":
            median_index = int(torch.argmin(torch.abs(quantiles - 0.5)))
            point = predictions[:, median_index, :]
            shape = daily_extreme_shape_loss(
                point,
                target,
                positive_threshold=spec.positive_threshold,
                temperature=extreme_temperature,
                amplitude_scale=price_scale,
                event_weight=event_score_weight,
            )
            loss = 0.5 * base + 0.5 * shape
        else:
            hurdle = daily_hurdle_tail_loss(
                predictions,
                target,
                quantiles,
                positive_threshold=spec.positive_threshold,
                probability_temperature=event_temperature,
                extrema_temperature=extreme_temperature,
                amplitude_scale=price_scale,
                event_weight=event_score_weight,
            )
            loss = base + float(tail_loss_weight) * hurdle
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{mode} loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    state = {
        name: module.adapter_state_dict()
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    return losses, state, sum(parameter.numel() for parameter in parameters)


def screen_gate(
    frozen: dict[str, Any], ordinary: dict[str, Any], proposed: dict[str, Any]
) -> dict[str, Any]:
    del frozen
    ordinary_ok = proposed["ordinary_mae"] <= ordinary["ordinary_mae"] * 1.02
    spike_mae_change = (
        ordinary["positive"]["conditional_mae"]
        - proposed["positive"]["conditional_mae"]
    ) / ordinary["positive"]["conditional_mae"]
    spike_mae_improves = spike_mae_change >= 0.005
    ordinary_peak = ordinary["positive"]["daily_peak_amplitude_ratio"]
    proposed_peak = proposed["positive"]["daily_peak_amplitude_ratio"]
    peak_ratio_change = (
        abs(1.0 - ordinary_peak) - abs(1.0 - proposed_peak)
        if ordinary_peak is not None and proposed_peak is not None
        else None
    )
    peak_ratio_improves = (
        ordinary_peak is not None
        and proposed_peak is not None
        and peak_ratio_change >= 0.02
    )
    negative_brier_change = (
        ordinary["negative"]["brier"] - proposed["negative"]["brier"]
    ) / ordinary["negative"]["brier"]
    negative_brier_improves = negative_brier_change >= 0.01
    ramp_ok = proposed["ramp_mae"] <= ordinary["ramp_mae"] * 1.02
    positive_tail_improves = spike_mae_improves or peak_ratio_improves
    return {
        "ordinary_mae_within_2pct": ordinary_ok,
        "positive_tail_improves": positive_tail_improves,
        "positive_conditional_mae_relative_improvement": spike_mae_change,
        "peak_amplitude_ratio_absolute_improvement": peak_ratio_change,
        "negative_brier_improves": negative_brier_improves,
        "negative_brier_relative_improvement": negative_brier_change,
        "ramp_mae_within_2pct": ramp_ok,
        "reference": "ordinary_lora",
        "passed": (
            ordinary_ok
            and positive_tail_improves
            and negative_brier_improves
            and ramp_ok
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--market", default="CAISO")
    parser.add_argument("--node")
    parser.add_argument("--all-market-nodes", action="store_true")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--positive-slope", type=float, default=4.0)
    parser.add_argument("--negative-slope", type=float, default=2.0)
    parser.add_argument("--event-temperature", type=float, default=5.0)
    parser.add_argument("--event-score-weight", type=float, default=0.25)
    parser.add_argument(
        "--candidate",
        choices=(
            "proper_tail",
            "extreme_shape",
            "hurdle_tail",
            "calendar_ordinary",
            "calendar_tail",
            "official_tail",
            "official_nested_tail",
        ),
        default="proper_tail",
    )
    parser.add_argument("--extreme-temperature", type=float, default=10.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="data/results/chronos2_proper_tail_smoke_v1"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this Chronos-2 smoke requires the project CUDA environment")

    started = perf_counter()
    config_path = ROOT / args.config
    (
        train_context,
        train_target,
        train_offsets,
        train_calendar_context,
        train_calendar_future,
        val_context,
        val_target,
        val_offsets,
        val_calendar_context,
        val_calendar_future,
        time_contract,
        node,
    ) = load_source_node(
        config_path,
        args.market,
        args.node,
        all_market_nodes=args.all_market_nodes,
    )
    positive_threshold = float(np.quantile(train_target, 0.99))
    nested_positive_thresholds = tuple(
        float(np.quantile(train_target, level)) for level in (0.90, 0.95, 0.99)
    )
    spec = TailChainingSpec(
        positive_threshold=positive_threshold,
        negative_threshold=0.0,
        positive_slope=args.positive_slope,
        negative_slope=args.negative_slope,
    )

    _configure_chronos_imports()
    from chronos import Chronos2Pipeline

    device = torch.device("cuda")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(MODEL_SNAPSHOT),
        device_map="cuda",
        dtype=torch.float32,
        local_files_only=True,
    )
    model = pipeline.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    quantiles_np = np.asarray(model.chronos_config.quantiles, dtype=np.float32)
    quantiles = torch.from_numpy(quantiles_np).to(device)

    frozen_predictions = predict_quantiles(
        model,
        val_context,
        val_offsets,
        None,
        None,
        batch_size=args.eval_batch_size,
        device=device,
    )
    frozen_metrics = evaluate(
        frozen_predictions,
        val_target,
        quantiles_np,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
    )

    ordinary_losses, ordinary_state, trainable_parameters = train_adapter(
        model,
        train_context,
        train_target,
        train_offsets,
        None,
        None,
        quantiles,
        spec,
        nested_positive_thresholds,
        mode="ordinary",
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        event_temperature=args.event_temperature,
        event_score_weight=args.event_score_weight,
        extreme_temperature=args.extreme_temperature,
        tail_loss_weight=args.tail_loss_weight,
        adapter_scope=(
            "official"
            if args.candidate in {"official_tail", "official_nested_tail"}
            else "output"
        ),
        seed=args.seed,
        device=device,
    )
    ordinary_predictions = predict_quantiles(
        model,
        val_context,
        val_offsets,
        None,
        None,
        batch_size=args.eval_batch_size,
        device=device,
    )
    ordinary_metrics = evaluate(
        ordinary_predictions,
        val_target,
        quantiles_np,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
    )

    _remove_lora_adapters(model)
    proposed_losses, proposed_state, proposed_parameters = train_adapter(
        model,
        train_context,
        train_target,
        train_offsets,
        train_calendar_context if args.candidate.startswith("calendar_") else None,
        train_calendar_future if args.candidate.startswith("calendar_") else None,
        quantiles,
        spec,
        nested_positive_thresholds,
        mode=(
            "ordinary"
            if args.candidate == "calendar_ordinary"
            else "proper_tail"
            if args.candidate in {"calendar_tail", "official_tail"}
            else "nested_tail"
            if args.candidate == "official_nested_tail"
            else args.candidate
        ),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        event_temperature=args.event_temperature,
        event_score_weight=args.event_score_weight,
        extreme_temperature=args.extreme_temperature,
        tail_loss_weight=args.tail_loss_weight,
        adapter_scope=(
            "official"
            if args.candidate in {"official_tail", "official_nested_tail"}
            else "output"
        ),
        seed=args.seed,
        device=device,
    )
    if proposed_parameters != trainable_parameters:
        raise AssertionError("ordinary and proposed adapters differ in parameter count")
    proposed_predictions = predict_quantiles(
        model,
        val_context,
        val_offsets,
        val_calendar_context if args.candidate.startswith("calendar_") else None,
        val_calendar_future if args.candidate.startswith("calendar_") else None,
        batch_size=args.eval_batch_size,
        device=device,
    )
    proposed_metrics = evaluate(
        proposed_predictions,
        val_target,
        quantiles_np,
        positive_threshold=positive_threshold,
        event_temperature=args.event_temperature,
    )
    gate = screen_gate(frozen_metrics, ordinary_metrics, proposed_metrics)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=val_target,
        quantiles=quantiles_np,
        frozen=frozen_predictions,
        ordinary=ordinary_predictions,
        proposed=proposed_predictions,
    )
    adapter_path = output_dir / "adapters.pt"
    torch.save(
        {
            "format": "chronos2_output_head_dual_lora_smoke_v1",
            "ordinary": ordinary_state,
            "proposed": proposed_state,
            "tail_spec": asdict(spec),
        },
        adapter_path,
    )
    summary = {
        "status": "passed_screen" if gate["passed"] else "failed_screen",
        "scope": "single_source_node_screen_not_paper_result",
        "data_contract": {
            "market": args.market,
            "node": node,
            "all_market_nodes": args.all_market_nodes,
            "train_years": "2020-2022",
            "screen_year": 2023,
            "ercot_read": False,
            "year_2025_read": False,
            "train_days": int(len(train_target)),
            "screen_days": int(len(val_target)),
            "chronos_direct_forecast_alignment": time_contract,
            "output_patch_size": int(model.chronos_config.output_patch_size),
            "num_output_patches": _required_output_patches(
                np.concatenate([train_offsets, val_offsets]),
                model.chronos_config.output_patch_size,
            ),
        },
        "model": "amazon/chronos-2",
        "model_snapshot": MODEL_SNAPSHOT.relative_to(ROOT).as_posix(),
        "official_forward_path": True,
        "training_path": "frozen_backbone_output_block_lora",
        "tail_spec": asdict(spec),
        "nested_positive_thresholds": list(nested_positive_thresholds),
        "candidate": args.candidate,
        "objective": {
            "ordinary": "unweighted_quantile_score",
            "proposed": {
                "proper_tail": (
                    "base_quantile_plus_monotone_chained_quantile_plus_unweighted_"
                    "negative_and_positive_event_brier"
                ),
                "extreme_shape": (
                    "base_quantile_plus_daily_smooth_max_min_amplitude_and_event_score"
                ),
                "hurdle_tail": (
                    "base_quantile_plus_daily_event_brier_and_event_conditional_"
                    "positive_negative_amplitude"
                ),
                "calendar_ordinary": (
                    "ordinary_quantile_score_with_Chronos2_native_known_future_calendar"
                ),
                "calendar_tail": (
                    "calendar_conditioning_plus_monotone_chained_quantile_and_event_brier"
                ),
                "official_tail": (
                    "official_attention_and_output_LoRA_with_monotone_chained_"
                    "quantile_and_event_brier"
                ),
                "official_nested_tail": (
                    "official_attention_and_output_LoRA_with_fixed_q90_q95_q99_"
                    "nested_monotone_tail_score"
                ),
            }[args.candidate],
            "sample_reweighting": False,
        },
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "rank": args.rank,
            "alpha": args.alpha,
            "event_temperature": args.event_temperature,
            "event_score_weight": args.event_score_weight,
            "tail_loss_weight": args.tail_loss_weight,
            "seed": args.seed,
        },
        "trainable_parameters": trainable_parameters,
        "adapter_scope": (
            "official_attention_qkvo_plus_output_layer"
            if args.candidate in {"official_tail", "official_nested_tail"}
            else "output_and_residual_layers"
        ),
        "frozen": frozen_metrics,
        "ordinary_lora": ordinary_metrics,
        "proposed_lora": proposed_metrics,
        "loss_trace": {
            "ordinary_first": ordinary_losses[0],
            "ordinary_last": ordinary_losses[-1],
            "proposed_first": proposed_losses[0],
            "proposed_last": proposed_losses[-1],
        },
        "screen_gate": gate,
        "adapter_path": adapter_path.relative_to(ROOT).as_posix(),
        "adapter_sha256": _sha256(adapter_path),
        "torch_version": torch.__version__,
        "device": str(device),
        "wall_seconds": perf_counter() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
