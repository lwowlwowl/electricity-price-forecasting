#!/usr/bin/env python
"""在ERCOT 2023拟合经济价差分解Transformer的跨任务头。

DA与RT-at-DA基础Transformer保持冻结，只训练联合价差头。该拟合范围与
冻结经济价差协调器一致，2024仅用于一次性确认。
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import (  # noqa: E402
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
)
from decision_aware.economic_spread_factorization import (  # noqa: E402
    EconomicSpreadFactorizedTransformer,
    economic_spread_factorization_loss,
    economic_spread_regime,
)
from decision_aware.global_data import (  # noqa: E402
    NodeBalancedSampler,
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_from_checkpoint(checkpoint: dict) -> PilotConfig:
    allowed = {field.name for field in fields(PilotConfig)}
    return PilotConfig(**{
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed
    })


def _load_base_model(
    checkpoint_path: Path,
    model_class,
    expected_task: str,
) -> tuple[torch.nn.Module, dict, PilotConfig]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
    if checkpoint.get("task") != expected_task:
        raise ValueError(
            f"{checkpoint_path}任务为{checkpoint.get('task')!r}，"
            f"期望{expected_task!r}"
        )
    if checkpoint.get("phase") != "inner":
        raise ValueError("第一轮只允许使用inner checkpoint，避免读取2024/2025")
    config = _config_from_checkpoint(checkpoint)
    if expected_task == "rt_at_da" and (
        str(config.rt_at_da_parameterization).lower() != "direct_price"
    ):
        raise ValueError("基础RT-at-DA必须是独立direct_price模型")
    model = model_class(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint, config


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _training_class_statistics(
    dataset, threshold: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    counts = torch.zeros(3, dtype=torch.long)
    counts_by_market: dict[str, torch.Tensor] = {}
    # 直接读取训练Dataset已经冻结的window和底层价格数组，避免为了统计类别
    # 重复构造168小时上下文。这里只读取训练标签。
    for (market, _), node_dataset in zip(dataset.entities, dataset.datasets):
        market_counts = counts_by_market.setdefault(
            market, torch.zeros(3, dtype=torch.long)
        )
        for window in node_dataset.windows:
            positions = window.target_positions
            spread = torch.from_numpy(
                np.asarray(
                    node_dataset.price_da[positions]
                    - node_dataset.price_rt[positions],
                    dtype=np.float32,
                )
            )
            window_counts = torch.bincount(
                economic_spread_regime(spread, threshold), minlength=3
            )
            counts += window_counts
            market_counts += window_counts
    if bool((counts == 0).any()):
        raise ValueError(f"训练集存在空经济状态: {counts.tolist()}")
    if any(bool((value == 0).any()) for value in counts_by_market.values()):
        raise ValueError("至少一个训练市场存在空经济状态")
    # 当前公平v1只有ERCOT；保留按市场平均的写法，使类别权重定义明确且
    # 不依赖节点数。
    market_rates = torch.stack([
        value.float() / value.sum().float()
        for _, value in sorted(counts_by_market.items())
    ])
    balanced_rate = market_rates.mean(dim=0)
    weights = 1.0 / (3.0 * balanced_rate)
    return counts, counts_by_market, weights


def _run_epoch(
    model: EconomicSpreadFactorizedTransformer,
    loader: DataLoader,
    device: torch.device,
    class_weight: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    # 两套基础Transformer完全冻结并保持确定性，只训练联合价差头。
    for base_model in (model.da_model, model.rt_at_da_model):
        base_model.eval()
        if training and any(
            parameter.requires_grad for parameter in base_model.parameters()
        ):
            base_model.decoder.train()
            base_model.price_head.train()
    sums = {"total": 0.0, "regime": 0.0, "excess": 0.0, "reconstruction": 0.0}
    sample_count = 0
    correct = torch.zeros(3, dtype=torch.long)
    support = torch.zeros(3, dtype=torch.long)
    predicted = torch.zeros(3, dtype=torch.long)
    market_loss_sum: dict[int, float] = {}
    market_sample_count: dict[int, int] = {}
    market_correct: dict[int, torch.Tensor] = {}
    market_support: dict[int, torch.Tensor] = {}
    market_predicted: dict[int, torch.Tensor] = {}
    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            scale = torch.sqrt(
                batch["price_da_std"].square()
                + batch["price_rt_std"].square()
            ).clamp_min(1.0e-6)
            loss = economic_spread_factorization_loss(
                output,
                batch["price_da_tgt"],
                batch["price_rt_tgt"],
                threshold=model.factorization_head.threshold,
                price_scale=scale,
                class_weight=class_weight,
            )
            if training:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    grad_clip,
                )
                optimizer.step()
        batch_size = int(batch["price_da_tgt"].shape[0])
        sample_count += batch_size
        sums["total"] += float(loss.total.detach()) * batch_size
        sums["regime"] += float(loss.regime.detach()) * batch_size
        sums["excess"] += float(loss.conditional_excess.detach()) * batch_size
        sums["reconstruction"] += float(loss.reconstruction.detach()) * batch_size
        target = economic_spread_regime(
            batch["price_da_tgt"] - batch["price_rt_tgt"],
            model.factorization_head.threshold,
        ).detach().cpu()
        guess = output["regime_class"].detach().cpu()
        market_ids = batch["market_id"].detach().cpu()
        for state in range(3):
            target_state = target == state
            guess_state = guess == state
            support[state] += target_state.sum()
            predicted[state] += guess_state.sum()
            correct[state] += (target_state & guess_state).sum()
        for market_id_tensor in torch.unique(market_ids):
            market_id = int(market_id_tensor)
            mask_cpu = market_ids == market_id
            mask = mask_cpu.to(device)
            count = int(mask_cpu.sum())
            market_output = {
                key: value[mask]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
                for key, value in output.items()
            }
            market_loss = economic_spread_factorization_loss(
                market_output,
                batch["price_da_tgt"][mask],
                batch["price_rt_tgt"][mask],
                threshold=model.factorization_head.threshold,
                price_scale=scale[mask],
                class_weight=class_weight,
            )
            market_loss_sum[market_id] = market_loss_sum.get(market_id, 0.0) + (
                float(market_loss.total.detach()) * count
            )
            market_sample_count[market_id] = (
                market_sample_count.get(market_id, 0) + count
            )
            if market_id not in market_correct:
                market_correct[market_id] = torch.zeros(3, dtype=torch.long)
                market_support[market_id] = torch.zeros(3, dtype=torch.long)
                market_predicted[market_id] = torch.zeros(3, dtype=torch.long)
            market_target = target[mask_cpu]
            market_guess = guess[mask_cpu]
            for state in range(3):
                target_state = market_target == state
                guess_state = market_guess == state
                market_support[market_id][state] += target_state.sum()
                market_predicted[market_id][state] += guess_state.sum()
                market_correct[market_id][state] += (
                    target_state & guess_state
                ).sum()
    if sample_count == 0:
        raise ValueError("DataLoader没有样本")
    precision = correct.float() / predicted.clamp_min(1).float()
    recall = correct.float() / support.clamp_min(1).float()
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    market_totals = [
        market_loss_sum[market_id] / market_sample_count[market_id]
        for market_id in sorted(market_loss_sum)
    ]
    market_f1 = []
    for market_id in sorted(market_correct):
        market_precision = market_correct[market_id].float() / market_predicted[
            market_id
        ].clamp_min(1).float()
        market_recall = market_correct[market_id].float() / market_support[
            market_id
        ].clamp_min(1).float()
        values = 2.0 * market_precision * market_recall / (
            market_precision + market_recall
        ).clamp_min(1.0e-12)
        market_f1.append(float(values.mean()))
    return {
        **{key: value / sample_count for key, value in sums.items()},
        "macro_f1": float(f1.mean()),
        "neutral_f1": float(f1[0]),
        "da_high_f1": float(f1[1]),
        "rt_high_f1": float(f1[2]),
        "market_macro_total": float(np.mean(market_totals)),
        "market_macro_f1": float(np.mean(market_f1)),
        "samples": sample_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-checkpoint",
        default="data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt",
    )
    parser.add_argument(
        "--rt-at-da-checkpoint",
        default="data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/rt_at_da_inner_seed0/best.pt",
    )
    parser.add_argument(
        "--output",
        default="data/checkpoints/economic_spread_factorization_lomo_ercot_v1/best.pt",
    )
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--scope", choices=("head",), default="head")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    da_path = ROOT / args.da_checkpoint
    rt_path = ROOT / args.rt_at_da_checkpoint
    output_path = ROOT / args.output
    da_model, da_checkpoint, da_config = _load_base_model(
        da_path, DecisionAwareDAForecaster, "da"
    )
    rt_model, rt_checkpoint, rt_config = _load_base_model(
        rt_path, DecisionAwareRTAtDAForecaster, "rt_at_da"
    )
    if da_config.d_model != rt_config.d_model:
        raise ValueError("DA与RT-at-DA的d_model必须一致")
    if da_checkpoint.get("market_mapping") != rt_checkpoint.get("market_mapping"):
        raise ValueError("两个基础checkpoint的市场映射不一致")
    if da_checkpoint.get("node_mapping") != rt_checkpoint.get("node_mapping"):
        raise ValueError("两个基础checkpoint的节点映射不一致")
    if tuple(da_config.global_training_markets) != tuple(
        rt_config.global_training_markets
    ):
        raise ValueError("两个基础checkpoint的训练市场不一致")

    torch.manual_seed(int(da_config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(da_config.seed))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    backbone_markets = tuple(da_config.global_training_markets)
    adaptation_market = "ERCOT"
    if adaptation_market in backbone_markets:
        raise ValueError("基础Transformer必须是留出ERCOT训练的无ID模型")
    mappings = load_formal_entity_mappings(
        ROOT / "configs" / "nodes.yaml"
    ).subset([adaptation_market])
    all_entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    if args.smoke:
        entities = []
        entities = all_entities[:1]
    else:
        entities = all_entities
    adaptation_config = replace(
        da_config, global_training_markets=[adaptation_market]
    )
    train_dataset, _ = build_global_task_datasets(
        adaptation_config,
        "da",
        entities,
        mappings,
        train_stage=AnnualStage.INNER_VALIDATION,
        validation_stage=AnnualStage.INNER_VALIDATION,
        read_end_utc="2024-01-01T06:00:00+00:00",
    )
    counts, counts_by_market, weights_cpu = _training_class_statistics(
        train_dataset, args.threshold
    )
    weights = weights_cpu.to(device)

    if args.smoke:
        train_sampler = NodeBalancedSampler(
            train_dataset, samples_per_node=4, seed=da_config.seed
        )
        epochs = 1
    else:
        train_sampler = NodeBalancedSampler(train_dataset, seed=da_config.seed)
        epochs = int(args.epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=da_config.batch_size,
        sampler=train_sampler,
        num_workers=da_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = EconomicSpreadFactorizedTransformer(
        da_model,
        rt_model,
        d_model=da_config.d_model,
        threshold=args.threshold,
        dropout=da_config.dropout,
    ).to(device)
    model.set_trainable_scope(args.scope)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=da_config.lr,
        weight_decay=da_config.weight_decay,
    )
    history = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            weights,
            optimizer=optimizer,
            grad_clip=da_config.grad_clip,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    torch.save({
        "method": "economic_spread_factorization_transformer_v1",
        "phase": "inner",
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "threshold_usd_per_mwh": float(args.threshold),
        "trainable_scope": args.scope,
        "class_counts": counts.tolist(),
        "class_counts_by_market": {
            market: value.tolist()
            for market, value in sorted(counts_by_market.items())
        },
        "class_weights": weights.detach().cpu().tolist(),
        "training_markets": [adaptation_market],
        "backbone_training_markets": list(backbone_markets),
        "adaptation_market": adaptation_market,
        "participating_entities": [list(entity) for entity in entities],
        "da_checkpoint": {
            "path": args.da_checkpoint,
            "sha256": _sha256(da_path),
        },
        "rt_at_da_checkpoint": {
            "path": args.rt_at_da_checkpoint,
            "sha256": _sha256(rt_path),
        },
        "data_boundary": {
            "head_fit": "ERCOT 2023",
            "outer_confirmation": "ERCOT 2024 unread during fitting",
            "backbone_train": "CAISO/NYISO/PJM 2020-2022",
            "year_2024_used_for_selection": False,
            "year_2025_used": False,
        },
        "config": da_checkpoint["config"],
    }, output_path)
    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "device": str(device),
        "epochs": epochs,
        "selection_rule": "fixed final epoch; 2024 not read",
        "elapsed_seconds": perf_counter() - started,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "class_counts": counts.tolist(),
        "class_counts_by_market": {
            market: value.tolist()
            for market, value in sorted(counts_by_market.items())
        },
        "class_weights": weights.detach().cpu().tolist(),
        "adaptation_market": adaptation_market,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

