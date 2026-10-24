#!/usr/bin/env python
"""训练一个任务的单市场或跨市场多节点global价格模型。

DA、RT-at-DA、滚动RT必须分别启动本脚本，因此模型、optimizer和checkpoint
天然独立。支持Huber、MAE、MSE和伪Huber；年度边界来自正式合同，不接受命令行任意日期。
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.critical_event_weighting import (  # noqa: E402
    critical_event_weights,
    fit_critical_event_thresholds,
    weighted_elementwise_mean,
)
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    MarketNodeBalancedSampler,
    NodeBalancedSampler,
    build_global_task_datasets,
    flatten_rt_delivery_day_batch,
    global_checkpoint_metadata,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.local_peak_dynamics import (  # noqa: E402
    daily_peak_excursion,
    daily_peak_excursion_loss,
)
from decision_aware.spread_factorization import (  # noqa: E402
    normalized_spread_target,
)


TASKS = {
    "da": {
        "model": DecisionAwareDAForecaster,
        "prediction_raw": "p_da",
        "prediction_normalized": "p_da_normalized",
        "target": "price_da_tgt",
        "mean": "price_da_mean",
        "std": "price_da_std",
    },
    "rt_at_da": {
        "model": DecisionAwareRTAtDAForecaster,
        "prediction_raw": "p_rt_at_da",
        "prediction_normalized": "p_rt_at_da_normalized",
        "target": "price_rt_at_da_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    },
    "rt": {
        "model": DecisionAwareRTForecaster,
        "prediction_raw": "p_rt",
        "prediction_normalized": "p_rt_normalized",
        "target": "price_rt_tgt",
        "mean": "price_rt_mean",
        "std": "price_rt_std",
    },
}


def _forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: PilotConfig,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Global训练的显式稳健损失；所有模式都作用于逐节点标准化价格。"""
    mode = str(getattr(cfg, "global_training_loss", "smooth_l1")).lower()
    delta = float(getattr(cfg, "global_loss_delta", 1.0))
    if delta <= 0:
        raise ValueError("global_loss_delta必须为正")
    if mode == "smooth_l1":
        return F.smooth_l1_loss(
            prediction, target, beta=delta, reduction=reduction
        )
    if mode == "mae":
        return F.l1_loss(prediction, target, reduction=reduction)
    if mode == "mse":
        return F.mse_loss(prediction, target, reduction=reduction)
    if mode == "pseudo_huber":
        elementwise = delta * delta * (
            torch.sqrt(1.0 + ((prediction - target) / delta).square()) - 1.0
        )
        if reduction == "sum":
            return elementwise.sum()
        if reduction == "mean":
            return elementwise.mean()
        if reduction == "none":
            return elementwise
        raise ValueError(f"未知reduction: {reduction}")
    raise ValueError(
        "global_training_loss必须是smooth_l1、mae、mse或pseudo_huber"
    )


def _norm_stats_equal(left, right) -> bool:
    """递归比较训练实体scaler或context-robust只读合同。

    早期实现只支持``node -> stream -> statistic -> array``；无ID模型的
    context-robust元数据则是``node -> contract fields``。两者都必须严格
    比较，但字符串合同不能被当成第三层统计字典索引。
    """
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != set(right):
            return False
        return all(_norm_stats_equal(left[key], right[key]) for key in left)
    if isinstance(left, (str, bytes, bool, type(None))) or isinstance(
        right, (str, bytes, bool, type(None))
    ):
        return left == right
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except (TypeError, ValueError):
        return False
    if left_array.shape != right_array.shape:
        return False
    if np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
        right_array.dtype, np.number
    ):
        return bool(np.allclose(
            left_array.astype(np.float64), right_array.astype(np.float64),
            rtol=1e-7, atol=1e-7,
        ))
    return bool(np.array_equal(left_array, right_array))


def _configure_trainable_scope(
    model: torch.nn.Module, scope: str
) -> tuple[list[torch.nn.Parameter], dict[str, int]]:
    allowed = {
        "all", "decoder_head", "price_head", "scarcity_residual",
        "economic_regime",
    }
    if scope not in allowed:
        raise ValueError(f"global_trainable_scope必须属于{sorted(allowed)}")
    for name, parameter in model.named_parameters():
        if scope == "all":
            trainable = True
        elif scope == "decoder_head":
            trainable = name.startswith("decoder.") or name.startswith("price_head.")
        elif scope == "price_head":
            trainable = name.startswith("price_head.")
        elif scope == "scarcity_residual":
            trainable = name.startswith("scarcity_residual.")
        else:
            trainable = name.startswith("economic_regime_head.")
        parameter.requires_grad_(trainable)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"scope={scope}没有可训练参数")
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in parameters),
    }
    return parameters, counts

PHASES = {
    "inner": (AnnualStage.INNER_TRAIN, AnnualStage.INNER_VALIDATION),
    "outer": (AnnualStage.OUTER_TRAIN, AnnualStage.OUTER_VALIDATION),
    # 最终部署训练使用2020—2024。2024诊断只在固定末轮记录一次，
    # 不参与epoch、学习率、checkpoint或结构选择；训练器绝不构造2025数据集。
    "final": (AnnualStage.FINAL_TRAIN, AnnualStage.OUTER_VALIDATION),
}

FINAL_TRAIN_READ_END_UTC = {
    "CAISO": "2025-01-01T08:00:00+00:00",
    "ERCOT": "2025-01-01T06:00:00+00:00",
    "NYISO": "2025-01-01T05:00:00+00:00",
    "PJM": "2025-01-01T05:00:00+00:00",
}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_data_hash(market: str) -> str:
    audit_path = ROOT / "data" / "results" / "multimarket_global_data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    record = audit["markets"][market]
    # Old audits created on Windows may contain backslashes.  Normalizing here
    # keeps the frozen audit portable to Linux cloud workers.
    source_path = ROOT / Path(str(record["path"]).replace("\\", "/"))
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(
            f"{market}源Parquet哈希与冻结数据审计不一致: "
            f"actual={actual}, expected={expected}"
        )
    return actual


def _prepare_batch(batch: dict, task_name: str, device: torch.device) -> dict:
    if task_name == "rt":
        batch = flatten_rt_delivery_day_batch(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _normalized_target(
    batch: dict,
    task: dict,
    cfg: PilotConfig,
    task_name: str,
) -> torch.Tensor:
    if (
        task_name == "rt_at_da"
        and str(cfg.rt_at_da_parameterization).lower() == "da_minus_spread"
    ):
        return normalized_spread_target(
            batch,
            da_target_key="price_da_tgt_spread_only",
            rt_target_key="price_rt_at_da_tgt",
        )
    return (
        batch[task["target"]] - batch[task["mean"]].reshape(-1, 1)
    ) / batch[task["std"]].reshape(-1, 1)


def _economic_regime_target(batch: dict, cfg: PilotConfig) -> torch.Tensor:
    """0=中性，1=DA>RT可交易，2=RT>DA可交易。"""
    spread = (
        batch["price_da_tgt_spread_only"] - batch["price_rt_at_da_tgt"]
    )
    threshold = float(cfg.resolved_spread_threshold)
    target = torch.zeros_like(spread, dtype=torch.long)
    target = torch.where(spread > threshold, torch.ones_like(target), target)
    target = torch.where(spread < -threshold, torch.full_like(target, 2), target)
    return target


def _elementwise_training_loss(
    output: dict,
    batch: dict,
    task: dict,
    cfg: PilotConfig,
    task_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        task_name == "rt_at_da"
        and str(cfg.rt_at_da_parameterization).lower() == "economic_regime"
    ):
        target = _economic_regime_target(batch, cfg)
        logits = output["economic_regime_logits"]
        if logits.shape != (*target.shape, 3):
            raise ValueError("economic_regime_logits形状必须是[B,H,3]")
        loss = F.cross_entropy(
            logits.reshape(-1, 3), target.reshape(-1), reduction="none"
        ).reshape_as(target)
        return loss, target
    target = _normalized_target(batch, task, cfg, task_name)
    loss = _forecast_loss(
        output[_normalized_prediction_key(task, cfg, task_name)],
        target,
        cfg,
        reduction="none",
    )
    return loss, target


def _normalized_prediction_key(task: dict, cfg: PilotConfig, task_name: str) -> str:
    if (
        task_name == "rt_at_da"
        and str(cfg.rt_at_da_parameterization).lower() == "da_minus_spread"
    ):
        return "p_da_minus_rt_at_da_normalized"
    return str(task["prediction_normalized"])


def _forward_task_model(
    model: torch.nn.Module,
    batch: dict,
    task_name: str,
    upstream_da_model: torch.nn.Module | None,
) -> dict:
    if task_name != "rt_at_da" or upstream_da_model is None:
        return model(batch)
    with torch.no_grad():
        p_da_upstream = upstream_da_model(batch)["p_da"].detach()
    conditioned_batch = dict(batch)
    conditioned_batch["p_da_upstream"] = p_da_upstream
    return model(conditioned_batch)


def _per_market_node_macro_mae(
    abs_sum: torch.Tensor,
    counts: torch.Tensor,
    node_market_ids: torch.Tensor,
    market_names_by_id: list[str],
) -> dict[str, float]:
    """先对节点求MAE，再在各市场内对节点等权平均。"""
    if abs_sum.shape != counts.shape or abs_sum.shape != node_market_ids.shape:
        raise ValueError("节点误差、计数与市场归属形状必须一致")
    result: dict[str, float] = {}
    valid = counts > 0
    node_mae = torch.zeros_like(abs_sum, dtype=torch.float64)
    node_mae[valid] = abs_sum[valid].double() / counts[valid].double()
    for market_id, market_name in enumerate(market_names_by_id):
        mask = valid & (node_market_ids == market_id)
        if bool(mask.any()):
            result[market_name] = float(node_mae[mask].mean())
    if not result:
        raise RuntimeError("验证集没有任何可用市场指标")
    return result


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    task_name: str,
    node_count: int,
    market_count: int,
    node_market_ids: torch.Tensor,
    market_names_by_id: list[str],
    device: torch.device,
    amp_enabled: bool,
    upstream_da_model: torch.nn.Module | None = None,
    max_batches: int | None = None,
) -> dict:
    task = TASKS[task_name]
    abs_sum = torch.zeros(node_count, dtype=torch.float64)
    squared_sum = torch.zeros(node_count, dtype=torch.float64)
    counts = torch.zeros(node_count, dtype=torch.int64)
    market_abs_sum = torch.zeros(market_count, dtype=torch.float64)
    market_counts = torch.zeros(market_count, dtype=torch.int64)
    spread_abs_sum = torch.zeros(node_count, dtype=torch.float64)
    spread_counts = torch.zeros(node_count, dtype=torch.int64)
    spread_direction_correct = torch.zeros(node_count, dtype=torch.float64)
    important_direction_correct = torch.zeros(node_count, dtype=torch.float64)
    important_direction_counts = torch.zeros(node_count, dtype=torch.int64)
    regime_correct = torch.zeros(node_count, dtype=torch.float64)
    regime_counts = torch.zeros(node_count, dtype=torch.int64)
    regime_confusion = torch.zeros(
        market_count, 3, 3, dtype=torch.int64
    )
    peak_excursion_abs_sum = torch.zeros(node_count, dtype=torch.float64)
    peak_excursion_counts = torch.zeros(node_count, dtype=torch.int64)
    huber_sum = 0.0
    normalized_elements = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _prepare_batch(batch, task_name, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = _forward_task_model(
                    model, batch, task_name, upstream_da_model
                )
                elementwise_loss, training_target = _elementwise_training_loss(
                    output, batch, task, model.cfg, task_name
                )
                normalized_loss = elementwise_loss.sum()
            error = (
                output[task["prediction_raw"]] - batch[task["target"]]
            ).detach().float().cpu()
            ids = batch["node_id"].detach().cpu().long()
            market_ids = batch["market_id"].detach().cpu().long()
            elements_per_sample = error[0].numel()
            per_sample_abs = error.abs().reshape(len(error), -1).sum(dim=1).double()
            per_sample_squared = error.square().reshape(len(error), -1).sum(dim=1).double()
            abs_sum.scatter_add_(0, ids, per_sample_abs)
            squared_sum.scatter_add_(0, ids, per_sample_squared)
            counts.scatter_add_(
                0,
                ids,
                torch.full_like(ids, elements_per_sample, dtype=torch.int64),
            )
            market_abs_sum.scatter_add_(0, market_ids, per_sample_abs)
            market_counts.scatter_add_(
                0,
                market_ids,
                torch.full_like(
                    market_ids, elements_per_sample, dtype=torch.int64
                ),
            )
            if task_name == "da":
                predicted_excursion = daily_peak_excursion(
                    output[task["prediction_raw"]]
                ).detach().cpu()
                true_excursion = daily_peak_excursion(
                    batch[task["target"]]
                ).detach().cpu()
                per_sample_peak_abs = (
                    predicted_excursion - true_excursion
                ).abs().double()
                peak_excursion_abs_sum.scatter_add_(
                    0, ids, per_sample_peak_abs
                )
                peak_excursion_counts.scatter_add_(
                    0, ids, torch.ones_like(ids, dtype=torch.int64)
                )
            if "p_da_minus_rt_at_da" in output:
                true_spread = (
                    batch["price_da_tgt_spread_only"]
                    - batch[task["target"]]
                ).detach().float().cpu()
                predicted_spread = output[
                    "p_da_minus_rt_at_da"
                ].detach().float().cpu()
                per_sample_spread_abs = (
                    predicted_spread - true_spread
                ).abs().reshape(len(error), -1).sum(dim=1).double()
                per_sample_direction_correct = (
                    (predicted_spread > 0) == (true_spread > 0)
                ).reshape(len(error), -1).sum(dim=1).double()
                important = (
                    true_spread.abs()
                    >= float(model.cfg.resolved_spread_threshold)
                )
                per_sample_important_correct = (
                    ((predicted_spread > 0) == (true_spread > 0)) & important
                ).reshape(len(error), -1).sum(dim=1).double()
                per_sample_important_counts = important.reshape(
                    len(error), -1
                ).sum(dim=1).long()
                spread_abs_sum.scatter_add_(0, ids, per_sample_spread_abs)
                spread_counts.scatter_add_(
                    0,
                    ids,
                    torch.full_like(ids, elements_per_sample, dtype=torch.int64),
                )
                spread_direction_correct.scatter_add_(
                    0, ids, per_sample_direction_correct
                )
                important_direction_correct.scatter_add_(
                    0, ids, per_sample_important_correct
                )
                important_direction_counts.scatter_add_(
                    0, ids, per_sample_important_counts
                )
            if "economic_regime_class" in output:
                regime_target = _economic_regime_target(batch, model.cfg).cpu()
                regime_prediction = output[
                    "economic_regime_class"
                ].detach().cpu()
                per_sample_regime_correct = (
                    regime_prediction == regime_target
                ).reshape(len(error), -1).sum(dim=1).double()
                regime_correct.scatter_add_(0, ids, per_sample_regime_correct)
                regime_counts.scatter_add_(
                    0,
                    ids,
                    torch.full_like(ids, elements_per_sample, dtype=torch.int64),
                )
                flat_market = market_ids.reshape(-1, 1).expand_as(
                    regime_target
                ).reshape(-1)
                flat_index = (
                    flat_market * 9
                    + regime_target.reshape(-1) * 3
                    + regime_prediction.reshape(-1)
                )
                regime_confusion += torch.bincount(
                    flat_index,
                    minlength=market_count * 9,
                ).reshape(market_count, 3, 3)
            huber_sum += float(normalized_loss.detach().cpu())
            normalized_elements += training_target.numel()
    valid = counts > 0
    if not bool(valid.any()):
        raise RuntimeError("验证集没有产生任何预测")
    node_mae = abs_sum[valid] / counts[valid]
    node_rmse = torch.sqrt(squared_sum[valid] / counts[valid])
    valid_markets = market_counts > 0
    per_market_micro_mae = {
        market_names_by_id[market_id]: float(
            market_abs_sum[market_id] / market_counts[market_id]
        )
        for market_id in range(market_count)
        if bool(valid_markets[market_id])
    }
    per_market_mae = _per_market_node_macro_mae(
        abs_sum, counts, node_market_ids, market_names_by_id
    )
    result = {
        "macro_mae": float(node_mae.mean()),
        "market_macro_mae": float(sum(per_market_mae.values()) / len(per_market_mae)),
        "per_market_node_macro_mae": per_market_mae,
        "worst_market_node_macro_mae": float(max(per_market_mae.values())),
        "per_market_micro_mae": per_market_micro_mae,
        "macro_rmse": float(node_rmse.mean()),
        "micro_mae": float(abs_sum.sum() / counts.sum()),
        "normalized_training_objective": huber_sum / normalized_elements,
        "training_loss": model.cfg.global_training_loss,
        # 兼容已存在的汇总读取器；新结果应读取上面的通用字段。
        "normalized_huber": huber_sum / normalized_elements,
        "evaluated_nodes": int(valid.sum()),
        "evaluated_markets": int(valid_markets.sum()),
        "evaluated_elements": int(counts.sum()),
    }
    peak_valid = peak_excursion_counts > 0
    if bool(peak_valid.any()):
        per_market_peak = _per_market_node_macro_mae(
            peak_excursion_abs_sum,
            peak_excursion_counts,
            node_market_ids,
            market_names_by_id,
        )
        result.update({
            "macro_daily_peak_excursion_mae": float(
                (
                    peak_excursion_abs_sum[peak_valid]
                    / peak_excursion_counts[peak_valid]
                ).mean()
            ),
            "market_macro_daily_peak_excursion_mae": float(
                sum(per_market_peak.values()) / len(per_market_peak)
            ),
            "per_market_node_macro_daily_peak_excursion_mae": per_market_peak,
            "evaluated_daily_peak_excursions": int(
                peak_excursion_counts.sum()
            ),
        })
    spread_valid = spread_counts > 0
    if bool(spread_valid.any()):
        important_valid = important_direction_counts > 0
        result.update({
            "macro_spread_mae": float(
                (spread_abs_sum[spread_valid] / spread_counts[spread_valid]).mean()
            ),
            "macro_spread_direction_accuracy": float(
                (
                    spread_direction_correct[spread_valid]
                    / spread_counts[spread_valid]
                ).mean()
            ),
            "important_spread_abs_threshold": float(
                model.cfg.resolved_spread_threshold
            ),
            "macro_important_spread_direction_accuracy": (
                float(
                    (
                        important_direction_correct[important_valid]
                        / important_direction_counts[important_valid]
                    ).mean()
                )
                if bool(important_valid.any())
                else None
            ),
        })
    regime_valid = regime_counts > 0
    if bool(regime_valid.any()):
        result["macro_economic_regime_accuracy"] = float(
            (regime_correct[regime_valid] / regime_counts[regime_valid]).mean()
        )
        result["economic_regime_confusion_by_market"] = {
            market_names_by_id[market_id]: regime_confusion[
                market_id
            ].tolist()
            for market_id in range(market_count)
            if int(regime_confusion[market_id].sum()) > 0
        }
        result["economic_regime_class_names"] = [
            "neutral",
            "da_above_rt_tradeable",
            "rt_above_da_tradeable",
        ]
    return result


def _validation_selection_score(validation: dict, cfg: PilotConfig) -> float:
    if str(cfg.rt_at_da_parameterization).lower() == "economic_regime":
        return 1.0 - float(validation["macro_economic_regime_accuracy"])
    aggregation = str(
        getattr(cfg, "global_validation_aggregation", "node_macro")
    ).lower()
    if aggregation == "node_macro":
        return float(validation["macro_mae"])
    if aggregation == "market_macro":
        return float(validation["market_macro_mae"])
    raise ValueError(
        "global_validation_aggregation必须是node_macro或market_macro"
    )


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    config: PilotConfig,
    metadata: dict,
    task_name: str,
    phase: str,
    epoch: int,
    best_epoch: int,
    best_macro_mae: float,
    stale_epochs: int,
    history: list[dict],
) -> dict:
    return {
        "task": task_name,
        "phase": phase,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_macro_mae": best_macro_mae,
        "best_validation_score": best_macro_mae,
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state": scaler.state_dict(),
        "stale_epochs": int(stale_epochs),
        "config": config.to_dict(),
        "git_commit": _git_commit(),
        **metadata,
    }


def _rank_candidate_records(history: list[dict], keep_top_k: int) -> list[dict]:
    """按冻结的验证聚合分数返回稳定、可审计的前K个epoch。"""
    if keep_top_k < 1:
        raise ValueError("global_keep_top_k_checkpoints必须至少为1")
    records = []
    for record in history:
        if not isinstance(record.get("validation"), dict):
            continue
        if record["validation"].get("macro_mae") is None:
            continue
        item = {
                "epoch": int(record["epoch"]),
                "macro_mae": float(record["validation"]["macro_mae"]),
        }
        if record.get("selection_score") is not None:
            item["selection_score"] = float(record["selection_score"])
        records.append(item)
    ranked = sorted(
        records,
        key=lambda item: (
            item.get("selection_score", item["macro_mae"]), item["epoch"]
        ),
    )
    return ranked[:keep_top_k]


def _save_ranked_candidates(
    output_dir: Path,
    payload: dict,
    history: list[dict],
    keep_top_k: int,
) -> list[dict]:
    """只保留预测指标前K个权重，供之后完整经济结算选择。"""
    ranked = _rank_candidate_records(history, keep_top_k)
    ranked_epochs = {item["epoch"] for item in ranked}
    current_epoch = int(payload["epoch"])
    current_path = output_dir / f"candidate_epoch_{current_epoch:03d}.pt"
    if current_epoch in ranked_epochs:
        torch.save(payload, current_path)

    # 删除范围严格限制在本次run自己的candidate_epoch文件。
    for path in output_dir.glob("candidate_epoch_*.pt"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if epoch not in ranked_epochs:
            path.unlink()

    manifest = []
    for rank, item in enumerate(ranked, start=1):
        path = output_dir / f"candidate_epoch_{item['epoch']:03d}.pt"
        if path.is_file():
            manifest.append({
                "forecast_rank": rank,
                **item,
                "checkpoint": path.name,
                "requires_full_settlement_selection": True,
            })
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--phase", default="inner", choices=sorted(PHASES))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--frozen-training-plan-from",
        default="",
        help=(
            "用于outer/final：从同seed的inner checkpoint冻结最终epoch和每轮学习率轨迹；"
            "诊断集只在最终epoch评估一次，不参与早停、调度或checkpoint选择"
        ),
    )
    parser.add_argument(
        "--initialize-from",
        default="",
        help="从已有同任务Global checkpoint权重开始，但使用全新优化器",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        raise ValueError("--resume与--initialize-from不能同时使用")

    cfg = PilotConfig.from_yaml(args.config)
    frozen_plan = None
    frozen_plan_path = None
    if args.frozen_training_plan_from:
        if args.phase not in {"outer", "final"}:
            raise ValueError("--frozen-training-plan-from只能用于outer或final阶段")
        if args.resume and args.epochs is not None:
            raise ValueError("冻结训练计划续训时不能另行覆盖--epochs")
        frozen_plan_path = ROOT / args.frozen_training_plan_from
        frozen_source = torch.load(
            frozen_plan_path, map_location="cpu", weights_only=False
        )
        frozen_seed = cfg.seed if args.seed is None else args.seed
        if frozen_source.get("task") != args.task:
            raise ValueError("冻结训练计划的task与当前task不一致")
        if frozen_source.get("phase") != "inner":
            raise ValueError("冻结训练计划必须来自inner checkpoint")
        if int(frozen_source.get("config", {}).get("seed", -1)) != frozen_seed:
            raise ValueError("冻结训练计划的seed与当前seed不一致")
        frozen_epochs = int(frozen_source.get("best_epoch", 0))
        if frozen_epochs < 1:
            raise ValueError("冻结训练计划缺少有效best_epoch")
        if args.epochs is not None and args.epochs != frozen_epochs:
            raise ValueError("--epochs必须与冻结训练计划的best_epoch一致")
        history_by_epoch = {
            int(record["epoch"]): record
            for record in frozen_source.get("history", [])
            if int(record.get("epoch", -1)) >= 1
        }
        if any(epoch not in history_by_epoch for epoch in range(1, frozen_epochs + 1)):
            raise ValueError("冻结训练计划的history不完整")
        lr_before_epoch = {1: float(cfg.lr)}
        for epoch in range(2, frozen_epochs + 1):
            lr_before_epoch[epoch] = float(
                history_by_epoch[epoch - 1]["learning_rate_after_scheduler"]
            )
        lr_after_epoch = {
            epoch: float(history_by_epoch[epoch]["learning_rate_after_scheduler"])
            for epoch in range(1, frozen_epochs + 1)
        }
        frozen_plan = {
            "epochs": frozen_epochs,
            "lr_before_epoch": lr_before_epoch,
            "lr_after_epoch": lr_after_epoch,
        }
    cfg = replace(
        cfg,
        seed=cfg.seed if args.seed is None else args.seed,
        epochs=(
            frozen_plan["epochs"]
            if frozen_plan is not None
            else cfg.epochs if args.epochs is None else args.epochs
        ),
        batch_size=cfg.batch_size if args.batch_size is None else args.batch_size,
    )
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    if args.phase == "final" and frozen_plan is None and not args.smoke:
        raise ValueError("final阶段必须使用预先冻结的同seed inner训练计划")
    training_markets = tuple(cfg.global_training_markets) or (cfg.market,)
    if len(set(training_markets)) != len(training_markets):
        raise ValueError("global_training_markets不能包含重复市场")
    issue_hours_by_market = {
        market: cfg.resolved_da_issue_hour_local(
            market, require_explicit=len(training_markets) > 1
        )
        for market in training_markets
    }
    verified_source_hashes = {
        market: _raw_data_hash(market) for market in training_markets
    }
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")

    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        training_markets
    )
    market_names_by_id = [
        market
        for market, _ in sorted(mappings.market_to_id.items(), key=lambda item: item[1])
    ]
    node_market_ids = torch.tensor([
        mappings.market_to_id[mappings.node_to_market[node]]
        for node, _ in sorted(mappings.node_to_id.items(), key=lambda item: item[1])
    ], dtype=torch.long)
    all_entities = [
        (mappings.node_to_market[node], node) for node in mappings.node_to_id
    ]
    if args.smoke:
        entities = []
        for market in training_markets:
            entities.extend(
                [entity for entity in all_entities if entity[0] == market][:2]
            )
    else:
        entities = all_entities
    train_stage, validation_stage = PHASES[args.phase]
    train_dataset, validation_dataset = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=train_stage,
        validation_stage=validation_stage,
        read_end_utc=(FINAL_TRAIN_READ_END_UTC if args.phase == "final" else None),
    )
    metadata = global_checkpoint_metadata(
        cfg,
        train_dataset,
        require_all_nodes=not args.smoke,
    )
    metadata["source_data_sha256"] = verified_source_hashes
    metadata["global_training_markets"] = list(training_markets)
    metadata["da_issue_hour_local_by_market"] = issue_hours_by_market
    metadata["global_sampling_mode"] = cfg.global_sampling_mode
    metadata["global_validation_aggregation"] = cfg.global_validation_aggregation
    event_thresholds = None
    if cfg.global_use_critical_event_weighting:
        event_thresholds = fit_critical_event_thresholds(
            train_dataset,
            args.task,
            cfg,
            node_count=len(mappings.node_to_id),
        )
        metadata["critical_event_weighting"] = {
            "enabled": True,
            "fit_stage": str(train_stage.value),
            "validation_weighted": False,
            "thresholds_by_node_id": event_thresholds.to_dict(),
        }
    else:
        metadata["critical_event_weighting"] = {"enabled": False}
    sampling_mode = str(cfg.global_sampling_mode).lower()
    if sampling_mode == "node_balanced":
        if len(training_markets) > 1:
            raise ValueError(
                "跨市场训练必须使用market_node_balanced，避免多节点市场主导batch"
            )
        sampler = NodeBalancedSampler(
            train_dataset,
            samples_per_node=3 if args.smoke else None,
            seed=cfg.seed,
        )
    elif sampling_mode == "market_node_balanced":
        if args.smoke:
            samples_per_market = 6
        elif cfg.global_samples_per_market > 0:
            samples_per_market = int(cfg.global_samples_per_market)
        else:
            totals: dict[str, int] = {}
            for (market, _), length in zip(
                train_dataset.entities, train_dataset.entity_lengths
            ):
                totals[market] = totals.get(market, 0) + int(length)
            samples_per_market = max(totals.values())
        sampler = MarketNodeBalancedSampler(
            train_dataset,
            samples_per_market=samples_per_market,
            seed=cfg.seed,
        )
    else:
        raise ValueError(
            "global_sampling_mode必须是node_balanced或market_node_balanced"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_sampler = (
        NodeBalancedSampler(
            validation_dataset, samples_per_node=2, seed=cfg.seed + 10_000
        )
        if args.smoke
        else None
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = TASKS[args.task]["model"](cfg).to(device)
    upstream_da_model = None
    rt_at_da_parameterization = str(cfg.rt_at_da_parameterization).lower()
    if rt_at_da_parameterization in {"da_minus_spread", "economic_regime"}:
        if args.task != "rt_at_da":
            raise ValueError(
                "需要上游DA的参数化只能用于rt_at_da任务"
            )
        if not cfg.rt_at_da_upstream_da_checkpoint:
            raise ValueError(
                "该RT-at-DA参数化需要rt_at_da_upstream_da_checkpoint"
            )
        upstream_path = ROOT / cfg.rt_at_da_upstream_da_checkpoint
        upstream = torch.load(
            upstream_path, map_location="cpu", weights_only=False
        )
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(upstream)
        mappings.validate_checkpoint(upstream)
        if upstream.get("task") != "da" or upstream.get("phase") != args.phase:
            raise ValueError("上游checkpoint必须是同阶段的DA模型")
        upstream_stats = upstream.get("norm_stats_by_node", {})
        participating_upstream_stats = {
            node: upstream_stats[node]
            for node in metadata["norm_stats_by_node"]
            if node in upstream_stats
        }
        if not _norm_stats_equal(
            participating_upstream_stats, metadata["norm_stats_by_node"]
        ):
            raise ValueError("上游DA checkpoint的逐节点scaler与当前数据不一致")
        allowed_config_fields = {item.name for item in fields(PilotConfig)}
        upstream_cfg = PilotConfig(**{
            key: value
            for key, value in upstream["config"].items()
            if key in allowed_config_fields
        })
        upstream_da_model = DecisionAwareDAForecaster(upstream_cfg).to(device)
        upstream_da_model.load_state_dict(upstream["model_state"], strict=True)
        upstream_da_model.eval()
        upstream_da_model.requires_grad_(False)
        metadata["rt_at_da_upstream_da"] = {
            "path": cfg.rt_at_da_upstream_da_checkpoint,
            "sha256": _sha256(upstream_path),
            "epoch": upstream.get("epoch"),
            "frozen": True,
            "same_information_boundary": True,
        }
    elif args.task == "rt_at_da" and cfg.rt_at_da_upstream_da_checkpoint:
        raise ValueError(
            "direct_price模式不能静默携带上游DA checkpoint"
        )
    if args.initialize_from:
        source_path = ROOT / args.initialize_from
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(source)
        mappings.validate_checkpoint(source)
        if source.get("task") != args.task or source.get("phase") != args.phase:
            raise ValueError("初始化checkpoint的task或phase不一致")
        source_stats = source.get("norm_stats_by_node", {})
        participating_source_stats = {
            node: source_stats[node]
            for node in metadata["norm_stats_by_node"] if node in source_stats
        }
        if not _norm_stats_equal(
            participating_source_stats, metadata["norm_stats_by_node"]
        ):
            raise ValueError("初始化checkpoint的逐节点scaler与当前数据不一致")
        source_state = source["model_state"]
        allowed_new_prefixes = []
        if str(getattr(cfg, "global_output_decomposition", "direct")).lower() == (
            "physical_scarcity_residual"
        ):
            allowed_new_prefixes.append("scarcity_residual.")
        if rt_at_da_parameterization == "economic_regime":
            allowed_new_prefixes.append("economic_regime_head.")
        if allowed_new_prefixes:
            incompatible = model.load_state_dict(source_state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            if unexpected or any(
                not any(key.startswith(prefix) for prefix in allowed_new_prefixes)
                for key in missing
            ):
                raise ValueError(
                    "结构初始化除显式新增头外不得改变旧模型参数："
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            model.load_state_dict(source_state, strict=True)
        metadata["initialized_from"] = {
            "path": args.initialize_from,
            "sha256": _sha256(source_path),
            "source_epoch": source.get("epoch"),
            "optimizer_reused": False,
        }
    trainable_parameters, parameter_counts = _configure_trainable_scope(
        model, cfg.global_trainable_scope
    )
    metadata["global_trainable_scope"] = cfg.global_trainable_scope
    metadata["parameter_counts"] = parameter_counts
    if frozen_plan is not None and frozen_plan_path is not None:
        metadata["frozen_training_plan"] = {
            "source": args.frozen_training_plan_from,
            "sha256": _sha256(frozen_plan_path),
            "final_epoch": frozen_plan["epochs"],
            "diagnostic_usage": (
                "2024_training_overlap_diagnostic_at_final_epoch_only"
                if args.phase == "final"
                else "outer_validation_at_final_epoch_only"
            ),
            "learning_rate_source": "inner_history",
        }
    if args.phase == "final":
        metadata["locked_final_training"] = {
            "train_stage": AnnualStage.FINAL_TRAIN.value,
            "training_years": [2020, 2021, 2022, 2023, 2024],
            "read_end_utc_exclusive_by_market": FINAL_TRAIN_READ_END_UTC,
            "locked_2025_read": False,
            "checkpoint_selection": "fixed_final_epoch_from_same_seed_inner_plan",
            "diagnostic_selection_allowed": False,
        }
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if cfg.global_keep_top_k_checkpoints < 1:
        raise ValueError("global_keep_top_k_checkpoints必须至少为1")
    if cfg.global_use_daily_peak_excursion_loss:
        if args.task != "da":
            raise ValueError("日峰值幅度损失第一版只允许用于DA任务")
        if cfg.global_use_critical_event_weighting:
            raise ValueError("日峰值幅度损失不得与关键事件加权同时开启")
        if cfg.global_daily_peak_excursion_weight <= 0:
            raise ValueError("global_daily_peak_excursion_weight必须为正")
        if cfg.global_daily_peak_excursion_beta <= 0:
            raise ValueError("global_daily_peak_excursion_beta必须为正")
    if not 0.0 < cfg.global_lr_scheduler_factor < 1.0:
        raise ValueError("global_lr_scheduler_factor必须位于(0,1)")
    if cfg.global_lr_scheduler_patience < 0:
        raise ValueError("global_lr_scheduler_patience不能为负")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.global_lr_scheduler_factor,
            patience=cfg.global_lr_scheduler_patience,
            min_lr=cfg.global_min_lr,
        )
        if cfg.global_use_lr_scheduler else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = f"{args.task}_{args.phase}_seed{cfg.seed}"
    if args.smoke:
        run_name += "_smoke"
    output_dir = ROOT / cfg.checkpoint_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    history: list[dict] = []
    start_epoch = 1
    best_epoch = 0
    best_macro_mae = float("inf")
    stale_epochs = 0
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"找不到续训文件: {latest_path}")
        state = torch.load(latest_path, map_location=device, weights_only=False)
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(state)
        mappings.validate_checkpoint(state)
        if state.get("task") != args.task or state.get("phase") != args.phase:
            raise ValueError("续训checkpoint的task或phase不一致")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["grad_scaler_state"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_macro_mae = float(
            state.get("best_validation_score", state["best_macro_mae"])
        )
        stale_epochs = int(state.get("stale_epochs", max(0, int(state["epoch"]) - best_epoch)))

    task = TASKS[args.task]
    epochs = 1 if args.smoke else cfg.epochs
    # 普通随机初始化没有候选价值；但残差锚点模型的epoch 0接近“原始锚点”，
    # 可能比训练后的残差更稳、更赚钱，必须保留供完整结算选择。
    evaluate_initial_candidate = bool(args.initialize_from) or (
        getattr(cfg, "global_residual_anchor", "none") != "none"
    )
    if evaluate_initial_candidate and not args.resume:
        baseline_validation = evaluate(
            model,
            validation_loader,
            args.task,
            len(mappings.node_to_id),
            len(mappings.market_to_id),
            node_market_ids,
            market_names_by_id,
            device,
            amp_enabled,
            upstream_da_model=upstream_da_model,
            max_batches=None,
        )
        best_epoch = 0
        best_macro_mae = _validation_selection_score(baseline_validation, cfg)
        history.append({
            "epoch": 0,
            "candidate_role": "unchanged_initial_checkpoint",
            "training_loss": cfg.global_training_loss,
            "train_normalized_objective": None,
            "train_normalized_huber": None,
            "validation": baseline_validation,
            "selection_score": best_macro_mae,
            "elapsed_seconds": 0.0,
            "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
        })
        baseline_payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, cfg, metadata, args.task, args.phase,
            0, best_epoch, best_macro_mae, stale_epochs, history,
        )
        torch.save(baseline_payload, best_path)
        _save_ranked_candidates(
            output_dir, baseline_payload, history, cfg.global_keep_top_k_checkpoints
        )
        print(json.dumps({
            "event": "initial_candidate_evaluated",
            "macro_mae": best_macro_mae,
            "source": args.initialize_from or "residual_anchor_initialization",
        }, ensure_ascii=False), flush=True)
    candidate_manifest = _rank_candidate_records(
        history, cfg.global_keep_top_k_checkpoints
    ) if history else []
    for epoch in range(start_epoch, epochs + 1):
        if frozen_plan is not None:
            for group in optimizer.param_groups:
                group["lr"] = frozen_plan["lr_before_epoch"][epoch]
        started = perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_elements = 0
        for batch in train_loader:
            batch = _prepare_batch(batch, args.task, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = _forward_task_model(
                    model, batch, args.task, upstream_da_model
                )
                elementwise_loss, target = _elementwise_training_loss(
                    output, batch, task, cfg, args.task
                )
                if event_thresholds is None:
                    point_loss = elementwise_loss.mean()
                    if cfg.global_use_daily_peak_excursion_loss:
                        peak_loss = daily_peak_excursion_loss(
                            output[task["prediction_raw"]],
                            batch[task["target"]],
                            batch[task["std"]],
                            beta=cfg.global_daily_peak_excursion_beta,
                        )
                        peak_weight = cfg.global_daily_peak_excursion_weight
                        loss = (
                            point_loss + peak_weight * peak_loss
                        ) / (1.0 + peak_weight)
                    else:
                        loss = point_loss
                    batch_weight_sum = float(target.numel())
                else:
                    event_weights = critical_event_weights(
                        batch, args.task, event_thresholds, cfg
                    )
                    loss = weighted_elementwise_mean(elementwise_loss, event_weights)
                    batch_weight_sum = float(event_weights.detach().sum())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("训练出现NaN或Inf loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * batch_weight_sum
            train_elements += batch_weight_sum

        should_evaluate = frozen_plan is None or epoch == epochs
        validation = (
            evaluate(
                model,
                validation_loader,
                args.task,
                len(mappings.node_to_id),
                len(mappings.market_to_id),
                node_market_ids,
                market_names_by_id,
                device,
                amp_enabled,
                upstream_da_model=upstream_da_model,
                max_batches=None,
            )
            if should_evaluate
            else None
        )
        record = {
            "epoch": epoch,
            "training_loss": cfg.global_training_loss,
            "train_normalized_objective": train_loss_sum / train_elements,
            "train_normalized_huber": train_loss_sum / train_elements,
            "validation": validation,
            "selection_score": (
                _validation_selection_score(validation, cfg)
                if validation is not None else None
            ),
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(record)
        improved = (
            epoch == epochs
            if frozen_plan is not None
            else record["selection_score"] < best_macro_mae
        )
        if improved:
            best_epoch = epoch
            best_macro_mae = record["selection_score"]
            stale_epochs = 0
        elif frozen_plan is None:
            stale_epochs += 1
        if frozen_plan is not None:
            for group in optimizer.param_groups:
                group["lr"] = frozen_plan["lr_after_epoch"][epoch]
        elif scheduler is not None:
            scheduler.step(record["selection_score"])
        record["learning_rate_after_scheduler"] = float(
            optimizer.param_groups[0]["lr"]
        )
        payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, cfg, metadata, args.task, args.phase,
            epoch, best_epoch, best_macro_mae, stale_epochs, history,
        )
        torch.save(payload, latest_path)
        if improved:
            torch.save(payload, best_path)
        candidate_manifest = _save_ranked_candidates(
            output_dir,
            payload,
            history,
            cfg.global_keep_top_k_checkpoints,
        )
        print(json.dumps(record, ensure_ascii=False))
        if (
            frozen_plan is None
            and not args.smoke
            and stale_epochs >= cfg.early_stop_patience
        ):
            break

    summary = {
        "task": args.task,
        "phase": args.phase,
        "seed": cfg.seed,
        "smoke_only": args.smoke,
        "entities": len(entities),
        "best_epoch": best_epoch,
        "best_macro_mae": best_macro_mae,
        "best_validation_score": best_macro_mae,
        "validation_aggregation": cfg.global_validation_aggregation,
        "candidate_checkpoints": candidate_manifest,
        "checkpoint_selection_rule": (
            "use the final epoch and learning-rate trajectory frozen from the "
            "same-seed inner checkpoint; the diagnostic split is evaluated "
            "once and cannot select the checkpoint"
            if frozen_plan is not None
            else "retain top-K by inner-validation "
            f"{cfg.global_validation_aggregation} MAE; select the final "
            "three-task triplet only by frozen full settlement"
        ),
        "best_checkpoint": best_path.relative_to(ROOT).as_posix(),
        "latest_checkpoint": latest_path.relative_to(ROOT).as_posix(),
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "entity_mapping_hash": mappings.mapping_hash,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
