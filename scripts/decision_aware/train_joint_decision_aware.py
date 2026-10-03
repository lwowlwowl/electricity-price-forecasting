#!/usr/bin/env python
"""从三个冻结结构的纯Huber checkpoint开始做完整双结算联合微调。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.joint_system import (  # noqa: E402
    estimate_system_zo_gradient,
    joint_proxy_loss,
    revenue_utility,
    settle_joint_episode,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.zero_order import compute_epsilon  # noqa: E402


PHYSICAL_FIELDS = (
    "bess_power_mw", "bess_energy_mwh", "bess_eta", "bess_init_soc_frac",
    "bess_kappa", "bess_soc_min", "bess_soc_max", "bess_e_cyc",
)


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"checkpoint结构不完整: {path}")
    return checkpoint


def _config_from_checkpoint(checkpoint: dict) -> PilotConfig:
    fields = PilotConfig.__dataclass_fields__
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items()
        if key in fields
    })


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_contract(experiment: PilotConfig, model_cfg: PilotConfig, label: str):
    mismatches = {
        name: [getattr(experiment, name), getattr(model_cfg, name)]
        for name in PHYSICAL_FIELDS
        if getattr(experiment, name) != getattr(model_cfg, name)
    }
    if mismatches:
        raise ValueError(f"{label}物理参数不一致: {mismatches}")


def _check_norm_stats(current: dict, saved: dict, label: str):
    for stream, statistics in current.items():
        if stream not in saved:
            raise ValueError(f"{label}缺少{stream}归一化统计")
        for name in ("mean", "std"):
            if not np.allclose(
                np.asarray(statistics[name], dtype=np.float64),
                np.asarray(saved[stream][name], dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(f"{label}的{stream}.{name}与联合数据不一致")


def _move(value, device):
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _flatten_rt_batch(batch: dict) -> dict:
    days, hours = next(iter(batch.values())).shape[:2]
    if hours != 24:
        raise ValueError("联合滚动RT batch第二维必须是24小时")
    return {
        key: value.reshape(days * hours, *value.shape[2:])
        for key, value in batch.items()
    }


def _parameter_ids(model) -> set[int]:
    return {parameter.data_ptr() for parameter in model.parameters()}


def _assert_independent(models: dict):
    names = list(models)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = _parameter_ids(models[left]) & _parameter_ids(models[right])
            if overlap:
                raise ValueError(f"{left}和{right}意外共享参数")


def _configure_trainable_scope(models: dict, scope: str) -> dict[str, int]:
    """冻结或开放每个独立模型的参数；不改变三个模型之间的独立性。"""
    allowed = {"all", "decoder_head", "price_head"}
    if scope not in allowed:
        raise ValueError(f"joint_trainable_scope必须属于{sorted(allowed)}")
    counts = {}
    for model_name, model in models.items():
        for name, parameter in model.named_parameters():
            if scope == "all":
                trainable = True
            elif scope == "decoder_head":
                trainable = name.startswith("decoder.") or name.startswith(
                    "price_head."
                )
            else:
                trainable = name.startswith("price_head.")
            parameter.requires_grad_(trainable)
        counts[model_name] = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        if counts[model_name] == 0:
            raise ValueError(f"{model_name}在scope={scope}下没有可训练参数")
    return counts


def _forward(models: dict, batch: dict, amp_enabled: bool, device: torch.device):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        p_da = models["da"](batch["da"])["p_da"].float()
        p_rt_at_da = models["rt_at_da"](
            batch["rt_at_da"]
        )["p_rt_at_da"].float()
        rt_flat = _flatten_rt_batch(batch["rt"])
        p_rt = models["rt"](rt_flat)["p_rt"].float()
    days = p_da.shape[0]
    return p_da, p_rt_at_da, p_rt.reshape(days, 24, -1)


def _prediction_loss(outputs, batch, cfg, source_outputs=None):
    p_da, p_rt_at_da, p_rt = outputs
    if cfg.joint_prediction_loss_mode == "truth_huber":
        targets = (
            batch["da"]["price_da_tgt"].float(),
            batch["rt_at_da"]["price_rt_at_da_tgt"].float(),
            batch["rt"]["price_rt_tgt"].float(),
        )
    elif cfg.joint_prediction_loss_mode == "source_anchor":
        if source_outputs is None:
            raise ValueError("source_anchor模式必须提供基线输出")
        targets = tuple(value.detach().float() for value in source_outputs)
    else:
        raise ValueError(
            "joint_prediction_loss_mode必须是truth_huber或source_anchor"
        )
    losses = {
        "da": F.huber_loss(
            p_da, targets[0],
            delta=cfg.huber_delta, reduction="mean",
        ) / cfg.pred_scale,
        "rt_at_da": F.huber_loss(
            p_rt_at_da, targets[1],
            delta=cfg.huber_delta, reduction="mean",
        ) / cfg.pred_scale,
        "rt": F.huber_loss(
            p_rt, targets[2],
            delta=cfg.huber_delta, reduction="mean",
        ) / cfg.pred_scale,
    }
    if cfg.joint_pred_weight_spread > 0:
        target_spread = targets[0] - targets[1]
        losses["spread"] = F.huber_loss(
            p_da - p_rt_at_da,
            target_spread,
            delta=cfg.huber_delta,
            reduction="mean",
        ) / cfg.pred_scale
    weights = {
        "da": cfg.joint_pred_weight_da,
        "rt_at_da": cfg.joint_pred_weight_rt_at_da,
        "rt": cfg.joint_pred_weight_rt,
    }
    if "spread" in losses:
        weights["spread"] = cfg.joint_pred_weight_spread
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("三项预测损失权重之和必须为正")
    total = sum(weights[name] * losses[name] for name in losses) / denominator
    return total, losses


@torch.no_grad()
def _precompute_source_outputs(models: dict, dataset, indices: list[int],
                               device: torch.device,
                               amp_enabled: bool) -> tuple[torch.Tensor, ...]:
    """一次性缓存三个原checkpoint在训练日的输出。

    缓存只有约20万个float，放在CPU；训练时按顺序切片。
    这避免每个batch再保留三个教师模型，不额外占用GPU显存。
    """
    windows = [dataset.windows[index] for index in indices]
    da_indices = [window.da_index for window in windows]
    rt_da_indices = [window.rt_at_da_index for window in windows]
    rt_indices = [
        rt_index for window in windows for rt_index in window.rt_indices
    ]
    p_da = _predict_indices(
        models["da"], dataset.da_dataset, da_indices, "p_da", 64,
        device, amp_enabled,
    )
    p_rt_at_da = _predict_indices(
        models["rt_at_da"], dataset.rt_at_da_dataset, rt_da_indices,
        "p_rt_at_da", 64, device, amp_enabled,
    )
    p_rt = _predict_indices(
        models["rt"], dataset.rt_dataset, rt_indices, "p_rt", 128,
        device, amp_enabled,
    ).reshape(len(indices), 24, -1)
    return tuple(torch.from_numpy(value).float() for value in (
        p_da, p_rt_at_da, p_rt
    ))


def _beta(epoch: int, cfg: PilotConfig, mode: str) -> float:
    if mode == "huber_control":
        return 0.0
    warmup = max(1, int(cfg.joint_beta_warmup_epochs))
    fraction = min(1.0, epoch / max(1, warmup - 1))
    return float(
        cfg.joint_beta_start
        + fraction * (cfg.joint_beta_end - cfg.joint_beta_start)
    )


def _gradient_norm(model) -> float:
    # 用float64累计，避免“每个梯度有限、但数百万个float32平方和溢出”的
    # 假Infinity；真正的NaN/Inf仍由下面的显式检查和clip守卫拦截。
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                return float("inf")
            total += float(gradient.double().square().sum().cpu())
    return math.sqrt(total)


def _block_bootstrap_ci(values: np.ndarray, samples: int, seed: int,
                        block: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    block = min(block, len(values))
    count = math.ceil(len(values) / block)
    starts_max = len(values) - block + 1
    means = []
    for _ in range(samples):
        starts = rng.integers(0, starts_max, size=count)
        draw = np.concatenate([values[start:start + block] for start in starts])
        means.append(float(draw[:len(values)].mean()))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _compact_report(report: dict, cfg: PilotConfig, seed: int) -> dict:
    daily = {
        day: float(row["net_revenue"])
        for day, row in report["daily"].items()
    }
    values = np.asarray(list(daily.values()), dtype=np.float64)
    tail_count = max(1, math.ceil(0.1 * len(values)))
    cumulative = np.cumsum(values)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    return {
        "mean_daily_revenue": float(report["mean_daily_revenue"]),
        "positive_day_rate": float(report["positive_day_rate"]),
        "lower_tail_10pct_mean": float(np.sort(values)[:tail_count].mean()),
        "max_drawdown": float(np.max(peaks - cumulative, initial=0.0)),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            values, cfg.bootstrap_samples, seed
        ),
        "days": int(report["days"]),
        "segments": int(report["segments"]),
        "revenue_components": report["revenue_components"],
        "plan_feasibility": report["plan_feasibility"],
        "actual_state": report["actual_state"],
        "daily_net_revenue": daily,
    }


def _validation_selection_score(validation: dict, cfg: PilotConfig) -> float:
    full_dual = validation["full_dual"]
    weight = float(cfg.joint_selection_tail_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("joint_selection_tail_weight必须位于[0,1]")
    return float(
        (1.0 - weight) * full_dual["mean_daily_revenue"]
        + weight * full_dual["lower_tail_10pct_mean"]
    )


def _forecast_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


@torch.no_grad()
def _predict_indices(model, dataset, indices, output_key: str,
                     batch_size: int, device: torch.device,
                     amp_enabled: bool) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    values = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            values.append(model(batch)[output_key].float().cpu().numpy())
    return np.concatenate(values)


@torch.no_grad()
def evaluate(models: dict, dataset, cfg: PilotConfig, device: torch.device,
             amp_enabled: bool, max_days: int | None = None) -> dict:
    for model in models.values():
        model.eval()
    joint_indices = list(range(len(dataset)))
    if max_days is not None:
        joint_indices = joint_indices[:max_days]
    windows = [dataset.windows[index] for index in joint_indices]
    da_indices = [window.da_index for window in windows]
    rt_da_indices = [window.rt_at_da_index for window in windows]
    rt_indices = [
        rt_index for window in windows for rt_index in window.rt_indices
    ]
    # 严格沿用融合消融的历史推理batch，避免AMP舍入改变HardTopK并列排序。
    p_da = _predict_indices(
        models["da"], dataset.da_dataset, da_indices, "p_da", 64,
        device, amp_enabled,
    )
    p_rt_at_da = _predict_indices(
        models["rt_at_da"], dataset.rt_at_da_dataset, rt_da_indices,
        "p_rt_at_da", 64, device, amp_enabled,
    )
    p_rt_flat = _predict_indices(
        models["rt"], dataset.rt_dataset, rt_indices, "p_rt", 128,
        device, amp_enabled,
    )
    p_rt = p_rt_flat.reshape(len(windows), 24, -1)
    true_da = np.stack([
        dataset.da_dataset[index]["price_da_tgt"].numpy()
        for index in da_indices
    ])
    true_rt = np.stack([
        dataset.rt_at_da_dataset[index]["price_rt_at_da_tgt"].numpy()
        for index in rt_da_indices
    ])
    true_rt_windows = np.stack([
        dataset.rt_dataset[index]["price_rt_tgt"].numpy()
        for index in rt_indices
    ]).reshape(len(windows), 24, -1)
    dates = [window.delivery_date for window in windows]
    timestamps = [
        dataset.rt_dataset.windows[index].issue_time_utc.tz_convert(
            dataset.timezone
        )
        for index in rt_indices
    ]
    report = locked_dual_backtest(
        {day: p_da[index] for index, day in enumerate(dates)},
        p_rt.reshape(-1, p_rt.shape[-1]),
        torch.from_numpy(true_da.reshape(-1)),
        torch.from_numpy(true_rt.reshape(-1)),
        timestamps,
        cfg,
        da_k_charge=cfg.topk_k_charge,
        da_k_discharge=cfg.topk_k_discharge,
        rt_k_charge=1,
        rt_k_discharge=1,
        coordination_mode="rt_only",
        rt_at_da_price_forecasts={
            day: p_rt_at_da[index] for index, day in enumerate(dates)
        },
    )
    return {
        "full_dual": _compact_report(report, cfg, cfg.seed),
        "forecast": {
            "da": _forecast_metrics(p_da, true_da),
            "rt_at_da": _forecast_metrics(p_rt_at_da, true_rt),
            "rt_all_horizons": _forecast_metrics(p_rt, true_rt_windows),
        },
    }


def _paired_delta(candidate: dict, baseline: dict, cfg: PilotConfig) -> dict:
    candidate_daily = candidate["full_dual"]["daily_net_revenue"]
    baseline_daily = baseline["full_dual"]["daily_net_revenue"]
    if list(candidate_daily) != list(baseline_daily):
        raise ValueError("候选和基线的验证日期不一致")
    values = np.asarray([
        candidate_daily[day] - baseline_daily[day]
        for day in candidate_daily
    ], dtype=np.float64)
    return {
        "mean_daily_revenue_difference": float(values.mean()),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            values, cfg.bootstrap_samples, cfg.seed + 1000
        ),
        "improved_day_rate": float(np.mean(values > 0)),
        "paired_days": int(len(values)),
    }


def _bundle(path: Path, models: dict, optimizers: dict, cfg: PilotConfig,
            epoch: int, validation: dict, selection_score: float,
            sources: dict, mode: str, trainable_counts: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_states": {
            name: {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            for name, model in models.items()
        },
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "experiment_config": cfg.to_dict(),
        "source_checkpoints": sources,
        "training_mode": mode,
        "best_epoch": epoch,
        "selection_metric": (
            "risk_adjusted_validation_utility"
            if cfg.joint_selection_tail_weight > 0
            else "validation.full_dual.mean_daily_revenue"
        ),
        "best_selection_score": float(selection_score),
        "best_validation_revenue": validation["full_dual"]["mean_daily_revenue"],
        "validation": validation,
        "trainable_parameter_counts": trainable_counts,
    }, path)


def _load_bundle(path: Path, models: dict):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    for name, model in models.items():
        model.load_state_dict(bundle["model_states"][name])
    return bundle


def _rng_state() -> dict:
    """保存可重复续训所需的随机数状态。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict | None):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_state(
        path: Path, models: dict, optimizers: dict, scaler,
        cfg: PilotConfig, completed_epoch: int, current_validation: dict,
        best_epoch: int, best_selection_score: float, best_revenue: float,
        stale: int, history: list, sources: dict, mode: str,
        trainable_counts: dict, best_checkpoint: Path):
    """每个完整epoch结束后保存续训状态；不会替代最佳模型文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "kind": "joint_training_state",
        "completed_epoch": int(completed_epoch),
        "model_states": {
            name: {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            for name, model in models.items()
        },
        "optimizer_states": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(),
        "experiment_config": cfg.to_dict(),
        "source_checkpoints": sources,
        "training_mode": mode,
        "selection_metric": (
            "risk_adjusted_validation_utility"
            if cfg.joint_selection_tail_weight > 0
            else "validation.full_dual.mean_daily_revenue"
        ),
        "current_validation": current_validation,
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_selection_score),
        "best_validation_revenue": float(best_revenue),
        "stale": int(stale),
        "history": history,
        "source_best_checkpoint": str(best_checkpoint),
        "trainable_parameter_counts": trainable_counts,
    }, path)


def _load_training_state(path: Path, models: dict, optimizers: dict, scaler) -> dict:
    """加载新式latest状态，也兼容旧版最佳checkpoint。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    for name, model in models.items():
        model.load_state_dict(state["model_states"][name])
    saved_optimizers = state.get("optimizer_states", {})
    if set(saved_optimizers) != set(optimizers):
        raise ValueError(
            "续训checkpoint的优化器与当前配置不一致: "
            f"saved={sorted(saved_optimizers)}, current={sorted(optimizers)}"
        )
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(saved_optimizers[name])
    if state.get("scaler_state") is not None:
        scaler.load_state_dict(state["scaler_state"])
    _restore_rng_state(state.get("rng_state"))
    return state


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/joint_decision_aware_v1.yaml"
    )
    parser.add_argument(
        "--mode", choices=("decision_aware", "huber_control"),
        default="decision_aware",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-val-days", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="只加载三模型并复现完整验证基线，不更新参数",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--resume-from", default=None,
        help=(
            "从latest训练状态或旧版最佳checkpoint继续。旧版checkpoint没有随机数"
            "状态，因此可恢复模型和优化器，但不能保证逐位复现原运行。"
        ),
    )
    parser.add_argument(
        "--latest-checkpoint", default=None,
        help="每个完整epoch结束后写入的可恢复训练状态路径。",
    )
    return parser.parse_args()


def main():
    args = _arguments()
    started = time.time()
    cfg = PilotConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.smoke:
        cfg.epochs = 1 if args.epochs is None else args.epochs
        args.max_train_days = args.max_train_days or 16
        args.max_val_days = args.max_val_days or 16
        # smoke也保留正式episode大小，才能在正式训练前真实验证显存峰值。
        cfg.joint_episode_days = min(cfg.joint_episode_days, 16)
        cfg.bootstrap_samples = min(cfg.bootstrap_samples, 200)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    # 历史融合消融的所有候选都用CUDA autocast做验证预测。联合训练也用AMP，
    # 但下面把GradScaler初始尺度降到64来避免默认尺度溢出；验证仍沿用历史
    # AMP和固定batch口径，才能精确复现123.06基线。
    evaluation_amp_enabled = device.type == "cuda"

    source_paths = {
        "da": cfg.joint_da_checkpoint,
        "rt_at_da": cfg.joint_rt_at_da_checkpoint,
        "rt": cfg.joint_rt_checkpoint,
    }
    if any(not path for path in source_paths.values()):
        raise ValueError("联合训练配置必须提供三个基础checkpoint")
    checkpoints = {name: _load_checkpoint(path) for name, path in source_paths.items()}
    model_cfgs = {
        name: _config_from_checkpoint(checkpoint)
        for name, checkpoint in checkpoints.items()
    }
    for name, model_cfg in model_cfgs.items():
        _check_contract(cfg, model_cfg, name)
    reference_stats = checkpoints["da"]["norm_stats"]
    train_ds, val_ds, report_ds, _ = build_joint_datasets(
        cfg, norm_stats=reference_stats
    )
    for name, checkpoint in checkpoints.items():
        _check_norm_stats(train_ds.da_dataset.norm_stats, checkpoint["norm_stats"], name)

    models = {
        "da": DecisionAwareDAForecaster(model_cfgs["da"]),
        "rt_at_da": DecisionAwareRTAtDAForecaster(model_cfgs["rt_at_da"]),
        "rt": DecisionAwareRTForecaster(model_cfgs["rt"]),
    }
    for name, model in models.items():
        model.load_state_dict(checkpoints[name]["model_state"])
        model.to(device)
    _assert_independent(models)
    trainable_counts = _configure_trainable_scope(
        models, cfg.joint_trainable_scope
    )

    source_contract = {
        name: {
            "path": str(path),
            "sha256": _sha256(path),
            "fusion_mode": checkpoints[name].get("fusion_mode"),
            "seed": checkpoints[name]["config"].get("seed"),
        }
        for name, path in source_paths.items()
    }
    epsilon_da = compute_epsilon(
        reference_stats["price_da"]["std"], cfg.zo_rho
    )
    epsilon_rt = compute_epsilon(
        reference_stats["price_rt"]["std"], cfg.zo_rho
    )
    epsilon_spread = (
        float(cfg.joint_zo_epsilon_spread)
        if cfg.joint_zo_epsilon_spread > 0
        else epsilon_da
    )
    epsilon_rt_joint = (
        float(cfg.joint_zo_epsilon_rt)
        if cfg.joint_zo_epsilon_rt > 0
        else epsilon_rt
    )
    directions_spread = int(
        cfg.joint_zo_directions_spread or cfg.zo_K
    )
    directions_rt = int(cfg.joint_zo_directions_rt or cfg.zo_K)
    print(json.dumps({
        "event": "contract",
        "mode": args.mode,
        "device": str(device),
        "train_days": len(train_ds),
        "validation_days": len(val_ds),
        "report_only_days": len(report_ds),
        "episode_days": cfg.joint_episode_days,
        "epsilon_da": epsilon_da,
        "epsilon_rt": epsilon_rt,
        "epsilon_spread_used": epsilon_spread,
        "epsilon_rt_used": epsilon_rt_joint,
        "directions_spread": directions_spread,
        "directions_rt": directions_rt,
        "da_proxy_mode": cfg.joint_da_proxy_mode,
        "zo_direction_mode": cfg.joint_zo_direction_mode,
        "zo_feedback_mode": cfg.joint_zo_feedback_mode,
        "zo_rt_feedback_mode": cfg.joint_zo_rt_feedback_mode,
        "proxy_reduction": cfg.joint_proxy_reduction,
        "prediction_loss_mode": cfg.joint_prediction_loss_mode,
        "trainable_scope": cfg.joint_trainable_scope,
        "trainable_parameter_counts": trainable_counts,
        "separate_optimizers": cfg.joint_separate_optimizers,
        "dropout_disabled_during_finetune": cfg.joint_disable_dropout,
        "sources": source_contract,
    }, ensure_ascii=False), flush=True)

    baseline_validation = evaluate(
        models, val_ds, cfg, device, evaluation_amp_enabled, args.max_val_days
    )
    print(json.dumps({
        "event": "baseline_validation",
        "mean_daily_revenue": baseline_validation["full_dual"]["mean_daily_revenue"],
        "days": baseline_validation["full_dual"]["days"],
    }, ensure_ascii=False), flush=True)
    if args.audit_only:
        return

    train_indices = list(range(len(train_ds)))
    if args.max_train_days is not None:
        train_indices = train_indices[:args.max_train_days]
    source_output_cache = None
    if cfg.joint_prediction_loss_mode == "source_anchor":
        source_output_cache = _precompute_source_outputs(
            models, train_ds, train_indices, device, evaluation_amp_enabled
        )
        print(json.dumps({
            "event": "source_anchor_cached",
            "days": len(train_indices),
            "shapes": [list(value.shape) for value in source_output_cache],
        }, ensure_ascii=False), flush=True)
    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=cfg.joint_episode_days,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    parameters_by_model = {
        name: [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        for name, model in models.items()
    }
    parameters = [
        parameter for values in parameters_by_model.values()
        for parameter in values
    ]
    if cfg.joint_separate_optimizers:
        optimizers = {
            name: torch.optim.AdamW(
                values, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            for name, values in parameters_by_model.items()
        }
    else:
        optimizers = {
            "joint": torch.optim.AdamW(
                parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        }
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=64.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000,
    )
    output = Path(args.output or (
        f"data/results/joint_{args.mode}_seed{cfg.seed}"
        + ("_smoke.json" if args.smoke else ".json")
    ))
    checkpoint_path = Path(args.checkpoint or (
        f"data/checkpoints/joint_{args.mode}_seed{cfg.seed}/"
        + ("smoke.pt" if args.smoke else "best_validation_revenue.pt")
    ))
    latest_checkpoint_path = Path(args.latest_checkpoint or (
        checkpoint_path.parent / ("latest_smoke.pt" if args.smoke else "latest.pt")
    ))

    best_revenue = -float("inf")
    best_selection_score = -float("inf")
    best_epoch = None
    stale = 0
    history = []
    start_epoch = 0
    resume_contract = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"找不到续训checkpoint: {resume_path}")
        state = _load_training_state(
            resume_path, models, optimizers, scaler
        )
        if state.get("training_mode") != args.mode:
            raise ValueError(
                "续训checkpoint的training_mode不一致: "
                f"saved={state.get('training_mode')!r}, current={args.mode!r}"
            )
        saved_sources = state.get("source_checkpoints", {})
        for name, source in source_contract.items():
            if saved_sources.get(name, {}).get("sha256") != source["sha256"]:
                raise ValueError(f"续训checkpoint的{name}源模型哈希不一致")

        is_exact_state = state.get("kind") == "joint_training_state"
        start_epoch = int(
            state.get("completed_epoch", state.get("best_epoch", 0))
        )
        best_epoch = int(state.get("best_epoch", start_epoch))
        current_validation = state.get(
            "current_validation", state.get("validation")
        )
        if current_validation is None:
            raise ValueError("续训checkpoint缺少验证结果")
        best_revenue = float(state.get(
            "best_validation_revenue",
            current_validation["full_dual"]["mean_daily_revenue"],
        ))
        if is_exact_state:
            expected_metric = (
                "risk_adjusted_validation_utility"
                if cfg.joint_selection_tail_weight > 0
                else "validation.full_dual.mean_daily_revenue"
            )
            if state.get("selection_metric") != expected_metric:
                raise ValueError(
                    "续训状态的选模指标与当前配置不一致: "
                    f"saved={state.get('selection_metric')!r}, "
                    f"current={expected_metric!r}"
                )
            best_selection_score = float(state["best_selection_score"])
            stale = int(state.get("stale", 0))
            history = list(state.get("history", []))
            saved_best_path = state.get("source_best_checkpoint")
            if (
                saved_best_path is not None
                and Path(saved_best_path).resolve() != checkpoint_path.resolve()
            ):
                raise ValueError(
                    "续训时--checkpoint必须与latest状态记录的最佳模型路径一致: "
                    f"saved={saved_best_path}, current={checkpoint_path}"
                )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "恢复latest状态时还必须保留对应的最佳checkpoint: "
                    f"{checkpoint_path}"
                )
        else:
            # 旧版最佳checkpoint没有latest所需的随机数、scaler和stale状态。
            # 它仍可作为一个新的、非逐位复现的续训起点。选模分数按当前
            # 配置重新计算，并另存到新的最佳checkpoint，避免覆盖旧文件。
            best_selection_score = _validation_selection_score(
                current_validation, cfg
            )
            stale = 0
            history = []
            _bundle(
                checkpoint_path, models, optimizers, cfg, best_epoch,
                current_validation, best_selection_score, source_contract,
                args.mode, trainable_counts,
            )
        resume_contract = {
            "path": str(resume_path),
            "completed_epoch": start_epoch,
            "exact_state": is_exact_state,
        }
        print(json.dumps({
            "event": "resumed",
            **resume_contract,
            "best_epoch": best_epoch,
            "best_selection_score": best_selection_score,
            "best_validation_revenue": best_revenue,
        }, ensure_ascii=False), flush=True)
    elif cfg.joint_include_baseline_candidate:
        best_revenue = baseline_validation["full_dual"]["mean_daily_revenue"]
        best_selection_score = _validation_selection_score(
            baseline_validation, cfg
        )
        best_epoch = 0
        _bundle(
            checkpoint_path, models, optimizers, cfg, best_epoch,
            baseline_validation, best_selection_score, source_contract,
            args.mode, trainable_counts,
        )
        print(json.dumps({
            "event": "baseline_candidate_saved",
            "selection_score": best_selection_score,
            "mean_daily_revenue": best_revenue,
        }, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        for model in models.values():
            if cfg.joint_disable_dropout:
                model.eval()
            else:
                model.train()
        beta = _beta(epoch, cfg, args.mode)
        plan_soc = None
        actual_soc = None
        rows = []
        epoch_started = time.time()
        batch_offset = 0
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            outputs = _forward(models, batch, amp_enabled, device)
            p_da, p_rt_at_da, p_rt = outputs
            true_da = batch["da"]["price_da_tgt"].float()
            true_rt = batch["rt_at_da"]["price_rt_at_da_tgt"].float()
            reset = batch["starts_new_segment"]
            batch_days = p_da.shape[0]
            source_outputs = None
            if source_output_cache is not None:
                source_outputs = tuple(
                    value[batch_offset:batch_offset + batch_days].to(
                        device, non_blocking=True
                    )
                    for value in source_output_cache
                )
            pred_loss, pred_parts = _prediction_loss(
                outputs, batch, cfg, source_outputs
            )
            batch_offset += batch_days

            def settle(da_value, rt_da_value, rt_value):
                return settle_joint_episode(
                    da_value, rt_da_value, rt_value,
                    true_da, true_rt, reset, cfg,
                    initial_plan_soc_mwh=plan_soc,
                    initial_actual_soc_mwh=actual_soc,
                )

            base = settle(p_da.detach(), p_rt_at_da.detach(), p_rt.detach())
            if beta > 0.0:
                zo_kwargs = {
                    "pred_clamp": tuple(cfg.pred_clamp),
                    "direction_mode": cfg.joint_zo_direction_mode,
                    "tail_weight": cfg.joint_tail_weight,
                    "tail_fraction": cfg.joint_tail_fraction,
                }
                if cfg.joint_da_proxy_mode == "coupled_spread":
                    spread = p_da - p_rt_at_da
                    grad_spread, diag_spread = estimate_system_zo_gradient(
                        spread,
                        lambda value: settle(
                            value,
                            torch.zeros_like(value),
                            p_rt.detach(),
                        )["daily_revenue"],
                        epsilon=epsilon_spread,
                        directions=directions_spread,
                        feedback_mode=cfg.joint_zo_feedback_mode,
                        **zo_kwargs,
                    )
                    proxy_parts = {
                        "spread": joint_proxy_loss(
                            spread, grad_spread, cfg.joint_proxy_scale,
                            reduction=cfg.joint_proxy_reduction,
                        )
                    }
                    diagnostics = {"spread": diag_spread}
                elif cfg.joint_da_proxy_mode == "independent_prices":
                    grad_da, diag_da = estimate_system_zo_gradient(
                        p_da,
                        lambda value: settle(
                            value, p_rt_at_da.detach(), p_rt.detach()
                        )["daily_revenue"],
                        epsilon=epsilon_da,
                        directions=directions_spread,
                        feedback_mode=cfg.joint_zo_feedback_mode,
                        **zo_kwargs,
                    )
                    grad_rt_da, diag_rt_da = estimate_system_zo_gradient(
                        p_rt_at_da,
                        lambda value: settle(
                            p_da.detach(), value, p_rt.detach()
                        )["daily_revenue"],
                        epsilon=epsilon_rt,
                        directions=directions_spread,
                        feedback_mode=cfg.joint_zo_feedback_mode,
                        **zo_kwargs,
                    )
                    proxy_parts = {
                        "da": joint_proxy_loss(
                            p_da, grad_da, cfg.joint_proxy_scale,
                            reduction=cfg.joint_proxy_reduction,
                        ),
                        "rt_at_da": joint_proxy_loss(
                            p_rt_at_da, grad_rt_da, cfg.joint_proxy_scale,
                            reduction=cfg.joint_proxy_reduction,
                        ),
                    }
                    diagnostics = {"da": diag_da, "rt_at_da": diag_rt_da}
                else:
                    raise ValueError(
                        "joint_da_proxy_mode必须是independent_prices或"
                        "coupled_spread"
                    )
                grad_rt, diag_rt = estimate_system_zo_gradient(
                    p_rt,
                    lambda value: settle(
                        p_da.detach(), p_rt_at_da.detach(), value
                    )[
                        "hourly_revenue"
                        if cfg.joint_zo_rt_feedback_mode == "per_group"
                        else "daily_revenue"
                    ],
                    epsilon=epsilon_rt_joint,
                    directions=directions_rt,
                    feedback_mode=cfg.joint_zo_rt_feedback_mode,
                    **zo_kwargs,
                )
                proxy_parts["rt"] = joint_proxy_loss(
                    p_rt, grad_rt, cfg.joint_proxy_scale,
                    reduction=cfg.joint_proxy_reduction,
                )
                diagnostics["rt"] = diag_rt
                if cfg.joint_da_proxy_mode == "coupled_spread":
                    proxy_weights = {
                        "spread": cfg.joint_proxy_weight_spread,
                        "rt": cfg.joint_proxy_weight_rt,
                    }
                else:
                    proxy_weights = {name: 1.0 for name in proxy_parts}
                proxy_denominator = sum(proxy_weights.values())
                if proxy_denominator <= 0:
                    raise ValueError("联合收益代理权重之和必须为正")
                proxy_loss = sum(
                    proxy_weights[name] * proxy_parts[name]
                    for name in proxy_parts
                ) / proxy_denominator
            else:
                diagnostic_names = (
                    ("spread", "rt")
                    if cfg.joint_da_proxy_mode == "coupled_spread"
                    else tuple(models)
                )
                proxy_parts = {
                    name: pred_loss.new_zeros(()) for name in diagnostic_names
                }
                proxy_loss = pred_loss.new_zeros(())
                diagnostics = {
                    name: {
                        "gradient_norm": 0.0,
                        "gradient_rms": 0.0,
                        "nonzero_fraction": 0.0,
                        "changed_direction_fraction": 0.0,
                        "finite": True,
                    }
                    for name in diagnostic_names
                }
            loss = cfg.joint_alpha * pred_loss + beta * proxy_loss
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            for optimizer in optimizers.values():
                scaler.unscale_(optimizer)
            parameter_gradient_norms = {
                name: _gradient_norm(model) for name, model in models.items()
            }
            if any(not math.isfinite(value) for value in parameter_gradient_norms.values()):
                raise FloatingPointError(
                    f"检测到非有限参数梯度: {parameter_gradient_norms}"
                )
            if cfg.joint_separate_optimizers:
                for name, values in parameters_by_model.items():
                    torch.nn.utils.clip_grad_norm_(
                        values, cfg.grad_clip, error_if_nonfinite=True
                    )
            else:
                torch.nn.utils.clip_grad_norm_(
                    parameters, cfg.grad_clip, error_if_nonfinite=True
                )
            for optimizer in optimizers.values():
                scaler.step(optimizer)
            scaler.update()

            plan_soc = base["final_state"].plan_soc_mwh
            actual_soc = base["final_state"].actual_soc_mwh
            rows.append({
                "loss_total": float(loss.detach().cpu()),
                "loss_pred": float(pred_loss.detach().cpu()),
                "loss_proxy": float(proxy_loss.detach().cpu()),
                "train_revenue": float(base["daily_revenue"].mean().cpu()),
                "train_utility": float(revenue_utility(
                    base["daily_revenue"],
                    tail_weight=cfg.joint_tail_weight,
                    tail_fraction=cfg.joint_tail_fraction,
                ).cpu()),
                **{
                    f"pred_{name}": float(value.detach().cpu())
                    for name, value in pred_parts.items()
                },
                **{
                    f"proxy_{name}": float(value.detach().cpu())
                    for name, value in proxy_parts.items()
                },
                **{
                    f"zo_{name}_gradient_norm": diagnostics[name]["gradient_norm"]
                    for name in diagnostics
                },
                **{
                    f"zo_{name}_changed_fraction": diagnostics[name]["changed_direction_fraction"]
                    for name in diagnostics
                },
                **{
                    f"parameter_{name}_gradient_norm": parameter_gradient_norms[name]
                    for name in models
                },
            })

        validation = evaluate(
            models, val_ds, cfg, device, evaluation_amp_enabled, args.max_val_days
        )
        train_summary = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        epoch_row = {
            "epoch": epoch + 1,
            "beta": beta,
            "train": train_summary,
            "validation": {
                "mean_daily_revenue": validation["full_dual"]["mean_daily_revenue"],
                "positive_day_rate": validation["full_dual"]["positive_day_rate"],
                "lower_tail_10pct_mean": validation["full_dual"]["lower_tail_10pct_mean"],
                "max_drawdown": validation["full_dual"]["max_drawdown"],
                "selection_score": _validation_selection_score(validation, cfg),
                "forecast": validation["forecast"],
            },
            "wall_time_seconds": round(time.time() - epoch_started, 2),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        revenue = validation["full_dual"]["mean_daily_revenue"]
        selection_score = _validation_selection_score(validation, cfg)
        if selection_score > best_selection_score:
            best_revenue = revenue
            best_selection_score = selection_score
            best_epoch = epoch + 1
            stale = 0
            _bundle(
                checkpoint_path, models, optimizers, cfg, best_epoch,
                validation, selection_score, source_contract, args.mode,
                trainable_counts,
            )
        else:
            stale += 1
        _save_training_state(
            latest_checkpoint_path, models, optimizers, scaler, cfg,
            completed_epoch=epoch + 1,
            current_validation=validation,
            best_epoch=best_epoch,
            best_selection_score=best_selection_score,
            best_revenue=best_revenue,
            stale=stale,
            history=history,
            sources=source_contract,
            mode=args.mode,
            trainable_counts=trainable_counts,
            best_checkpoint=checkpoint_path,
        )
        print(json.dumps({
            "event": "latest_saved",
            "path": str(latest_checkpoint_path),
            "completed_epoch": epoch + 1,
            "best_epoch": best_epoch,
        }, ensure_ascii=False), flush=True)
        if stale >= cfg.early_stop_patience:
            break

    if best_epoch is None:
        raise RuntimeError("联合训练没有产生可保存的checkpoint")
    bundle = _load_bundle(checkpoint_path, models)
    for model in models.values():
        model.to(device)
    best_validation = evaluate(
        models, val_ds, cfg, device, evaluation_amp_enabled, args.max_val_days
    )
    paired = _paired_delta(best_validation, baseline_validation, cfg)
    report_only = None
    baseline_report_only = None
    if not args.smoke:
        # 仅在训练和验证选择全部结束后披露；绝不参与epoch选择或超参数回选。
        report_only = evaluate(
            models, report_ds, cfg, device, evaluation_amp_enabled
        )
        baseline_models = {
            "da": DecisionAwareDAForecaster(model_cfgs["da"]),
            "rt_at_da": DecisionAwareRTAtDAForecaster(model_cfgs["rt_at_da"]),
            "rt": DecisionAwareRTForecaster(model_cfgs["rt"]),
        }
        for name, model in baseline_models.items():
            model.load_state_dict(checkpoints[name]["model_state"])
            model.to(device)
        baseline_report_only = evaluate(
            baseline_models, report_ds, cfg, device, evaluation_amp_enabled
        )
        del baseline_models

    result = {
        "run_type": "smoke" if args.smoke else "formal_training",
        "training_mode": args.mode,
        "contract": (
            "three independent pretrained Transformers; locked DA plan; hourly "
            "rolling RT first action; continuous SOC within observed segments; "
            "terminal SOC reported in MWh and not monetized"
        ),
        "selection_metric": (
            "risk-adjusted validation utility"
            if cfg.joint_selection_tail_weight > 0
            else "validation full-dual mean daily revenue"
        ),
        "selection_tail_weight": cfg.joint_selection_tail_weight,
        "report_only_role": "disclosed after selection; never used for model selection",
        "device": str(device),
        "mixed_precision_training": amp_enabled,
        "mixed_precision_evaluation": evaluation_amp_enabled,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "source_checkpoints": source_contract,
        "parameter_counts": {
            name: sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "trainable_parameter_counts": trainable_counts,
        "samples": {
            "train": len(train_indices),
            "validation": min(len(val_ds), args.max_val_days or len(val_ds)),
            "report_only": len(report_ds) if not args.smoke else 0,
        },
        "epsilon": {
            "da_raw": epsilon_da,
            "rt_raw": epsilon_rt,
            "spread_used": epsilon_spread,
            "rt_used": epsilon_rt_joint,
        },
        "zo_directions": {
            "spread": directions_spread,
            "rt": directions_rt,
        },
        "history": history,
        "resume": resume_contract,
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "baseline_validation": baseline_validation,
        "best_validation": best_validation,
        "paired_validation_candidate_minus_baseline": paired,
        "baseline_report_only": baseline_report_only,
        "candidate_report_only": report_only,
        "checkpoint": str(checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "checkpoint_best_validation_revenue": bundle["best_validation_revenue"],
        "wall_time_seconds": round(time.time() - started, 2),
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            if device.type == "cuda" else 0.0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "complete",
        "output": str(output),
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "baseline_validation_revenue": baseline_validation["full_dual"]["mean_daily_revenue"],
        "candidate_validation_revenue": best_validation["full_dual"]["mean_daily_revenue"],
        "paired_difference": paired,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
