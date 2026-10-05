#!/usr/bin/env python
"""在同一182日、同一物理/结算合同下横向比较五类完整系统。"""
from __future__ import annotations

import argparse
import copy
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

from decision_aware.baselines_joint import (  # noqa: E402
    JointGRUBaseline,
    baseline_dataset_to_numpy,
    fit_joint_xgboost,
    weekly_seasonal_task_predictions,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.joint_system import settle_joint_episode  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import (  # noqa: E402
    DecisionAwareRTAtDAForecaster,
)


MODEL_TYPES = {
    "da": DecisionAwareDAForecaster,
    "rt_at_da": DecisionAwareRTAtDAForecaster,
    "rt": DecisionAwareRTForecaster,
}
TARGET_KEYS = {
    "da": "price_da_tgt",
    "rt_at_da": "price_rt_at_da_tgt",
    "rt": "price_rt_tgt",
}
OUTPUT_KEYS = {"da": "p_da", "rt_at_da": "p_rt_at_da", "rt": "p_rt"}
SOURCE_CONFIG_KEYS = {
    "da": "joint_da_checkpoint",
    "rt_at_da": "joint_rt_at_da_checkpoint",
    "rt": "joint_rt_checkpoint",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_from_dict(values: dict) -> PilotConfig:
    fields = PilotConfig.__dataclass_fields__
    return PilotConfig(**{key: value for key, value in values.items() if key in fields})


def _load_source_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"基础checkpoint结构不完整: {path}")
    return checkpoint


def _resolve_bundle_source_path(path_text: str, bundle_path: Path) -> Path:
    """解析联合checkpoint记录的基础模型路径。

    历史bundle通常保存项目根目录相对路径；同时兼容绝对路径和
    以bundle所在目录为基准的相对路径。
    """
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, bundle_path.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"联合checkpoint的基础模型不存在: {path_text}; 已尝试: {attempted}"
    )


def _load_bundle_source_checkpoints(
    bundle: dict,
    bundle_path: str | Path,
) -> tuple[dict[str, Path], dict[str, dict], dict[str, dict]]:
    """从联合bundle自身记录恢复三个基础模型的结构合同。

    不能用当前比较配置中的source checkpoint去构建历史V2模型：
    fusion mode或其他结构参数变更时，这会造成架构错配，甚至静默误标。
    """
    bundle_path = Path(bundle_path)
    declared = bundle.get("source_checkpoints", {})
    experiment = bundle.get("experiment_config", {})
    paths: dict[str, Path] = {}
    checkpoints: dict[str, dict] = {}
    audit: dict[str, dict] = {}
    for task, config_key in SOURCE_CONFIG_KEYS.items():
        source = declared.get(task)
        expected_sha = None
        if isinstance(source, dict):
            path_text = source.get("path")
            expected_sha = source.get("sha256")
        elif isinstance(source, (str, Path)):
            path_text = str(source)
        else:
            path_text = None
        if not path_text:
            path_text = experiment.get(config_key)
        if not path_text:
            raise ValueError(
                f"联合checkpoint缺少{task}的source checkpoint路径: {bundle_path}"
            )

        resolved = _resolve_bundle_source_path(str(path_text), bundle_path)
        actual_sha = _sha256(resolved)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"联合checkpoint记录的{task} source SHA256不匹配: "
                f"expected={expected_sha}, actual={actual_sha}, path={resolved}"
            )
        checkpoint = _load_source_checkpoint(resolved)
        paths[task] = resolved
        checkpoints[task] = checkpoint
        audit[task] = {
            "path": str(resolved),
            "sha256": actual_sha,
            "fusion_mode": checkpoint.get("fusion_mode"),
            "seed": checkpoint["config"].get("seed"),
            "declared_by_bundle": bool(source),
        }
    return paths, checkpoints, audit


def _check_norm_stats(reference: dict, candidate: dict, label: str) -> None:
    """确保历史bundle的模型能在当前统一数据合同上正确推理。"""
    for stream, statistics in reference.items():
        if stream not in candidate:
            raise ValueError(f"{label}缺少{stream}归一化统计")
        for name in ("mean", "std"):
            if not np.allclose(
                np.asarray(statistics[name], dtype=np.float64),
                np.asarray(candidate[stream][name], dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{label}的{stream}.{name}与统一比较数据合同不一致"
                )


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _task_datasets(joint_dataset) -> dict:
    return {
        "da": joint_dataset.da_dataset,
        "rt_at_da": joint_dataset.rt_at_da_dataset,
        "rt": joint_dataset.rt_dataset,
    }


def _task_indices(joint_dataset) -> dict[str, list[int]]:
    return {
        "da": [window.da_index for window in joint_dataset.windows],
        "rt_at_da": [
            window.rt_at_da_index for window in joint_dataset.windows
        ],
        "rt": [
            index for window in joint_dataset.windows for index in window.rt_indices
        ],
    }


def _truth_and_contract(joint_dataset, max_days: int | None = None) -> dict:
    windows = joint_dataset.windows
    if max_days is not None:
        windows = windows[:max_days]
    da_indices = [window.da_index for window in windows]
    rt_da_indices = [window.rt_at_da_index for window in windows]
    rt_indices = [index for window in windows for index in window.rt_indices]
    true_da = np.stack([
        joint_dataset.da_dataset[index]["price_da_tgt"].numpy()
        for index in da_indices
    ]).astype(np.float32)
    true_rt = np.stack([
        joint_dataset.rt_at_da_dataset[index]["price_rt_at_da_tgt"].numpy()
        for index in rt_da_indices
    ]).astype(np.float32)
    true_rt_windows = np.stack([
        joint_dataset.rt_dataset[index]["price_rt_tgt"].numpy()
        for index in rt_indices
    ]).reshape(len(windows), 24, -1).astype(np.float32)
    return {
        "dates": [str(window.delivery_date) for window in windows],
        "da_indices": da_indices,
        "rt_at_da_indices": rt_da_indices,
        "rt_indices": rt_indices,
        "starts_new_segment": np.asarray(
            [window.starts_new_segment for window in windows], dtype=bool
        ),
        "true_da": true_da,
        "true_rt": true_rt,
        "true_rt_windows": true_rt_windows,
    }


@torch.no_grad()
def _predict_model(
    model,
    dataset,
    indices: list[int],
    output_key: str,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    predictions = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions.append(model(batch)[output_key].float().cpu().numpy())
    return np.concatenate(predictions)


def _build_transformers(
    source_checkpoints: dict,
    device: torch.device,
    model_states: dict | None = None,
) -> dict:
    models = {}
    for task, checkpoint in source_checkpoints.items():
        model = MODEL_TYPES[task](_config_from_dict(checkpoint["config"]))
        state = checkpoint["model_state"] if model_states is None else model_states[task]
        model.load_state_dict(state)
        model.to(device)
        models[task] = model
    return models


def _predict_transformer_system(
    models: dict,
    val_dataset,
    contract: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    amp_enabled = device.type == "cuda"
    datasets = _task_datasets(val_dataset)
    p_da = _predict_model(
        models["da"], datasets["da"], contract["da_indices"], "p_da",
        64, device, amp_enabled,
    )
    p_rt_at_da = _predict_model(
        models["rt_at_da"], datasets["rt_at_da"],
        contract["rt_at_da_indices"], "p_rt_at_da", 64, device, amp_enabled,
    )
    p_rt = _predict_model(
        models["rt"], datasets["rt"], contract["rt_indices"], "p_rt",
        128, device, amp_enabled,
    ).reshape(len(contract["dates"]), 24, -1)
    return {"da": p_da, "rt_at_da": p_rt_at_da, "rt": p_rt}


def _clamp_predictions(predictions: dict, cfg: PilotConfig) -> dict:
    lo, hi = (float(value) for value in cfg.pred_clamp)
    return {
        key: np.clip(np.asarray(value, dtype=np.float32), lo, hi)
        for key, value in predictions.items()
    }


def _seasonal_predictions(val_dataset, contract: dict) -> dict:
    datasets = _task_datasets(val_dataset)
    predictions = {}
    for task in ("da", "rt_at_da", "rt"):
        values = weekly_seasonal_task_predictions(
            datasets[task], task, contract[f"{task}_indices"]
        )
        if task == "rt":
            values = values.reshape(len(contract["dates"]), 24, -1)
        predictions[task] = values
    return predictions


def _cache_metadata(
    cfg: PilotConfig,
    contract: dict,
    model: str,
    seed: int,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "format_version": 1,
        "model": model,
        "seed": seed,
        "market": cfg.market,
        "node": cfg.node,
        "dates": contract["dates"],
        "train_bounds": list(cfg.split_bounds("train")),
        "validation_bounds": list(cfg.split_bounds("val")),
        "context_len": cfg.context_len,
        "horizon_rt": cfg.horizon_rt,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _read_prediction_cache(path: Path, expected: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata != expected:
                return None
            return {
                "da": cached["da"].astype(np.float32),
                "rt_at_da": cached["rt_at_da"].astype(np.float32),
                "rt": cached["rt"].astype(np.float32),
            }
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def _write_prediction_cache(path: Path, predictions: dict, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        da=predictions["da"],
        rt_at_da=predictions["rt_at_da"],
        rt=predictions["rt"],
        metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )


def _xgboost_predictions(
    train_dataset,
    val_dataset,
    train_indices: dict,
    contract: dict,
    cfg: PilotConfig,
    seed: int,
    cache_dir: Path,
    force: bool,
) -> tuple[dict, dict]:
    metadata = _cache_metadata(cfg, contract, "xgboost", seed, {
        "parameters": {
            "n_estimators": 180,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        }
    })
    cache = cache_dir / f"xgboost_seed{seed}_validation_predictions.npz"
    if not force:
        predictions = _read_prediction_cache(cache, metadata)
        if predictions is not None:
            return predictions, {"cache": str(cache), "cache_hit": True}

    datasets_train = _task_datasets(train_dataset)
    datasets_val = _task_datasets(val_dataset)
    predictions = {}
    task_seconds = {}
    for task in ("da", "rt_at_da", "rt"):
        started = time.time()
        X_train, y_train = baseline_dataset_to_numpy(
            datasets_train[task], task, train_indices[task]
        )
        X_val, _ = baseline_dataset_to_numpy(
            datasets_val[task], task, contract[f"{task}_indices"]
        )
        estimator = fit_joint_xgboost(X_train, y_train, seed)
        values = estimator.predict(X_val).astype(np.float32)
        if task == "rt":
            values = values.reshape(len(contract["dates"]), 24, -1)
        predictions[task] = values
        task_seconds[task] = round(time.time() - started, 2)
        print(json.dumps({
            "event": "xgboost_task_complete", "seed": seed, "task": task,
            "seconds": task_seconds[task], "train_samples": len(X_train),
        }, ensure_ascii=False), flush=True)
        del X_train, y_train, X_val, estimator
    _write_prediction_cache(cache, predictions, metadata)
    return predictions, {
        "cache": str(cache), "cache_hit": False, "task_seconds": task_seconds,
    }


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_gru_task(
    task: str,
    train_dataset,
    val_dataset,
    train_indices: list[int],
    val_indices: list[int],
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[JointGRUBaseline, list[dict]]:
    _seed_everything(seed)
    model = JointGRUBaseline(task, hidden_size=64).to(device)
    train_batch_size = 256 if task == "rt" else 64
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=256 if task == "rt" else 128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    best_state = None
    best_mae = float("inf")
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            prediction = model(batch)
            loss = F.smooth_l1_loss(prediction, model.normalized_target(batch))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        errors = []
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _move(raw_batch, device)
                prediction = model.to_price(model(batch), batch)
                errors.append(torch.abs(
                    prediction - batch[model.spec.target_key]
                ).cpu())
        val_mae = float(torch.cat(errors).mean())
        row = {
            "epoch": epoch,
            "train_huber": float(np.mean(losses)),
            "validation_mae": val_mae,
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(json.dumps({
            "event": "gru_epoch", "task": task, "seed": seed, **row,
        }, ensure_ascii=False), flush=True)
        if val_mae < best_mae - 1e-8:
            best_mae = val_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"GRU {task}没有产生有效checkpoint")
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _predict_gru_task(
    model: JointGRUBaseline,
    dataset,
    indices: list[int],
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices), batch_size=256, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model.eval()
    values = []
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        values.append(model.to_price(model(batch), batch).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _gru_predictions(
    train_dataset,
    val_dataset,
    train_indices: dict,
    contract: dict,
    cfg: PilotConfig,
    seed: int,
    epochs: int,
    patience: int,
    device: torch.device,
    cache_dir: Path,
    force: bool,
) -> tuple[dict, dict]:
    metadata = _cache_metadata(cfg, contract, "gru", seed, {
        "parameters": {
            "hidden_size": 64,
            "max_epochs": epochs,
            "early_stop_patience": patience,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "selection": "validation_price_mae",
        }
    })
    cache = cache_dir / f"gru_seed{seed}_validation_predictions.npz"
    if not force:
        predictions = _read_prediction_cache(cache, metadata)
        if predictions is not None:
            return predictions, {"cache": str(cache), "cache_hit": True}

    datasets_train = _task_datasets(train_dataset)
    datasets_val = _task_datasets(val_dataset)
    predictions = {}
    histories = {}
    for task in ("da", "rt_at_da", "rt"):
        model, history = _train_gru_task(
            task, datasets_train[task], datasets_val[task],
            train_indices[task], contract[f"{task}_indices"], device,
            seed, epochs, patience,
        )
        values = _predict_gru_task(
            model, datasets_val[task], contract[f"{task}_indices"], device
        )
        if task == "rt":
            values = values.reshape(len(contract["dates"]), 24, -1)
        predictions[task] = values
        histories[task] = history
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_prediction_cache(cache, predictions, metadata)
    return predictions, {
        "cache": str(cache), "cache_hit": False,
        "training_history": histories,
    }


def _forecast_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _block_bootstrap_ci(
    values: np.ndarray, samples: int, seed: int, block: int = 7
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    block = min(block, len(values))
    draws_per_sample = math.ceil(len(values) / block)
    starts_max = len(values) - block + 1
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        starts = rng.integers(0, starts_max, size=draws_per_sample)
        draw = np.concatenate([values[start:start + block] for start in starts])
        means[sample] = draw[:len(values)].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _settle_and_summarize(
    predictions: dict,
    contract: dict,
    cfg: PilotConfig,
    bootstrap_seed: int,
) -> dict:
    predictions = _clamp_predictions(predictions, cfg)
    result = settle_joint_episode(
        torch.from_numpy(predictions["da"]),
        torch.from_numpy(predictions["rt_at_da"]),
        torch.from_numpy(predictions["rt"]),
        torch.from_numpy(contract["true_da"]),
        torch.from_numpy(contract["true_rt"]),
        torch.from_numpy(contract["starts_new_segment"]),
        cfg,
    )
    daily = result["daily_revenue"].cpu().numpy().astype(np.float64)
    cumulative = np.cumsum(daily)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    tail_count = max(1, math.ceil(0.10 * len(daily)))
    initial_soc = float(cfg.bess_energy_mwh * cfg.bess_init_soc_frac)
    actual_end = result["actual_soc_end_mwh"].cpu().numpy().astype(np.float64)
    plan_end = result["plan_soc_end_mwh"].cpu().numpy().astype(np.float64)
    resets = contract["starts_new_segment"]
    actual_start = np.empty_like(actual_end)
    plan_start = np.empty_like(plan_end)
    for index in range(len(daily)):
        if resets[index]:
            actual_start[index] = initial_soc
            plan_start[index] = initial_soc
        else:
            actual_start[index] = actual_end[index - 1]
            plan_start[index] = plan_end[index - 1]
    segment_starts = np.flatnonzero(resets)
    segment_ends = np.concatenate([
        segment_starts[1:] - 1,
        np.asarray([len(daily) - 1], dtype=np.int64),
    ])
    segment_reports = []
    for segment_index, (start, end) in enumerate(
        zip(segment_starts, segment_ends), start=1
    ):
        segment_reports.append({
            "segment": segment_index,
            "first_date": contract["dates"][int(start)],
            "last_date": contract["dates"][int(end)],
            "initial_actual_soc_mwh": initial_soc,
            "final_actual_soc_mwh": float(actual_end[end]),
            "actual_change_mwh": float(actual_end[end] - initial_soc),
            "initial_plan_soc_mwh": initial_soc,
            "final_plan_soc_mwh": float(plan_end[end]),
            "plan_change_mwh": float(plan_end[end] - initial_soc),
        })
    daily_map = {
        date: float(value) for date, value in zip(contract["dates"], daily)
    }
    return {
        "mean_daily_revenue": float(daily.mean()),
        "total_net_revenue": float(daily.sum()),
        "positive_day_rate": float(np.mean(daily > 0)),
        "lower_tail_10pct_mean": float(np.sort(daily)[:tail_count].mean()),
        "max_drawdown": float(np.max(peaks - cumulative, initial=0.0)),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            daily, cfg.bootstrap_samples, bootstrap_seed
        ),
        "days": int(len(daily)),
        "segments": int(np.sum(resets)),
        "revenue_components_mean_daily": {
            "da_leg": float(result["da_leg"].mean()),
            "rt_deviation_leg": float(result["rt_deviation_leg"].mean()),
            "degradation_cost": float(result["degradation_cost"].mean()),
            "deviation_penalty": float(result["deviation_penalty"].mean()),
        },
        "terminal_soc_report_mwh": {
            "monetized": False,
            "initial_soc_each_segment": initial_soc,
            "actual_first_day_start": float(actual_start[0]),
            "actual_final": float(actual_end[-1]),
            "actual_change_last_segment": float(actual_end[-1] - actual_start[
                np.flatnonzero(resets)[-1]
            ]),
            "plan_first_day_start": float(plan_start[0]),
            "plan_final": float(plan_end[-1]),
            "plan_change_last_segment": float(plan_end[-1] - plan_start[
                np.flatnonzero(resets)[-1]
            ]),
            "continuous_segments": segment_reports,
        },
        "forecast": {
            "da": _forecast_metrics(predictions["da"], contract["true_da"]),
            "rt_at_da": _forecast_metrics(
                predictions["rt_at_da"], contract["true_rt"]
            ),
            "rt_all_horizons": _forecast_metrics(
                predictions["rt"], contract["true_rt_windows"]
            ),
        },
        "daily_net_revenue": daily_map,
    }


def _paired_difference(candidate: dict, baseline: dict, cfg: PilotConfig,
                       seed: int) -> dict:
    if list(candidate["daily_net_revenue"]) != list(baseline["daily_net_revenue"]):
        raise ValueError("配对比较的验证日期不一致")
    values = np.asarray([
        candidate["daily_net_revenue"][date]
        - baseline["daily_net_revenue"][date]
        for date in baseline["daily_net_revenue"]
    ], dtype=np.float64)
    return {
        "reference": "transformer_huber",
        "mean_daily_revenue_difference": float(values.mean()),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            values, cfg.bootstrap_samples, seed
        ),
        "improved_day_rate": float(np.mean(values > 0)),
        "paired_days": int(len(values)),
    }


def _aggregate_seed_metrics(systems: dict, names: list[str]) -> dict:
    fields = (
        "mean_daily_revenue", "positive_day_rate", "lower_tail_10pct_mean",
        "max_drawdown",
    )
    return {
        "members": names,
        "seed_count": len(names),
        **{
            field: {
                "mean": float(np.mean([systems[name][field] for name in names])),
                "sample_std": float(np.std(
                    [systems[name][field] for name in names], ddof=1
                )) if len(names) > 1 else 0.0,
            }
            for field in fields
        },
    }


def _paired_seed_average_difference(
    systems: dict,
    names: list[str],
    baseline: dict,
    cfg: PilotConfig,
    seed: int,
) -> dict:
    dates = list(baseline["daily_net_revenue"])
    matrix = np.asarray([
        [systems[name]["daily_net_revenue"][date] for date in dates]
        for name in names
    ], dtype=np.float64)
    baseline_values = np.asarray([
        baseline["daily_net_revenue"][date] for date in dates
    ], dtype=np.float64)
    values = matrix.mean(axis=0) - baseline_values
    return {
        "reference": "transformer_huber",
        "members": names,
        "mean_daily_revenue_difference": float(values.mean()),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            values, cfg.bootstrap_samples, seed
        ),
        "improved_day_rate": float(np.mean(values > 0)),
        "paired_days": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/joint_decision_aware_v2.yaml"
    )
    parser.add_argument(
        "--models",
        default="seasonal,xgboost,gru,transformer_huber,transformer_v2",
    )
    parser.add_argument("--baseline-seeds", default="0")
    parser.add_argument("--gru-epochs", type=int, default=8)
    parser.add_argument("--gru-patience", type=int, default=3)
    parser.add_argument(
        "--v2-checkpoints",
        default=(
            "data/checkpoints/joint_decision_aware_v2_mean_seed0/"
            "best_validation_revenue.pt,"
            "data/checkpoints/joint_decision_aware_v2_mean_seed1/"
            "best_validation_revenue.pt,"
            "data/checkpoints/joint_decision_aware_v2_mean_seed2/"
            "best_validation_revenue.pt"
        ),
    )
    parser.add_argument(
        "--cache-dir", default="data/results/joint_baseline_cache_v1"
    )
    parser.add_argument(
        "--output", default="data/results/joint_system_baseline_comparison_v1.json"
    )
    parser.add_argument("--force-baselines", action="store_true")
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-validation-days", type=int, default=None)
    args = parser.parse_args()

    started_all = time.time()
    requested = [value.strip() for value in args.models.split(",") if value.strip()]
    allowed = {"seasonal", "xgboost", "gru", "transformer_huber", "transformer_v2"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"未知系统: {unknown}")
    baseline_seeds = [
        int(value.strip()) for value in args.baseline_seeds.split(",")
        if value.strip()
    ]
    if not baseline_seeds:
        raise ValueError("baseline-seeds不能为空")

    cfg = PilotConfig.from_yaml(args.config)
    source_paths = {
        "da": cfg.joint_da_checkpoint,
        "rt_at_da": cfg.joint_rt_at_da_checkpoint,
        "rt": cfg.joint_rt_checkpoint,
    }
    source_checkpoints = {
        task: _load_source_checkpoint(path) for task, path in source_paths.items()
    }
    reference_stats = source_checkpoints["da"]["norm_stats"]
    train_dataset, val_dataset, _, _ = build_joint_datasets(
        cfg, norm_stats=reference_stats
    )
    if args.max_train_days is not None:
        train_dataset = copy.copy(train_dataset)
        train_dataset.windows = train_dataset.windows[:args.max_train_days]
    contract = _truth_and_contract(val_dataset, args.max_validation_days)
    train_indices = _task_indices(train_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    systems = {}
    training = {}
    prediction_sources = {}

    print(json.dumps({
        "event": "comparison_contract",
        "device": str(device),
        "train_days": len(train_dataset),
        "validation_days": len(contract["dates"]),
        "validation_first": contract["dates"][0],
        "validation_last": contract["dates"][-1],
        "models": requested,
        "baseline_seeds": baseline_seeds,
        "deviation_penalty_enabled": cfg.use_deviation_penalty,
        "deviation_penalty_formula": (
            "2*abs(RT_price)*max(abs(u_RT_actual-u_DA)-0.03*abs(u_DA),0)"
        ),
        "coordination_mode": cfg.joint_coordination_mode,
        "da_signal_mode": cfg.joint_da_signal_mode,
        "same_182_day_contract": len(contract["dates"]) == 182,
    }, ensure_ascii=False), flush=True)

    def register(name: str, predictions: dict, seed: int, source: dict):
        metrics = _settle_and_summarize(predictions, contract, cfg, seed + 7000)
        systems[name] = metrics
        prediction_sources[name] = source
        print(json.dumps({
            "event": "system_complete", "system": name,
            "mean_daily_revenue": metrics["mean_daily_revenue"],
            "days": metrics["days"],
        }, ensure_ascii=False), flush=True)

    if "seasonal" in requested:
        register(
            "seasonal", _seasonal_predictions(val_dataset, contract), 0,
            {"type": "weekly_seasonal", "lag_hours": 168},
        )

    if "xgboost" in requested:
        for seed in baseline_seeds:
            predictions, details = _xgboost_predictions(
                train_dataset, val_dataset, train_indices, contract, cfg, seed,
                cache_dir, args.force_baselines,
            )
            name = f"xgboost_seed{seed}"
            register(name, predictions, seed, {
                "type": "three_independent_xgboost_models", "seed": seed,
            })
            training[name] = details

    if "gru" in requested:
        for seed in baseline_seeds:
            predictions, details = _gru_predictions(
                train_dataset, val_dataset, train_indices, contract, cfg, seed,
                args.gru_epochs, args.gru_patience, device, cache_dir,
                args.force_baselines,
            )
            name = f"gru_seed{seed}"
            register(name, predictions, seed, {
                "type": "three_independent_gru_models", "seed": seed,
            })
            training[name] = details

    if "transformer_huber" in requested:
        models = _build_transformers(source_checkpoints, device)
        predictions = _predict_transformer_system(
            models, val_dataset, contract, device
        )
        register("transformer_huber", predictions, 0, {
            "type": "three_frozen_pure_huber_transformers",
            "checkpoints": {
                task: {"path": path, "sha256": _sha256(path)}
                for task, path in source_paths.items()
            },
        })
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v2_names = []
    if "transformer_v2" in requested:
        v2_paths = [
            value.strip() for value in args.v2_checkpoints.split(",")
            if value.strip()
        ]
        for path_index, path_text in enumerate(v2_paths):
            path = Path(path_text)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            if "model_states" not in bundle:
                raise ValueError(f"V2 checkpoint结构不完整: {path}")
            seed = int(bundle.get("experiment_config", {}).get("seed", path_index))
            (
                bundle_source_paths,
                bundle_source_checkpoints,
                bundle_source_audit,
            ) = _load_bundle_source_checkpoints(bundle, path)
            for task, source_checkpoint in bundle_source_checkpoints.items():
                _check_norm_stats(
                    reference_stats,
                    source_checkpoint["norm_stats"],
                    f"{path.name}:{task}",
                )
            models = _build_transformers(
                bundle_source_checkpoints, device, bundle["model_states"]
            )
            predictions = _predict_transformer_system(
                models, val_dataset, contract, device
            )
            training_cfg = bundle.get("experiment_config", {})
            name = f"transformer_v2_seed{seed}"
            register(name, predictions, seed, {
                "type": "three_jointly_finetuned_independent_transformers",
                "path": str(path),
                "sha256": _sha256(path),
                "best_epoch": bundle.get("best_epoch"),
                "architecture_sources": bundle_source_audit,
                "training_contract": {
                    "deviation_penalty_enabled": training_cfg.get(
                        "use_deviation_penalty", False
                    ),
                    "coordination_mode": training_cfg.get(
                        "joint_coordination_mode", "rt_only (legacy default)"
                    ),
                    "da_signal_mode": training_cfg.get(
                        "joint_da_signal_mode", "spread (legacy default)"
                    ),
                    "validation_bounds": [
                        training_cfg.get("split_val_start"),
                        training_cfg.get("split_val_end"),
                    ],
                },
            })
            v2_names.append(name)
            del (
                models,
                bundle,
                bundle_source_paths,
                bundle_source_checkpoints,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    paired = {}
    if "transformer_huber" in systems:
        for index, (name, metrics) in enumerate(systems.items()):
            if name != "transformer_huber":
                paired[name] = _paired_difference(
                    metrics, systems["transformer_huber"], cfg, 9000 + index
                )

    aggregates = {}
    xgb_names = [f"xgboost_seed{seed}" for seed in baseline_seeds
                 if f"xgboost_seed{seed}" in systems]
    gru_names = [f"gru_seed{seed}" for seed in baseline_seeds
                 if f"gru_seed{seed}" in systems]
    if xgb_names:
        aggregates["xgboost"] = _aggregate_seed_metrics(systems, xgb_names)
    if gru_names:
        aggregates["gru"] = _aggregate_seed_metrics(systems, gru_names)
    if v2_names:
        aggregates["transformer_v2"] = _aggregate_seed_metrics(systems, v2_names)
    aggregate_paired = {}
    if "transformer_huber" in systems:
        for label, names in (
            ("xgboost", xgb_names),
            ("gru", gru_names),
            ("transformer_v2", v2_names),
        ):
            if names:
                aggregate_paired[label] = _paired_seed_average_difference(
                    systems, names, systems["transformer_huber"], cfg, 12000
                )

    report = {
        "experiment": "joint_system_baseline_comparison_v1",
        "status": "formal" if (
            len(train_dataset) == 1808 and len(contract["dates"]) == 182
        ) else "diagnostic_subset",
        "primary_metric": "validation_full_dual_mean_daily_net_revenue_usd",
        "selection_note": (
            "本表只使用验证集；2026 report-only没有读取、没有参与训练或排名。"
        ),
        "contract": {
            "market": cfg.market,
            "node": cfg.node,
            "train_days": len(train_dataset),
            "validation_days": len(contract["dates"]),
            "validation_dates": contract["dates"],
            "same_true_da_rt": True,
            "initial_soc_mwh_each_continuous_segment": float(
                cfg.bess_energy_mwh * cfg.bess_init_soc_frac
            ),
            "soc_rule": "连续日期继承上一日期末SOC；只在真实数据缺口的新段重置",
            "hard_topk": {
                "da_charge": cfg.topk_k_charge,
                "da_discharge": cfg.topk_k_discharge,
                "rt_charge": cfg.rt_topk_k_charge,
                "rt_discharge": cfg.rt_topk_k_discharge,
                "spread_threshold": cfg.resolved_spread_threshold,
            },
            "bess": {
                "power_mw": cfg.bess_power_mw,
                "energy_mwh": cfg.bess_energy_mwh,
                "eta": cfg.bess_eta,
                "soc_min_mwh": cfg.bess_soc_min,
                "soc_max_mwh": cfg.bess_soc_max,
                "degradation_usd_per_mwh": cfg.bess_kappa,
                "daily_discharge_limit_mwh": cfg.bess_e_cyc,
            },
            "deviation_penalty_enabled": cfg.use_deviation_penalty,
            "deviation_penalty": {
                "tolerance_fraction_of_abs_da_position": 0.03,
                "excess_multiplier_abs_rt_price": 2.0,
                "formula": (
                    "2*abs(RT_price)*max("
                    "abs(u_RT_actual-u_DA)-0.03*abs(u_DA),0)"
                ),
            },
            "coordination_mode": cfg.joint_coordination_mode,
            "da_signal_mode": cfg.joint_da_signal_mode,
            "settlement_formula": (
                "DA_price*u_DA + RT_price*(u_RT_actual-u_DA) "
                "- degradation - deviation_penalty"
            ),
            "terminal_soc": "只报告MWh，不折算货币",
        },
        "prediction_sources": prediction_sources,
        "systems": systems,
        "seed_aggregates": aggregates,
        "paired_vs_transformer_huber": paired,
        "seed_average_paired_vs_transformer_huber": aggregate_paired,
        "baseline_training": training,
        "runtime": {
            "device": str(device),
            "wall_time_seconds": round(time.time() - started_all, 2),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "saved", "path": str(output),
        "seconds": report["runtime"]["wall_time_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
