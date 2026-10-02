#!/usr/bin/env python
"""公平训练DA或RT-at-DA融合候选，并按冻结完整双结算收益选checkpoint。"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.dataset_rt_da import build_rt_at_da_datasets  # noqa: E402
from decision_aware.model_da import (  # noqa: E402
    DecisionAwareDAForecaster,
    canonical_da_fusion_mode,
)
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


SPLIT_FIELDS = (
    "split_train_start", "split_train_end",
    "split_val_start", "split_val_end",
    "split_test_start", "split_test_end",
)
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
        key: value for key, value in checkpoint["config"].items() if key in fields
    })


def _eval_config(model_cfg: PilotConfig, experiment_cfg: PilotConfig) -> PilotConfig:
    overrides = {name: getattr(experiment_cfg, name) for name in SPLIT_FIELDS}
    overrides.update({"val_stride": 1, "eval_stride": 1})
    return replace(model_cfg, **overrides)


def _check_physical(reference: PilotConfig, candidate: PilotConfig, label: str) -> None:
    mismatches = {
        name: [getattr(reference, name), getattr(candidate, name)]
        for name in PHYSICAL_FIELDS
        if getattr(reference, name) != getattr(candidate, name)
    }
    if mismatches:
        raise ValueError(f"{label}物理参数与融合消融配置不一致: {mismatches}")


def _check_norm_stats(dataset, checkpoint: dict, label: str) -> None:
    saved = checkpoint.get("norm_stats")
    if saved is None:
        raise ValueError(f"{label} checkpoint缺少训练集norm_stats")
    for stream, statistics in dataset.norm_stats.items():
        if stream not in saved:
            raise ValueError(f"{label} checkpoint缺少{stream}归一化统计")
        for statistic in ("mean", "std"):
            left = np.asarray(statistics[statistic], dtype=np.float64)
            right = np.asarray(saved[stream][statistic], dtype=np.float64)
            if not np.allclose(left, right, rtol=1e-6, atol=1e-6):
                raise ValueError(f"{label}的{stream}.{statistic}与当前训练数据不一致")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _predict(model, dataset, output_key: str, device: torch.device,
             batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    model.eval()
    amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(batch)[output_key]
            values.append(output.float().cpu().numpy())
    return np.concatenate(values)


def _da_mapping(dataset, predictions: np.ndarray) -> dict:
    if len(dataset) != len(predictions):
        raise ValueError("日预测数量与DA样本数量不一致")
    return {
        window.delivery_date: predictions[index]
        for index, window in enumerate(dataset.windows)
    }


def _rt_realized_and_times(dataset):
    positions = [window.issue_position for window in dataset.windows]
    return (
        torch.from_numpy(dataset.price_da[positions].copy()),
        torch.from_numpy(dataset.price_rt[positions].copy()),
        [dataset.sample_time_contract(i)["executed_target"] for i in range(len(dataset))],
    )


def _forecast_diagnostics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    peak_true = target.argmax(axis=1)
    valley_true = target.argmin(axis=1)
    peak_pred = prediction.argmax(axis=1)
    valley_pred = prediction.argmin(axis=1)
    pair_scores = []
    for pred_day, true_day in zip(prediction, target):
        left, right = np.triu_indices(len(true_day), k=1)
        true_order = np.sign(true_day[left] - true_day[right])
        valid = true_order != 0
        if valid.any():
            pred_order = np.sign(pred_day[left] - pred_day[right])
            pair_scores.append(float(np.mean(pred_order[valid] == true_order[valid])))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "peak_hour_accuracy": float(np.mean(peak_pred == peak_true)),
        "valley_hour_accuracy": float(np.mean(valley_pred == valley_true)),
        "pairwise_order_accuracy": float(np.mean(pair_scores)),
        "days": int(len(target)),
    }


def _block_bootstrap_ci(values: np.ndarray, samples: int, seed: int,
                        block: int = 7) -> list[float]:
    if len(values) == 0:
        raise ValueError("bootstrap输入不能为空")
    if len(values) == 1 or samples < 1:
        mean = float(values.mean())
        return [mean, mean]
    rng = np.random.default_rng(seed)
    block = min(block, len(values))
    starts_max = len(values) - block + 1
    n_blocks = math.ceil(len(values) / block)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sampled = np.concatenate([values[start:start + block] for start in starts])
        means[index] = sampled[:len(values)].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _compact_dual(report: dict, bootstrap_samples: int, seed: int) -> dict:
    daily = {key: float(value["net_revenue"]) for key, value in report["daily"].items()}
    values = np.asarray(list(daily.values()), dtype=np.float64)
    tail_count = max(1, math.ceil(0.1 * len(values)))
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    drawdown = running_peak - cumulative
    compact = {
        "mean_daily_revenue": float(report["mean_daily_revenue"]),
        "positive_day_rate": float(report["positive_day_rate"]),
        "lower_tail_10pct_mean": float(np.sort(values)[:tail_count].mean()),
        "max_drawdown": float(drawdown.max(initial=0.0)),
        "weekly_block_bootstrap_95ci": _block_bootstrap_ci(
            values, bootstrap_samples, seed
        ),
        "days": int(report["days"]),
        "segments": int(report["segments"]),
        "revenue_components": report["revenue_components"],
        "plan_feasibility": report["plan_feasibility"],
        "actual_state": report["actual_state"],
        "daily_net_revenue": daily,
    }
    if "per_reference_mean_daily_revenue" in report:
        compact["per_reference_mean_daily_revenue"] = report[
            "per_reference_mean_daily_revenue"
        ]
        compact["reference_panel_labels"] = report[
            "reference_panel_labels"
        ]
    return compact


class FrozenDualEvaluator:
    """只加载一次冻结RT预测，之后每个DA epoch只替换DA预测。"""

    def __init__(self, cfg: PilotConfig, da_datasets: dict,
                 device: torch.device):
        if not cfg.frozen_rt_at_da_checkpoint or not cfg.frozen_rt_checkpoint:
            raise ValueError("DA消融必须配置冻结RT-at-DA和滚动RT checkpoint")
        self.cfg = cfg
        self.da_datasets = da_datasets
        self.device = device
        self.fixed = {}

        rt_da_checkpoint = _load_checkpoint(cfg.frozen_rt_at_da_checkpoint)
        rt_da_model_cfg = _config_from_checkpoint(rt_da_checkpoint)
        _check_physical(cfg, rt_da_model_cfg, "RT-at-DA")
        rt_da_data_cfg = _eval_config(rt_da_model_cfg, cfg)
        rt_da_train, rt_da_val, rt_da_test, _ = build_rt_at_da_datasets(rt_da_data_cfg)
        _check_norm_stats(rt_da_train, rt_da_checkpoint, "RT-at-DA")
        rt_da_model = DecisionAwareRTAtDAForecaster(rt_da_model_cfg).to(device)
        rt_da_model.load_state_dict(rt_da_checkpoint["model_state"])
        self.rt_da_datasets = {"validation": rt_da_val, "test": rt_da_test}
        for split, dataset in self.rt_da_datasets.items():
            self.fixed.setdefault(split, {})["rt_at_da"] = _predict(
                rt_da_model, dataset, "p_rt_at_da", device, batch_size=64
            )
        del rt_da_model

        rt_checkpoint = _load_checkpoint(cfg.frozen_rt_checkpoint)
        rt_model_cfg = _config_from_checkpoint(rt_checkpoint)
        _check_physical(cfg, rt_model_cfg, "滚动RT")
        rt_data_cfg = _eval_config(rt_model_cfg, cfg)
        rt_train, rt_val, rt_test, _ = build_rt_datasets(rt_data_cfg)
        _check_norm_stats(rt_train, rt_checkpoint, "滚动RT")
        rt_model = DecisionAwareRTForecaster(rt_model_cfg).to(device)
        rt_model.load_state_dict(rt_checkpoint["model_state"])
        self.rt_datasets = {"validation": rt_val, "test": rt_test}
        for split, dataset in self.rt_datasets.items():
            self.fixed.setdefault(split, {})["rt"] = _predict(
                rt_model, dataset, "p_rt", device, batch_size=128
            )
            realized_da, realized_rt, timestamps = _rt_realized_and_times(dataset)
            self.fixed[split]["realized_da"] = realized_da
            self.fixed[split]["realized_rt"] = realized_rt
            self.fixed[split]["timestamps"] = timestamps
        del rt_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for split in ("validation", "test"):
            da_dates = [window.delivery_date for window in da_datasets[split].windows]
            rt_da_dates = [
                window.delivery_date for window in self.rt_da_datasets[split].windows
            ]
            if da_dates != rt_da_dates:
                raise ValueError(f"{split}的DA和RT-at-DA交付日没有逐日对齐")

    def evaluate(self, split: str, da_prediction: np.ndarray) -> dict:
        fixed = self.fixed[split]
        return locked_dual_backtest(
            _da_mapping(self.da_datasets[split], da_prediction),
            fixed["rt"],
            fixed["realized_da"],
            fixed["realized_rt"],
            fixed["timestamps"],
            self.cfg,
            da_k_charge=self.cfg.topk_k_charge,
            da_k_discharge=self.cfg.topk_k_discharge,
            rt_k_charge=1,
            rt_k_discharge=1,
            coordination_mode=self.cfg.frozen_coordination_mode,
            rt_at_da_price_forecasts=_da_mapping(
                self.rt_da_datasets[split], fixed["rt_at_da"]
            ),
        )


def _mean_dual_reports(labeled_reports: list[tuple[str, dict]]) -> dict:
    """对冻结参考面板逐日取均值，供候选模型公平选模。"""
    if not labeled_reports:
        raise ValueError("冻结DA参考报告不能为空")
    reports = [report for _, report in labeled_reports]
    date_sets = [set(report["daily"]) for report in reports]
    if any(dates != date_sets[0] for dates in date_sets[1:]):
        raise ValueError("冻结DA参考报告的交付日不一致")
    dates = sorted(date_sets[0])
    daily = {}
    for date in dates:
        rows = [report["daily"][date] for report in reports]
        daily[date] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0]
        }
    daily_values = list(daily.values())
    components = {
        "da_leg": sum(row["da_leg"] for row in daily_values),
        "rt_deviation_leg": sum(row["rt_deviation_leg"] for row in daily_values),
        "degradation_cost": sum(row["degradation_cost"] for row in daily_values),
        "deviation_penalty": sum(row["deviation_penalty"] for row in daily_values),
    }
    total_revenue = sum(row["net_revenue"] for row in daily_values)
    first = reports[0]
    action_digests = {
        json.dumps(report["actual_state"]["action_digest"], sort_keys=True)
        for report in reports
    }
    if len(action_digests) != 1:
        raise ValueError("rt_only下冻结DA参考不应改变滚动RT实际动作")
    return {
        "days": len(daily_values),
        "segments": first["segments"],
        "mean_daily_revenue": total_revenue / len(daily_values),
        "positive_day_rate": float(np.mean([
            row["net_revenue"] > 0 for row in daily_values
        ])),
        "revenue_components": components,
        "plan_feasibility": {
            "reference_mean_total_clipped_mwh": float(np.mean([
                report["plan_feasibility"]["total_clipped_mwh"]
                for report in reports
            ])),
            "reference_mean_clipped_days": float(np.mean([
                report["plan_feasibility"]["clipped_days"]
                for report in reports
            ])),
            "reference_count": len(reports),
        },
        "actual_state": first["actual_state"],
        "daily": daily,
        "reference_panel_labels": [label for label, _ in labeled_reports],
        "per_reference_mean_daily_revenue": {
            label: float(report["mean_daily_revenue"])
            for label, report in labeled_reports
        },
    }


class FrozenRTAtDADualEvaluator:
    """冻结三个DA-F1参考和滚动RT，只替换RT-at-DA预测。"""

    def __init__(self, cfg: PilotConfig, rt_da_datasets: dict,
                 device: torch.device):
        if not cfg.frozen_da_checkpoints or not cfg.frozen_rt_checkpoint:
            raise ValueError("RT-at-DA消融必须配置冻结DA面板和滚动RT checkpoint")
        if cfg.frozen_coordination_mode != "rt_only":
            raise ValueError("多DA参考面板当前只允许冻结rt_only实际动作")
        self.cfg = cfg
        self.rt_da_datasets = rt_da_datasets
        self.fixed = {"validation": {}, "test": {}}
        self.da_references = []

        for path in cfg.frozen_da_checkpoints:
            checkpoint = _load_checkpoint(path)
            model_cfg = _config_from_checkpoint(checkpoint)
            _check_physical(cfg, model_cfg, f"冻结DA {path}")
            data_cfg = _eval_config(model_cfg, cfg)
            train_ds, val_ds, test_ds, _ = build_da_datasets(data_cfg)
            _check_norm_stats(train_ds, checkpoint, f"冻结DA {path}")
            model = DecisionAwareDAForecaster(model_cfg).to(device)
            model.load_state_dict(checkpoint["model_state"])
            datasets = {"validation": val_ds, "test": test_ds}
            predictions = {
                split: _predict(model, dataset, "p_da", device, batch_size=64)
                for split, dataset in datasets.items()
            }
            self.da_references.append({
                "checkpoint": str(path),
                "datasets": datasets,
                "predictions": predictions,
            })
            del model

        rt_checkpoint = _load_checkpoint(cfg.frozen_rt_checkpoint)
        rt_model_cfg = _config_from_checkpoint(rt_checkpoint)
        _check_physical(cfg, rt_model_cfg, "滚动RT")
        rt_data_cfg = _eval_config(rt_model_cfg, cfg)
        rt_train, rt_val, rt_test, _ = build_rt_datasets(rt_data_cfg)
        _check_norm_stats(rt_train, rt_checkpoint, "滚动RT")
        rt_model = DecisionAwareRTForecaster(rt_model_cfg).to(device)
        rt_model.load_state_dict(rt_checkpoint["model_state"])
        self.rt_datasets = {"validation": rt_val, "test": rt_test}
        for split, dataset in self.rt_datasets.items():
            self.fixed[split]["rt"] = _predict(
                rt_model, dataset, "p_rt", device, batch_size=128
            )
            realized_da, realized_rt, timestamps = _rt_realized_and_times(dataset)
            self.fixed[split]["realized_da"] = realized_da
            self.fixed[split]["realized_rt"] = realized_rt
            self.fixed[split]["timestamps"] = timestamps
        del rt_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for split in ("validation", "test"):
            candidate_dates = [
                window.delivery_date for window in rt_da_datasets[split].windows
            ]
            for reference in self.da_references:
                da_dates = [
                    window.delivery_date
                    for window in reference["datasets"][split].windows
                ]
                if da_dates != candidate_dates:
                    raise ValueError(
                        f"{split}的冻结DA与RT-at-DA候选交付日没有逐日对齐"
                    )

    def evaluate(self, split: str, rt_da_prediction: np.ndarray) -> dict:
        fixed = self.fixed[split]
        rt_da_mapping = _da_mapping(
            self.rt_da_datasets[split], rt_da_prediction
        )
        labeled_reports = []
        for reference in self.da_references:
            report = locked_dual_backtest(
                _da_mapping(
                    reference["datasets"][split],
                    reference["predictions"][split],
                ),
                fixed["rt"],
                fixed["realized_da"],
                fixed["realized_rt"],
                fixed["timestamps"],
                self.cfg,
                da_k_charge=self.cfg.topk_k_charge,
                da_k_discharge=self.cfg.topk_k_discharge,
                rt_k_charge=1,
                rt_k_discharge=1,
                coordination_mode=self.cfg.frozen_coordination_mode,
                rt_at_da_price_forecasts=rt_da_mapping,
            )
            labeled_reports.append((reference["checkpoint"], report))
        return _mean_dual_reports(labeled_reports)


class FrozenRollingRTDualEvaluator:
    """冻结DA-F1和RT-at-DA-F0参考面板，只替换滚动RT预测。"""

    def __init__(self, cfg: PilotConfig, rt_datasets: dict,
                 device: torch.device):
        if (
            not cfg.frozen_da_checkpoints
            or not cfg.frozen_rt_at_da_checkpoints
        ):
            raise ValueError("滚动RT消融必须配置冻结DA和RT-at-DA参考面板")
        if cfg.frozen_coordination_mode != "rt_only":
            raise ValueError("多参考面板当前只允许冻结rt_only协调方式")
        self.cfg = cfg
        self.rt_datasets = rt_datasets
        self.da_references = []
        self.rt_da_references = []
        self.realized = {}

        for path in cfg.frozen_da_checkpoints:
            checkpoint = _load_checkpoint(path)
            model_cfg = _config_from_checkpoint(checkpoint)
            _check_physical(cfg, model_cfg, f"冻结DA {path}")
            data_cfg = _eval_config(model_cfg, cfg)
            train_ds, val_ds, test_ds, _ = build_da_datasets(data_cfg)
            _check_norm_stats(train_ds, checkpoint, f"冻结DA {path}")
            model = DecisionAwareDAForecaster(model_cfg).to(device)
            model.load_state_dict(checkpoint["model_state"])
            datasets = {"validation": val_ds, "test": test_ds}
            self.da_references.append({
                "checkpoint": str(path),
                "datasets": datasets,
                "predictions": {
                    split: _predict(model, dataset, "p_da", device, batch_size=64)
                    for split, dataset in datasets.items()
                },
            })
            del model

        for path in cfg.frozen_rt_at_da_checkpoints:
            checkpoint = _load_checkpoint(path)
            model_cfg = _config_from_checkpoint(checkpoint)
            _check_physical(cfg, model_cfg, f"冻结RT-at-DA {path}")
            data_cfg = _eval_config(model_cfg, cfg)
            train_ds, val_ds, test_ds, _ = build_rt_at_da_datasets(data_cfg)
            _check_norm_stats(train_ds, checkpoint, f"冻结RT-at-DA {path}")
            model = DecisionAwareRTAtDAForecaster(model_cfg).to(device)
            model.load_state_dict(checkpoint["model_state"])
            datasets = {"validation": val_ds, "test": test_ds}
            self.rt_da_references.append({
                "checkpoint": str(path),
                "datasets": datasets,
                "predictions": {
                    split: _predict(
                        model, dataset, "p_rt_at_da", device, batch_size=64
                    )
                    for split, dataset in datasets.items()
                },
            })
            del model

        for split, dataset in rt_datasets.items():
            realized_da, realized_rt, timestamps = _rt_realized_and_times(dataset)
            self.realized[split] = {
                "da": realized_da,
                "rt": realized_rt,
                "timestamps": timestamps,
            }
            expected_dates = [
                window.delivery_date
                for window in self.da_references[0]["datasets"][split].windows
            ]
            for reference in self.da_references[1:]:
                dates = [
                    window.delivery_date
                    for window in reference["datasets"][split].windows
                ]
                if dates != expected_dates:
                    raise ValueError(f"{split}的冻结DA参考交付日不一致")
            for reference in self.rt_da_references:
                dates = [
                    window.delivery_date
                    for window in reference["datasets"][split].windows
                ]
                if dates != expected_dates:
                    raise ValueError(
                        f"{split}的冻结DA与RT-at-DA参考交付日不一致"
                    )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def evaluate(self, split: str, rt_prediction: np.ndarray) -> dict:
        realized = self.realized[split]
        labeled_reports = []
        for da_reference in self.da_references:
            da_mapping = _da_mapping(
                da_reference["datasets"][split],
                da_reference["predictions"][split],
            )
            for rt_da_reference in self.rt_da_references:
                rt_da_mapping = _da_mapping(
                    rt_da_reference["datasets"][split],
                    rt_da_reference["predictions"][split],
                )
                report = locked_dual_backtest(
                    da_mapping,
                    rt_prediction,
                    realized["da"],
                    realized["rt"],
                    realized["timestamps"],
                    self.cfg,
                    da_k_charge=self.cfg.topk_k_charge,
                    da_k_discharge=self.cfg.topk_k_discharge,
                    rt_k_charge=1,
                    rt_k_discharge=1,
                    coordination_mode=self.cfg.frozen_coordination_mode,
                    rt_at_da_price_forecasts=rt_da_mapping,
                )
                label = (
                    f"DA={da_reference['checkpoint']}|"
                    f"RT-at-DA={rt_da_reference['checkpoint']}"
                )
                labeled_reports.append((label, report))
        return _mean_dual_reports(labeled_reports)


def _evaluate_forecaster(model, dataset, output_key: str, target_key: str,
                         device: torch.device, batch_size: int):
    prediction = _predict(model, dataset, output_key, device, batch_size)
    target = np.stack([
        dataset[index][target_key].numpy() for index in range(len(dataset))
    ])
    return prediction, target, _forecast_diagnostics(prediction, target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("da", "rt_at_da", "rt"), default="da"
    )
    parser.add_argument(
        "--config", default="configs/decision_aware/da_fusion_ablation_v1.yaml"
    )
    parser.add_argument(
        "--fusion-mode", required=True, choices=("F0", "F1", "F1b", "F2")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--dim-ff", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PilotConfig.from_yaml(args.config)
    fusion_implementation = canonical_da_fusion_mode(args.fusion_mode)
    if args.task == "da":
        cfg.da_fusion_mode = fusion_implementation
    elif args.task == "rt_at_da":
        cfg.rt_at_da_fusion_mode = fusion_implementation
    else:
        cfg.rt_fusion_mode = fusion_implementation
    cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.dim_ff is not None:
        cfg.dim_ff = args.dim_ff
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    elif args.fusion_mode == "F2":
        cfg.batch_size = 8
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    elif args.fusion_mode == "F2":
        cfg.grad_accum_steps = 4
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps必须至少为1")
    if not (
        cfg.alpha0 == cfg.alpha_end == 1.0
        and cfg.beta0 == cfg.beta_end == 0.0
        and not cfg.use_mse_loss
    ):
        raise ValueError("F0/F1/F2正式消融强制使用固定纯Huber损失")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    if args.task == "da":
        train_ds, val_ds, test_ds, _ = build_da_datasets(cfg)
        model = DecisionAwareDAForecaster(cfg).to(device)
        output_key = "p_da"
        target_key = "price_da_tgt"
        frozen = FrozenDualEvaluator(
            cfg, {"validation": val_ds, "test": test_ds}, device
        )
        experiment_name = "DA F0/F1/F2 fusion ablation"
        result_prefix = "da_fusion"
    elif args.task == "rt_at_da":
        train_ds, val_ds, test_ds, _ = build_rt_at_da_datasets(cfg)
        model = DecisionAwareRTAtDAForecaster(cfg).to(device)
        output_key = "p_rt_at_da"
        target_key = "price_rt_at_da_tgt"
        frozen = FrozenRTAtDADualEvaluator(
            cfg, {"validation": val_ds, "test": test_ds}, device
        )
        experiment_name = "RT-at-DA F0/F1 confirmation ablation"
        result_prefix = "rt_at_da_fusion"
    else:
        train_ds, val_ds, test_ds, _ = build_rt_datasets(cfg)
        model = DecisionAwareRTForecaster(cfg).to(device)
        output_key = "p_rt"
        target_key = "price_rt_tgt"
        frozen = FrozenRollingRTDualEvaluator(
            cfg, {"validation": val_ds, "test": test_ds}, device
        )
        experiment_name = "Rolling RT F0/F1 confirmation ablation"
        result_prefix = "rt_fusion"
    datasets = {"validation": val_ds, "test": test_ds}

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_data = train_ds
    if args.max_train is not None and args.max_train < len(train_ds):
        indices = np.linspace(0, len(train_ds) - 1, args.max_train, dtype=int).tolist()
        train_data = Subset(train_ds, indices)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = cfg.batch_size if args.fusion_mode == "F2" else 64
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_revenue = -float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            group_start = (step // cfg.grad_accum_steps) * cfg.grad_accum_steps
            group_size = min(cfg.grad_accum_steps, batches - group_start)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                prediction = model(batch)[output_key]
                raw_loss = F.huber_loss(
                    prediction.float(), batch[target_key].float(),
                    delta=cfg.huber_delta, reduction="mean",
                )
                loss = raw_loss / cfg.pred_scale
            scaler.scale(loss / group_size).backward()
            losses.append(float(raw_loss.detach().cpu()))
            end_group = (step + 1) % cfg.grad_accum_steps == 0 or step + 1 == batches
            if end_group:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        val_prediction, _, val_forecast = _evaluate_forecaster(
            model, val_ds, output_key, target_key, device, eval_batch_size
        )
        val_dual = frozen.evaluate("validation", val_prediction)
        val_revenue = float(val_dual["mean_daily_revenue"])
        row = {
            "epoch": epoch + 1,
            "train_huber": float(np.mean(losses)),
            "validation_mae": val_forecast["mae"],
            "validation_full_dual_revenue": val_revenue,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_revenue > best_revenue:
            best_revenue = val_revenue
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生可保存的状态")
    model.load_state_dict(best_state)
    model.to(device)

    final = {}
    for split, dataset in datasets.items():
        prediction, target, forecast = _evaluate_forecaster(
            model, dataset, output_key, target_key, device, eval_batch_size
        )
        dual = frozen.evaluate(split, prediction)
        final[split] = {
            "forecast": forecast,
            "full_dual": _compact_dual(
                dual, cfg.bootstrap_samples, cfg.seed + (0 if split == "validation" else 10_000)
            ),
        }

    checkpoint = (
        Path(cfg.checkpoint_dir)
        / f"{args.fusion_mode.lower()}_seed{cfg.seed}"
        / "pilot_LZ_LCRA_best_full_dual.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "config": cfg.to_dict(),
        "norm_stats": train_ds.norm_stats,
        "selection_metric": "frozen_full_dual_validation_mean_daily_revenue",
        "best_validation_revenue": best_revenue,
        "best_epoch": best_epoch,
        "fusion_mode": args.fusion_mode,
        "task": args.task,
    }, checkpoint)

    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    report = {
        "experiment": experiment_name,
        "task": args.task,
        "fusion_mode": args.fusion_mode,
        "fusion_implementation": fusion_implementation,
        "seed": cfg.seed,
        "device": str(device),
        "loss": "pure Huber only; alpha=1 beta=0",
        "selection_metric": "frozen full-dual validation mean daily revenue",
        "frozen_system": {
            "da_checkpoints": (
                cfg.frozen_da_checkpoints
                if args.task in {"rt_at_da", "rt"} else []
            ),
            "rt_at_da_checkpoint": (
                cfg.frozen_rt_at_da_checkpoint
                if args.task == "da"
                else ("candidate" if args.task == "rt_at_da" else "")
            ),
            "rt_at_da_checkpoints": (
                cfg.frozen_rt_at_da_checkpoints if args.task == "rt" else []
            ),
            "rolling_rt_checkpoint": (
                "candidate" if args.task == "rt" else cfg.frozen_rt_checkpoint
            ),
            "coordination_mode": cfg.frozen_coordination_mode,
        },
        "split": {
            name: list(cfg.split_bounds(name)) for name in ("train", "val", "test")
        },
        "test_status": "report-only; dates were exposed by earlier experiments",
        "samples": {
            "train": len(train_data), "validation": len(val_ds), "test": len(test_ds)
        },
        "parameter_count": parameter_count,
        "model_capacity": {
            "d_model": cfg.d_model,
            "dim_ff": cfg.dim_ff,
            "n_layers_enc": cfg.n_layers_enc,
            "n_layers_fusion": cfg.n_layers_fusion,
        },
        "learning_rate": cfg.lr,
        "physical_batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "validation": final["validation"],
        "test_report_only": final["test"],
        "compute": {
            "seconds": round(time.time() - started, 2),
            "peak_cuda_memory_mb": peak_memory_mb,
        },
        "checkpoint": str(checkpoint),
    }
    output = Path(args.output or (
        f"data/results/{result_prefix}_{args.fusion_mode.lower()}_seed{cfg.seed}.json"
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "fusion_mode": args.fusion_mode,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "validation_revenue": final["validation"]["full_dual"]["mean_daily_revenue"],
        "test_report_only_revenue": final["test"]["full_dual"]["mean_daily_revenue"],
        "checkpoint": str(checkpoint),
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
