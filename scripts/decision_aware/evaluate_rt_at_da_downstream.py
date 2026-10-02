"""冻结DA与滚动RT，确认RT-at-DA F1b能否向完整系统传播收益。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from scripts.decision_aware.train_da_fusion_ablation import (  # noqa: E402
    FrozenRollingRTDualEvaluator,
    _block_bootstrap_ci,
    _check_norm_stats,
    _compact_dual,
    _config_from_checkpoint,
    _eval_config,
    _load_checkpoint,
    _predict,
)


SEEDS = (0, 1, 2)
MODE_PATHS = {
    "F0": [
        f"data/checkpoints/rt_at_da_fusion_ablation/f0_seed{seed}/"
        "pilot_LZ_LCRA_best_full_dual.pt"
        for seed in SEEDS
    ],
    "F1b": [
        f"data/checkpoints/rt_at_da_fusion_ablation/f1b_seed{seed}/"
        "pilot_LZ_LCRA_best_full_dual.pt"
        for seed in SEEDS
    ],
}
DA_PATHS = [
    f"data/checkpoints/da_fusion_ablation/f1_seed{seed}/"
    "pilot_LZ_LCRA_best_full_dual.pt"
    for seed in SEEDS
]
ROLLING_RT_PATHS = [
    f"data/checkpoints/rt_fusion_ablation/f0_seed{seed}/"
    "pilot_LZ_LCRA_best_full_dual.pt"
    for seed in SEEDS
]


def _average_rolling_rt_reports(labeled_reports: list[tuple[str, dict]]) -> dict:
    """跨滚动RT seed逐日平均；允许不同seed产生不同实际动作。"""
    if not labeled_reports:
        raise ValueError("滚动RT参考报告不能为空")
    reports = [report for _, report in labeled_reports]
    date_sets = [set(report["daily"]) for report in reports]
    if any(dates != date_sets[0] for dates in date_sets[1:]):
        raise ValueError("不同滚动RT seed的回测日期不一致")

    daily = {}
    for date in sorted(date_sets[0]):
        rows = [report["daily"][date] for report in reports]
        daily[date] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0]
        }
    values = list(daily.values())
    components = {
        key: float(sum(row[key] for row in values))
        for key in (
            "da_leg", "rt_deviation_leg", "degradation_cost",
            "deviation_penalty",
        )
    }
    mean_by_rt = {
        label: float(report["mean_daily_revenue"])
        for label, report in labeled_reports
    }
    return {
        "days": len(values),
        "segments": reports[0]["segments"],
        "mean_daily_revenue": float(np.mean([
            row["net_revenue"] for row in values
        ])),
        "positive_day_rate": float(np.mean([
            row["net_revenue"] > 0 for row in values
        ])),
        "revenue_components": components,
        "plan_feasibility": {
            "rolling_rt_reference_count": len(reports),
            "per_rolling_rt": {
                label: report["plan_feasibility"]
                for label, report in labeled_reports
            },
        },
        "actual_state": {
            "rolling_rt_reference_count": len(reports),
            "per_rolling_rt_action_digest": {
                label: report["actual_state"]["action_digest"]
                for label, report in labeled_reports
            },
        },
        "daily": daily,
        "reference_panel_labels": [label for label, _ in labeled_reports],
        "per_reference_mean_daily_revenue": mean_by_rt,
    }


def _load_rolling_predictions(cfg: PilotConfig, device: torch.device) -> dict:
    predictions = {"validation": [], "test": []}
    for path in ROLLING_RT_PATHS:
        checkpoint = _load_checkpoint(path)
        model_cfg = _config_from_checkpoint(checkpoint)
        data_cfg = _eval_config(model_cfg, cfg)
        train_ds, val_ds, test_ds, _ = build_rt_datasets(data_cfg)
        _check_norm_stats(train_ds, checkpoint, f"冻结滚动RT {path}")
        model = DecisionAwareRTForecaster(model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        for split, dataset in {"validation": val_ds, "test": test_ds}.items():
            predictions[split].append((
                path,
                dataset,
                _predict(model, dataset, "p_rt", device, batch_size=128),
            ))
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def _mode_report(cfg: PilotConfig, mode: str, rolling_predictions: dict,
                 device: torch.device) -> dict:
    mode_cfg = replace(
        cfg,
        frozen_da_checkpoints=DA_PATHS,
        frozen_rt_at_da_checkpoints=MODE_PATHS[mode],
        frozen_coordination_mode="rt_only",
    )
    datasets = {
        split: values[0][1] for split, values in rolling_predictions.items()
    }
    evaluator = FrozenRollingRTDualEvaluator(mode_cfg, datasets, device)
    result = {}
    for split, values in rolling_predictions.items():
        labeled = []
        for path, _, prediction in values:
            labeled.append((path, evaluator.evaluate(split, prediction)))
        averaged = _average_rolling_rt_reports(labeled)
        result[split] = _compact_dual(
            averaged,
            mode_cfg.bootstrap_samples,
            20261002 + (0 if split == "validation" else 10_000),
        )
    del evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/rt_fusion_ablation_v1.yaml"
    )
    parser.add_argument(
        "--output",
        default="data/results/rt_at_da_f1b_downstream_propagation.json",
    )
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if cfg.frozen_coordination_mode != "rt_only":
        raise ValueError("传播确认必须固定rt_only协调方式")
    missing = [
        path for path in DA_PATHS + ROLLING_RT_PATHS + sum(MODE_PATHS.values(), [])
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"传播确认缺少checkpoint: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rolling_predictions = _load_rolling_predictions(cfg, device)
    modes = {
        mode: _mode_report(cfg, mode, rolling_predictions, device)
        for mode in ("F0", "F1b")
    }

    left = modes["F0"]["validation"]["daily_net_revenue"]
    right = modes["F1b"]["validation"]["daily_net_revenue"]
    dates = sorted(set(left) & set(right))
    if len(dates) != len(left) or len(dates) != len(right):
        raise ValueError("F0与F1b验证日期没有完全配对")
    difference = np.asarray(
        [right[date] - left[date] for date in dates], dtype=np.float64
    )
    interval = _block_bootstrap_ci(
        difference, cfg.bootstrap_samples, seed=20261002
    )
    stable_improvement = interval[0] > 0.0
    selected = "F1b" if stable_improvement else "F0"

    report = {
        "experiment": "RT-at-DA F1b downstream propagation confirmation",
        "selection_period": "validation only",
        "selection_rule": (
            "upgrade to F1b only when the weekly-block bootstrap 95% CI lower "
            "bound of paired validation daily revenue difference is above zero"
        ),
        "test_role": "report-only; never used for deployment selection",
        "device": str(device),
        "frozen_system": {
            "da_structure": "F1",
            "da_checkpoints": DA_PATHS,
            "rolling_rt_structure": "F0",
            "rolling_rt_checkpoints": ROLLING_RT_PATHS,
            "coordination_mode": "rt_only",
            "rt_at_da_panels": MODE_PATHS,
            "reference_combinations_per_mode": 27,
        },
        "split": {
            name: list(cfg.split_bounds(name))
            for name in ("train", "val", "test")
        },
        "modes": modes,
        "paired_validation_f1b_minus_f0": {
            "mean_daily_difference": float(difference.mean()),
            "weekly_block_bootstrap_95ci": interval,
            "paired_days": len(dates),
        },
        "selected_for_deployment": selected,
        "stable_f1b_improvement": stable_improvement,
        "compute": {
            "seconds": round(time.time() - started, 2),
            "peak_cuda_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda" else 0.0
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "validation_revenue": {
            mode: modes[mode]["validation"]["mean_daily_revenue"]
            for mode in modes
        },
        "paired_f1b_minus_f0": report["paired_validation_f1b_minus_f0"],
        "selected_for_deployment": selected,
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
