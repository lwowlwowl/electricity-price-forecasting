#!/usr/bin/env python
"""用DA-RT-at-DA价差信号选择完整双结算系统与固定RT协调方式。

验证集同时比较3个RT-at-DA seed、3个滚动RT seed和3种固定协调方式；
分数是Transformer DA与XGBoost DA两个参考的平均日收益。测试集只在选择
完成后评估验证胜者和旧DA-only诊断基线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "decision_aware"))
os.chdir(ROOT)

from evaluate_independent_dual import (  # noqa: E402
    _check_norm_stats,
    _config_from_checkpoint,
    _da_mapping,
    _load_checkpoint,
    _predict,
    _rt_realized_and_times,
)
from select_rt_coordination import _compact, _xgboost_predictions  # noqa: E402
from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.dataset_rt_da import build_rt_at_da_datasets  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


MODES = ("rt_only", "follow_da", "lookahead_mpc")
PHYSICAL_FIELDS = (
    "bess_power_mw", "bess_energy_mwh", "bess_eta", "bess_init_soc_frac",
    "bess_kappa", "bess_soc_min", "bess_soc_max", "bess_e_cyc",
)


def _check_physical(reference, candidate, label: str) -> None:
    mismatches = {
        name: [getattr(reference, name), getattr(candidate, name)]
        for name in PHYSICAL_FIELDS
        if getattr(reference, name) != getattr(candidate, name)
    }
    if mismatches:
        raise ValueError(f"{label}物理参数与DA不一致: {mismatches}")


def _check_daily_alignment(da_dataset, rt_at_da_dataset, split: str) -> None:
    da_dates = [window.delivery_date for window in da_dataset.windows]
    rt_dates = [window.delivery_date for window in rt_at_da_dataset.windows]
    if da_dates != rt_dates:
        raise ValueError(f"{split}的DA与RT-at-DA交付日没有逐日对齐")


def _run(
    da_dataset,
    da_prediction,
    rt_at_da_dataset,
    rt_at_da_prediction,
    rt_dataset,
    rt_prediction,
    cfg,
    mode: str,
):
    realized_da, realized_rt, timestamps = _rt_realized_and_times(rt_dataset)
    return locked_dual_backtest(
        _da_mapping(da_dataset, da_prediction),
        rt_prediction,
        realized_da,
        realized_rt,
        timestamps,
        cfg,
        da_k_charge=cfg.topk_k_charge,
        da_k_discharge=cfg.topk_k_discharge,
        rt_k_charge=1,
        rt_k_discharge=1,
        coordination_mode=mode,
        rt_at_da_price_forecasts=_da_mapping(
            rt_at_da_dataset, rt_at_da_prediction
        ),
    )


def _run_legacy(
    da_dataset, da_prediction, rt_dataset, rt_prediction, cfg
):
    realized_da, realized_rt, timestamps = _rt_realized_and_times(rt_dataset)
    return locked_dual_backtest(
        _da_mapping(da_dataset, da_prediction),
        rt_prediction,
        realized_da,
        realized_rt,
        timestamps,
        cfg,
        da_k_charge=cfg.topk_k_charge,
        da_k_discharge=cfg.topk_k_discharge,
        rt_k_charge=1,
        rt_k_discharge=1,
        coordination_mode="rt_only",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-checkpoint",
        default="data/checkpoints/da_independent_seed2/pilot_LZ_LCRA_best_revenue.pt",
    )
    parser.add_argument(
        "--rt-at-da-checkpoints", nargs="+", default=[
            "data/checkpoints/rt_at_da_independent_v1/pilot_LZ_LCRA_best_mae.pt",
            "data/checkpoints/rt_at_da_independent_seed1/pilot_LZ_LCRA_best_mae.pt",
            "data/checkpoints/rt_at_da_independent_seed2/pilot_LZ_LCRA_best_mae.pt",
        ],
    )
    parser.add_argument(
        "--rt-checkpoints", nargs="+", default=[
            "data/checkpoints/rt_independent_v1/pilot_LZ_LCRA_best_rolling_revenue.pt",
            "data/checkpoints/rt_independent_seed1/pilot_LZ_LCRA_best_rolling_revenue.pt",
            "data/checkpoints/rt_independent_seed2/pilot_LZ_LCRA_best_rolling_revenue.pt",
        ],
    )
    parser.add_argument(
        "--xgb-cache", default="data/results/da_xgboost_val_test_predictions_seed0.npz"
    )
    parser.add_argument(
        "--output", default="data/results/spread_coordination_validation_selection.json"
    )
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    da_checkpoint = _load_checkpoint(args.da_checkpoint)
    da_cfg = _config_from_checkpoint(da_checkpoint)
    da_train, da_val, da_test, _ = build_da_datasets(da_cfg)
    _check_norm_stats(da_train, da_checkpoint, "DA")

    da_model = DecisionAwareDAForecaster(da_cfg).to(device)
    da_model.load_state_dict(da_checkpoint["model_state"])
    transformer_da = {
        "validation": _predict(da_model, da_val, "p_da", device),
        "test": _predict(da_model, da_test, "p_da", device),
    }
    del da_model
    xgb_val, xgb_test, xgb_source = _xgboost_predictions(
        da_train, da_val, da_test, Path(args.xgb_cache)
    )
    da_predictions = {
        "transformer_da_seed2": transformer_da,
        "xgboost_da_seed0": {"validation": xgb_val, "test": xgb_test},
    }

    rt_at_da_runs = []
    rt_at_da_datasets = None
    for path_text in args.rt_at_da_checkpoints:
        checkpoint = _load_checkpoint(path_text)
        cfg = _config_from_checkpoint(checkpoint)
        _check_physical(da_cfg, cfg, "RT-at-DA")
        if rt_at_da_datasets is None:
            rt_da_train, rt_da_val, rt_da_test, _ = build_rt_at_da_datasets(cfg)
            rt_at_da_datasets = {
                "train": rt_da_train, "validation": rt_da_val, "test": rt_da_test
            }
            _check_daily_alignment(da_val, rt_da_val, "validation")
            _check_daily_alignment(da_test, rt_da_test, "test")
        _check_norm_stats(rt_at_da_datasets["train"], checkpoint, f"RT-at-DA seed {cfg.seed}")
        model = DecisionAwareRTAtDAForecaster(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        validation_prediction = _predict(
            model, rt_at_da_datasets["validation"], "p_rt_at_da", device
        )
        validation_target = np.stack([
            rt_at_da_datasets["validation"][i]["price_rt_at_da_tgt"].numpy()
            for i in range(len(rt_at_da_datasets["validation"]))
        ])
        rt_at_da_runs.append({
            "seed": int(cfg.seed),
            "checkpoint": str(path_text),
            "config": cfg,
            "validation_prediction": validation_prediction,
            "validation_mae": float(np.mean(np.abs(validation_prediction - validation_target))),
        })
        del model
    rt_at_da_runs.sort(key=lambda row: row["seed"])

    rt_runs = []
    rt_datasets = None
    for path_text in args.rt_checkpoints:
        checkpoint = _load_checkpoint(path_text)
        cfg = _config_from_checkpoint(checkpoint)
        _check_physical(da_cfg, cfg, "滚动RT")
        if rt_datasets is None:
            rt_train, rt_val, rt_test, _ = build_rt_datasets(cfg)
            rt_datasets = {
                "train": rt_train, "validation": rt_val, "test": rt_test
            }
        _check_norm_stats(rt_datasets["train"], checkpoint, f"滚动RT seed {cfg.seed}")
        model = DecisionAwareRTForecaster(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        rt_runs.append({
            "seed": int(cfg.seed),
            "checkpoint": str(path_text),
            "config": cfg,
            "validation_prediction": _predict(
                model, rt_datasets["validation"], "p_rt", device
            ),
        })
        del model
    rt_runs.sort(key=lambda row: row["seed"])

    validation_grid = []
    for rt_da_run in rt_at_da_runs:
        for rt_run in rt_runs:
            for mode in MODES:
                systems = {}
                for da_name, predictions in da_predictions.items():
                    systems[da_name] = _compact(_run(
                        da_val,
                        predictions["validation"],
                        rt_at_da_datasets["validation"],
                        rt_da_run["validation_prediction"],
                        rt_datasets["validation"],
                        rt_run["validation_prediction"],
                        da_cfg,
                        mode,
                    ))
                score = float(np.mean([
                    systems[name]["mean_daily_revenue"] for name in sorted(systems)
                ]))
                validation_grid.append({
                    "rt_at_da_seed": rt_da_run["seed"],
                    "rt_at_da_checkpoint": rt_da_run["checkpoint"],
                    "rt_at_da_validation_mae": rt_da_run["validation_mae"],
                    "rolling_rt_seed": rt_run["seed"],
                    "rolling_rt_checkpoint": rt_run["checkpoint"],
                    "coordination_mode": mode,
                    "selection_score": score,
                    "systems": systems,
                })

    selected = max(validation_grid, key=lambda row: row["selection_score"])
    selected_rt_da = next(
        row for row in rt_at_da_runs if row["seed"] == selected["rt_at_da_seed"]
    )
    selected_rt = next(
        row for row in rt_runs if row["seed"] == selected["rolling_rt_seed"]
    )
    old_rt = next(row for row in rt_runs if row["seed"] == 2)
    legacy_validation = {}
    for da_name, predictions in da_predictions.items():
        legacy_validation[da_name] = _compact(_run_legacy(
            da_val,
            predictions["validation"],
            rt_datasets["validation"],
            old_rt["validation_prediction"],
            da_cfg,
        ))
    legacy_validation_score = float(np.mean([
        legacy_validation[name]["mean_daily_revenue"]
        for name in sorted(legacy_validation)
    ]))

    selected_rt_da_checkpoint = _load_checkpoint(selected_rt_da["checkpoint"])
    selected_rt_da_model = DecisionAwareRTAtDAForecaster(
        selected_rt_da["config"]
    ).to(device)
    selected_rt_da_model.load_state_dict(selected_rt_da_checkpoint["model_state"])
    selected_rt_da_test_prediction = _predict(
        selected_rt_da_model, rt_at_da_datasets["test"], "p_rt_at_da", device
    )
    del selected_rt_da_model
    selected_rt_da_test_target = np.stack([
        rt_at_da_datasets["test"][i]["price_rt_at_da_tgt"].numpy()
        for i in range(len(rt_at_da_datasets["test"]))
    ])
    selected_rt_da_test_error = (
        selected_rt_da_test_prediction - selected_rt_da_test_target
    )

    selected_rt_checkpoint = _load_checkpoint(selected_rt["checkpoint"])
    selected_rt_model = DecisionAwareRTForecaster(selected_rt["config"]).to(device)
    selected_rt_model.load_state_dict(selected_rt_checkpoint["model_state"])
    selected_rt_test_prediction = _predict(
        selected_rt_model, rt_datasets["test"], "p_rt", device
    )
    del selected_rt_model

    if old_rt["seed"] == selected_rt["seed"]:
        old_rt_test_prediction = selected_rt_test_prediction
    else:
        old_checkpoint = _load_checkpoint(old_rt["checkpoint"])
        old_model = DecisionAwareRTForecaster(old_rt["config"]).to(device)
        old_model.load_state_dict(old_checkpoint["model_state"])
        old_rt_test_prediction = _predict(
            old_model, rt_datasets["test"], "p_rt", device
        )
        del old_model

    selected_test = {}
    old_test = {}
    for da_name, predictions in da_predictions.items():
        selected_test[da_name] = _compact(_run(
            da_test,
            predictions["test"],
            rt_at_da_datasets["test"],
            selected_rt_da_test_prediction,
            rt_datasets["test"],
            selected_rt_test_prediction,
            da_cfg,
            selected["coordination_mode"],
        ))
        old_test[da_name] = _compact(_run_legacy(
            da_test,
            predictions["test"],
            rt_datasets["test"],
            old_rt_test_prediction,
            da_cfg,
        ))

    report = {
        "purpose": "select RT-at-DA + rolling RT + fixed coordination before fusion ablation",
        "da_decision_signal": "predicted_DA_minus_predicted_RT_at_DA",
        "selection_split": "validation only",
        "selection_rule": (
            "maximize mean full-dual validation daily revenue across fixed "
            "Transformer DA seed2 and XGBoost DA seed0 references"
        ),
        "test_usage": "test evaluated only for legacy diagnostic and validation winner",
        "device": str(device),
        "candidate_counts": {
            "rt_at_da_seeds": len(rt_at_da_runs),
            "rolling_rt_seeds": len(rt_runs),
            "coordination_modes": len(MODES),
            "total_system_candidates": len(validation_grid),
        },
        "da_checkpoint": str(args.da_checkpoint),
        "xgboost_prediction_source": xgb_source,
        "validation_grid": validation_grid,
        "validation_comparison": {
            "legacy_da_only_signal_rt_seed2_rt_only_score": legacy_validation_score,
            "legacy_systems": legacy_validation,
            "selected_spread_system_score": selected["selection_score"],
            "score_change": selected["selection_score"] - legacy_validation_score,
        },
        "selected": {
            "rt_at_da_seed": selected["rt_at_da_seed"],
            "rt_at_da_checkpoint": selected["rt_at_da_checkpoint"],
            "rolling_rt_seed": selected["rolling_rt_seed"],
            "rolling_rt_checkpoint": selected["rolling_rt_checkpoint"],
            "coordination_mode": selected["coordination_mode"],
            "validation_selection_score": selected["selection_score"],
        },
        "test_comparison": {
            "legacy_da_only_signal_rt_seed2_rt_only": old_test,
            "selected_spread_system": selected_test,
        },
        "selected_rt_at_da_test_forecast": {
            "mae": float(np.mean(np.abs(selected_rt_da_test_error))),
            "rmse": float(np.sqrt(np.mean(selected_rt_da_test_error ** 2))),
            "days": len(rt_at_da_datasets["test"]),
            "note": "reported only after validation selection; not used to choose the model",
        },
        "seconds": round(time.time() - started, 2),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["selected"], ensure_ascii=False), flush=True)
    for name in sorted(selected_test):
        before = old_test[name]["mean_daily_revenue"]
        after = selected_test[name]["mean_daily_revenue"]
        print(json.dumps({
            "system": name,
            "legacy_test_revenue": before,
            "corrected_test_revenue": after,
            "change": after - before,
        }, ensure_ascii=False), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
