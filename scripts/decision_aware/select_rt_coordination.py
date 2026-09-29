#!/usr/bin/env python
"""历史DA-only信号下选择固定RT协调方式和RT checkpoint。

补齐RT-at-DA后的正式选择入口是``select_spread_coordination.py``。
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
    _da_dataset_signature,
    _da_mapping,
    _load_checkpoint,
    _predict,
    _rt_realized_and_times,
)
from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.baselines_da import dataset_to_numpy, fit_xgboost  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402


MODES = ("rt_only", "follow_da")


def _xgboost_predictions(train_ds, val_ds, test_ds, cache_path: Path):
    val_signature = _da_dataset_signature(val_ds)
    test_signature = _da_dataset_signature(test_ds)
    if cache_path.exists():
        cached = np.load(cache_path)
        if (
            str(cached["validation_signature"].item()) == val_signature
            and str(cached["test_signature"].item()) == test_signature
            and cached["validation_prediction"].shape == (len(val_ds), 24)
            and cached["test_prediction"].shape == (len(test_ds), 24)
        ):
            return cached["validation_prediction"], cached["test_prediction"], "cache"

    X_train, y_train = dataset_to_numpy(train_ds)
    X_val, _ = dataset_to_numpy(val_ds)
    X_test, _ = dataset_to_numpy(test_ds)
    model = fit_xgboost(X_train, y_train, seed=0)
    val_prediction = model.predict(X_val).astype(np.float32)
    test_prediction = model.predict(X_test).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        validation_prediction=val_prediction,
        test_prediction=test_prediction,
        validation_signature=val_signature,
        test_signature=test_signature,
    )
    return val_prediction, test_prediction, "trained_from_train_split"


def _compact(backtest: dict) -> dict:
    days = backtest["days"]
    return {
        "mean_daily_revenue": backtest["mean_daily_revenue"],
        "positive_day_rate": backtest["positive_day_rate"],
        "days": days,
        "hours": backtest["hours"],
        "coordination": backtest["coordination"],
        "mean_daily_components": {
            key: value / days for key, value in backtest["revenue_components"].items()
        },
        "plan_total_clipped_mwh": backtest["plan_feasibility"]["total_clipped_mwh"],
        "actual_segment_final_soc_mwh": backtest["actual_state"]["segment_final_soc_mwh"],
        "actual_action_digest": backtest["actual_state"]["action_digest"],
    }


def _run(
    da_dataset,
    da_prediction,
    rt_dataset,
    rt_prediction,
    cfg,
    mode: str,
) -> dict:
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-checkpoint",
        default="data/checkpoints/da_independent_seed2/pilot_LZ_LCRA_best_revenue.pt",
    )
    parser.add_argument(
        "--rt-checkpoints",
        nargs="+",
        default=[
            "data/checkpoints/rt_independent_v1/pilot_LZ_LCRA_best_rolling_revenue.pt",
            "data/checkpoints/rt_independent_seed1/pilot_LZ_LCRA_best_rolling_revenue.pt",
            "data/checkpoints/rt_independent_seed2/pilot_LZ_LCRA_best_rolling_revenue.pt",
        ],
    )
    parser.add_argument(
        "--xgb-cache",
        default="data/results/da_xgboost_val_test_predictions_seed0.npz",
    )
    parser.add_argument(
        "--output",
        default="data/results/rt_coordination_validation_selection.json",
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
        "transformer_da_seed2": {
            "validation": transformer_da["validation"],
            "test": transformer_da["test"],
        },
        "xgboost_da_seed0": {"validation": xgb_val, "test": xgb_test},
    }

    rt_runs = []
    rt_datasets = None
    physical_fields = (
        "bess_power_mw", "bess_energy_mwh", "bess_eta", "bess_init_soc_frac",
        "bess_kappa", "bess_soc_min", "bess_soc_max", "bess_e_cyc",
    )
    for path_text in args.rt_checkpoints:
        checkpoint = _load_checkpoint(path_text)
        cfg = _config_from_checkpoint(checkpoint)
        mismatches = {
            name: [getattr(da_cfg, name), getattr(cfg, name)]
            for name in physical_fields
            if getattr(da_cfg, name) != getattr(cfg, name)
        }
        if mismatches:
            raise ValueError(f"DA/RT物理参数不一致: {mismatches}")
        if rt_datasets is None:
            rt_train, rt_val, rt_test, _ = build_rt_datasets(cfg)
            rt_datasets = {"train": rt_train, "validation": rt_val, "test": rt_test}
        _check_norm_stats(rt_datasets["train"], checkpoint, f"RT seed {cfg.seed}")
        model = DecisionAwareRTForecaster(cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        rt_runs.append({
            "seed": int(cfg.seed),
            "checkpoint": str(path_text),
            "config": cfg,
            "validation_prediction": _predict(
                model, rt_datasets["validation"], "p_rt", device
            ),
            "test_prediction": _predict(model, rt_datasets["test"], "p_rt", device),
        })
        del model
    rt_runs.sort(key=lambda row: row["seed"])

    validation_grid = []
    for rt_run in rt_runs:
        for mode in MODES:
            systems = {}
            for da_name, predictions in da_predictions.items():
                result = _run(
                    da_val,
                    predictions["validation"],
                    rt_datasets["validation"],
                    rt_run["validation_prediction"],
                    da_cfg,
                    mode,
                )
                systems[da_name] = _compact(result)
            score = float(np.mean([
                systems[name]["mean_daily_revenue"] for name in sorted(systems)
            ]))
            validation_grid.append({
                "rt_seed": rt_run["seed"],
                "rt_checkpoint": rt_run["checkpoint"],
                "coordination_mode": mode,
                "selection_score": score,
                "selection_score_definition": (
                    "mean full-dual validation daily revenue across "
                    "transformer_da_seed2 and xgboost_da_seed0"
                ),
                "systems": systems,
            })

    selected = max(validation_grid, key=lambda row: row["selection_score"])
    selected_rt = next(row for row in rt_runs if row["seed"] == selected["rt_seed"])
    selected_test = {}
    old_test = {}
    old_rt = next(row for row in rt_runs if row["seed"] == 2)
    for da_name, predictions in da_predictions.items():
        selected_test[da_name] = _compact(_run(
            da_test,
            predictions["test"],
            rt_datasets["test"],
            selected_rt["test_prediction"],
            da_cfg,
            selected["coordination_mode"],
        ))
        old_test[da_name] = _compact(_run(
            da_test,
            predictions["test"],
            rt_datasets["test"],
            old_rt["test_prediction"],
            da_cfg,
            "rt_only",
        ))

    report = {
        "purpose": "freeze a DA-aware RT coordination rule before fusion ablation",
        "selection_split": "validation only",
        "selection_rule": (
            "maximize mean full-dual validation daily revenue across the fixed "
            "Transformer DA seed2 and XGBoost DA seed0 references"
        ),
        "test_usage": "test evaluated only for the predeclared old baseline and validation winner",
        "device": str(device),
        "da_checkpoint": str(args.da_checkpoint),
        "xgboost_prediction_source": xgb_source,
        "candidate_modes": list(MODES),
        "validation_grid": validation_grid,
        "selected": {
            "rt_seed": selected["rt_seed"],
            "rt_checkpoint": selected["rt_checkpoint"],
            "coordination_mode": selected["coordination_mode"],
            "validation_selection_score": selected["selection_score"],
        },
        "test_comparison": {
            "old_rt_seed2_rt_only": old_test,
            "selected_validation_winner": selected_test,
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
            "old_test_revenue": before,
            "selected_test_revenue": after,
            "change": after - before,
        }, ensure_ascii=False), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
