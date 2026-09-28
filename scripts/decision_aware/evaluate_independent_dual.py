#!/usr/bin/env python
"""在同一独立RT模型下比较Transformer DA与XGBoost DA的双结算收益。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.backtest_dual import locked_dual_backtest  # noqa: E402
from decision_aware.baselines_da import dataset_to_numpy, fit_xgboost  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_da import build_da_datasets  # noqa: E402
from decision_aware.dataset_rt import build_rt_datasets  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"checkpoint结构不完整: {path}")
    return checkpoint


def _config_from_checkpoint(checkpoint: dict) -> PilotConfig:
    fields = PilotConfig.__dataclass_fields__
    return PilotConfig(**{key: value for key, value in checkpoint["config"].items() if key in fields})


def _check_norm_stats(dataset, checkpoint: dict, label: str) -> None:
    saved = checkpoint.get("norm_stats")
    if saved is None:
        raise ValueError(f"{label} checkpoint缺少训练集norm_stats")
    for price_name in ("price_da", "price_rt"):
        for statistic in ("mean", "std"):
            left = float(dataset.norm_stats[price_name][statistic])
            right = float(saved[price_name][statistic])
            if not np.isclose(left, right, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"{label}的{price_name}.{statistic}与当前数据不一致: {left} != {right}"
                )


def _predict(model, dataset, output_key: str, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.append(model(batch)[output_key].float().cpu().numpy())
    return np.concatenate(predictions)


def _da_mapping(dataset, predictions: np.ndarray) -> dict:
    if len(dataset) != len(predictions):
        raise ValueError("DA预测数量与日样本数量不一致")
    return {
        window.delivery_date: predictions[index]
        for index, window in enumerate(dataset.windows)
    }


def _rt_realized_and_times(dataset):
    positions = [window.issue_position for window in dataset.windows]
    return (
        torch.from_numpy(dataset.price_da[positions].copy()),
        torch.from_numpy(dataset.price_rt[positions].copy()),
        [dataset.sample_time_contract(index)["executed_target"] for index in range(len(dataset))],
    )


def _mae(prediction: np.ndarray, dataset) -> float:
    target = np.stack([dataset[index]["price_da_tgt"].numpy() for index in range(len(dataset))])
    return float(np.mean(np.abs(prediction - target)))


def _da_dataset_signature(dataset) -> str:
    digest = hashlib.sha256()
    for index, window in enumerate(dataset.windows):
        digest.update(str(window.delivery_date).encode("ascii"))
        digest.update(dataset[index]["price_da_tgt"].numpy().astype("<f4").tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da-checkpoint",
        default="data/checkpoints/da_independent_seed2/pilot_LZ_LCRA_best_revenue.pt",
    )
    parser.add_argument(
        "--rt-checkpoint",
        default="data/checkpoints/rt_independent_seed2/pilot_LZ_LCRA_best_rolling_revenue.pt",
    )
    parser.add_argument(
        "--xgb-cache", default="data/results/da_xgboost_predictions_seed0.npz"
    )
    parser.add_argument(
        "--output", default="data/results/independent_da_rt_dual_comparison.json"
    )
    parser.add_argument("--retrain-xgboost", action="store_true")
    args = parser.parse_args()

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    da_checkpoint = _load_checkpoint(args.da_checkpoint)
    rt_checkpoint = _load_checkpoint(args.rt_checkpoint)
    da_cfg = _config_from_checkpoint(da_checkpoint)
    rt_cfg = _config_from_checkpoint(rt_checkpoint)
    physical_fields = (
        "bess_power_mw", "bess_energy_mwh", "bess_eta", "bess_init_soc_frac",
        "bess_kappa", "bess_soc_min", "bess_soc_max", "bess_e_cyc",
    )
    mismatches = {
        field: [getattr(da_cfg, field), getattr(rt_cfg, field)]
        for field in physical_fields
        if getattr(da_cfg, field) != getattr(rt_cfg, field)
    }
    if mismatches:
        raise ValueError(f"DA与RT物理参数不一致: {mismatches}")

    da_train, _, da_test, _ = build_da_datasets(da_cfg)
    rt_train, _, rt_test, _ = build_rt_datasets(rt_cfg)
    _check_norm_stats(da_train, da_checkpoint, "DA")
    _check_norm_stats(rt_train, rt_checkpoint, "RT")

    da_model = DecisionAwareDAForecaster(da_cfg).to(device)
    da_model.load_state_dict(da_checkpoint["model_state"])
    transformer_prediction = _predict(da_model, da_test, "p_da", device)
    del da_model

    rt_model = DecisionAwareRTForecaster(rt_cfg).to(device)
    rt_model.load_state_dict(rt_checkpoint["model_state"])
    rt_prediction = _predict(rt_model, rt_test, "p_rt", device)
    del rt_model

    cache_path = Path(args.xgb_cache)
    da_test_signature = _da_dataset_signature(da_test)
    xgb_source = "cache"
    if cache_path.exists() and not args.retrain_xgboost:
        cached = np.load(cache_path)
        xgb_prediction = cached["prediction"]
        if xgb_prediction.shape != (len(da_test), 24):
            raise ValueError("XGBoost预测缓存形状与当前DA测试集不一致")
        if str(cached["test_signature"].item()) != da_test_signature:
            raise ValueError("XGBoost预测缓存的数据签名与当前DA测试集不一致")
    else:
        xgb_source = "trained_from_train_split"
        X_train, y_train = dataset_to_numpy(da_train)
        X_test, _ = dataset_to_numpy(da_test)
        xgb_model = fit_xgboost(X_train, y_train, seed=0)
        xgb_prediction = xgb_model.predict(X_test).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, prediction=xgb_prediction, test_signature=da_test_signature
        )

    realized_da, realized_rt, local_timestamps = _rt_realized_and_times(rt_test)
    common = {
        "transformer_da": locked_dual_backtest(
            _da_mapping(da_test, transformer_prediction),
            rt_prediction,
            realized_da,
            realized_rt,
            local_timestamps,
            da_cfg,
            da_k_charge=da_cfg.topk_k_charge,
            da_k_discharge=da_cfg.topk_k_discharge,
            rt_k_charge=rt_cfg.topk_k_charge,
            rt_k_discharge=rt_cfg.topk_k_discharge,
        ),
        "xgboost_da": locked_dual_backtest(
            _da_mapping(da_test, xgb_prediction),
            rt_prediction,
            realized_da,
            realized_rt,
            local_timestamps,
            da_cfg,
            da_k_charge=da_cfg.topk_k_charge,
            da_k_discharge=da_cfg.topk_k_discharge,
            rt_k_charge=rt_cfg.topk_k_charge,
            rt_k_discharge=rt_cfg.topk_k_discharge,
        ),
    }
    left = common["transformer_da"]
    right = common["xgboost_da"]
    if left["actual_state"] != right["actual_state"]:
        raise AssertionError("公平性检查失败：两种DA模型没有使用完全相同的RT动作/SOC")
    if (left["days"], left["hours"], left["segments"]) != (
        right["days"], right["hours"], right["segments"]
    ):
        raise AssertionError("公平性检查失败：两种DA模型的回测时间范围不一致")

    difference = left["mean_daily_revenue"] - right["mean_daily_revenue"]
    winner = "transformer_da" if difference > 0 else "xgboost_da" if difference < 0 else "tie"
    report = {
        "purpose": "在固定同一个RT模型、动作和真实SOC轨迹下，只比较DA模型",
        "device": str(device),
        "da_checkpoint": str(args.da_checkpoint),
        "rt_checkpoint": str(args.rt_checkpoint),
        "xgboost_prediction_source": xgb_source,
        "fairness_checks": {
            "same_rt_forecast": True,
            "same_actual_action_and_soc": True,
            "same_test_hours": True,
            "no_test_data_in_xgboost_fit": True,
        },
        "forecast_mae_all_da_test_days": {
            "transformer_da": _mae(transformer_prediction, da_test),
            "xgboost_da": _mae(xgb_prediction, da_test),
        },
        "dual_settlement": common,
        "comparison": {
            "winner_by_mean_daily_revenue": winner,
            "transformer_minus_xgboost_mean_daily_revenue": difference,
        },
        "seconds": round(time.time() - started, 2),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "winner": winner,
        "transformer_mean_daily_revenue": left["mean_daily_revenue"],
        "xgboost_mean_daily_revenue": right["mean_daily_revenue"],
        "difference": difference,
        "days": left["days"],
        "hours": left["hours"],
        "device": str(device),
    }, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
