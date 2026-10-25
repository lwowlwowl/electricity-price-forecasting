#!/usr/bin/env python
"""Run frozen transferable systems on the read-only Austria 15-minute test."""
from __future__ import annotations

import argparse
from dataclasses import fields
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "external" / "timesfm" / "src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    add_batch_dimension,
    expand_daily_hourly_to_quarters,
    expand_hourly_rt_to_quarter_windows,
    quarter_hour_backtest,
    sha256,
)
from decision_aware.baselines_global import xgboost_feature  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dlinear import DLinearForecaster  # noqa: E402
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402
from decision_aware.robust_ensemble import (  # noqa: E402
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)
from decision_aware.tail_expert_router import (  # noqa: E402
    build_feature_cube,
    router_probability,
)
from scripts.decision_aware.run_universal_lomo_timesfm25_lora import (  # noqa: E402
    _mean_prediction,
)


TASKS = ("da", "rt_at_da", "rt")
PREDICTION_KEY = {"da": "p_da", "rt_at_da": "p_rt_at_da", "rt": "p_rt"}
MODEL_CLASS = {
    "da": DecisionAwareDAForecaster,
    "rt_at_da": DecisionAwareRTAtDAForecaster,
    "rt": DecisionAwareRTForecaster,
}
ALL_SYSTEMS = (
    "seasonal_lag168", "pooled_xgboost", "dlinear", "timesfm25_lora",
    "timesfm3_frozen", "transformer_three_seed_equal",
    "decision_aware_v4_seed0", "decision_aware_v4_three_seed_equal",
    "decision_aware_v4_robust_calibrated", "decision_aware_v4_final_2to1_blend",
    "three_feature_tail_expert_router",
)
ORIGINAL_ROOTS = {
    "da": "data/checkpoints/allmarket_final_da_mae_v1/da_final_seed{seed}/best.pt",
    "rt_at_da": "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1/rt_at_da_final_seed{seed}/best.pt",
    "rt": "data/checkpoints/allmarket_final_rt_mae_v1/rt_final_seed{seed}/best.pt",
}
V4_ROOTS = {
    task: (
        "data/checkpoints/global_joint_implicit_qp_final_v4/"
        "implicitqp_final_decoder_balanced_seed{seed}_v4/" + f"{task}.pt"
    )
    for task in TASKS
}


def _checkpoint_config(payload: dict) -> PilotConfig:
    allowed = {item.name for item in fields(PilotConfig)}
    return PilotConfig(**{key: value for key, value in payload["config"].items() if key in allowed})


def _stack(samples: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = samples[0]
    return {key: torch.stack([sample[key] for sample in samples]).to(device) for key in keys}


def _forecast_axes(data: AustriaExternalData, quarter: pd.DataFrame):
    local = pd.DatetimeIndex(quarter["timestamp_local"])
    dates = sorted(set(local.date))
    issues = pd.date_range(quarter.index[0].floor("1h"), quarter.index[-1].floor("1h"), freq="1h", tz="UTC")
    return dates, issues, local


def _sample_for(data: AustriaExternalData, task: str, value: Any) -> dict[str, torch.Tensor]:
    return data.day_sample(value) if task != "rt" else data.rt_sample(value)


def _run_batched_samples(
    data: AustriaExternalData,
    task: str,
    axes: list[Any] | pd.DatetimeIndex,
    callback: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(axes), batch_size):
        samples = [_sample_for(data, task, value) for value in axes[start:start + batch_size]]
        batch = _stack(samples, device)
        with torch.inference_mode():
            output = callback(batch)
        outputs.append(output.detach().float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _transformer_triplet(
    data: AustriaExternalData,
    dates: list,
    issues: pd.DatetimeIndex,
    templates: dict[str, str],
    seed: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    values: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for task in TASKS:
        path = ROOT / templates[task].format(seed=seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("task") != task:
            raise ValueError(f"checkpoint task错误: {path}")
        cfg = _checkpoint_config(payload)
        if cfg.global_use_entity_embeddings or cfg.context_len != 168:
            raise ValueError("奥地利只允许无ID、168小时冻结Transformer")
        model = MODEL_CLASS[task](cfg).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        axis = issues if task == "rt" else dates

        def predict(batch, model=model, task=task):
            return model(batch)[PREDICTION_KEY[task]]

        values[task] = _run_batched_samples(data, task, axis, predict, device, batch_size)
        hashes[task] = sha256(path)
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return values, hashes


def _dlinear_triplet(data, dates, issues, device, batch_size):
    values, hashes = {}, {}
    for task in TASKS:
        path = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/dlinear/{task}_seed0.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        horizon = 4 if task == "rt" else 24
        model = DLinearForecaster(168, horizon, 25).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        context_key = "price_da_ctx" if task == "da" else "price_rt_ctx"
        mean_key = "price_da_mean" if task == "da" else "price_rt_mean"
        std_key = "price_da_std" if task == "da" else "price_rt_std"

        def predict(batch, model=model):
            return model(batch[context_key]) * batch[std_key].reshape(-1, 1) + batch[mean_key].reshape(-1, 1)

        axis = issues if task == "rt" else dates
        values[task] = _run_batched_samples(data, task, axis, predict, device, batch_size)
        hashes[task] = sha256(path)
        del model, payload
    return values, hashes


def _xgboost_triplet(data, dates, issues):
    values, hashes = {}, {}
    compat_python = ROOT / "runtime_cache/austria_xgb_compat/Scripts/python.exe"
    compat_worker = ROOT / "scripts/decision_aware/predict_xgboost_compat.py"
    if not compat_python.is_file():
        raise FileNotFoundError("缺少项目内隔离XGBoost 3.4.0兼容环境")
    for task in TASKS:
        path = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/pooled_xgboost/{task}_seed0.joblib"
        axis = issues if task == "rt" else dates
        samples = [_sample_for(data, task, value) for value in axis]
        x = np.stack([xgboost_feature(sample, task) for sample in samples])
        runtime = ROOT / "runtime_cache/austria_xgb_compat/ipc"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as workdir:
            input_path = Path(workdir) / "input.npz"
            output_path = Path(workdir) / "output.npz"
            np.savez_compressed(input_path, features=x)
            completed = subprocess.run([
                str(compat_python), str(compat_worker),
                "--checkpoint", str(path), "--input", str(input_path),
                "--output", str(output_path),
            ], cwd=ROOT, capture_output=True, text=True)
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(
                    "XGBoost兼容worker失败\n" + completed.stdout[-2000:] + completed.stderr[-4000:]
                )
            with np.load(output_path, allow_pickle=False) as archive:
                normalized = archive["prediction"].copy()
        prefix = "price_da" if task == "da" else "price_rt"
        center = np.asarray([float(sample[f"{prefix}_mean"]) for sample in samples], dtype=np.float32)
        scale = np.asarray([float(sample[f"{prefix}_std"]) for sample in samples], dtype=np.float32)
        values[task] = normalized * scale[:, None] + center[:, None]
        hashes[task] = sha256(path)
    return values, hashes


def _timesfm25_triplet(data, dates, issues, batch_size):
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction

    verified_snapshot = ROOT / "external/timesfm25_transformers_snapshot"
    snapshot_root = ROOT / "hf_cache/hub/models--google--timesfm-2.5-200m-transformers/snapshots"
    snapshots = sorted(snapshot_root.iterdir())
    model_path = verified_snapshot if (verified_snapshot / "model.safetensors").is_file() else (
        snapshots[0] if len(snapshots) == 1 else None
    )
    if model_path is None or not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError("TimesFM-2.5本地完整model.safetensors不存在")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    values, hashes = {}, {}
    for task in TASKS:
        adapter = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/timesfm25/{task}_seed0"
        base = TimesFm2_5ModelForPrediction.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to(device)
        model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to(device)
        model.eval()
        axis = issues if task == "rt" else dates
        contexts = []
        for value in axis:
            sample = _sample_for(data, task, value)
            key = "price_da_ctx" if task == "da" else "price_rt_ctx"
            mean = sample["price_da_mean" if task == "da" else "price_rt_mean"]
            std = sample["price_da_std" if task == "da" else "price_rt_std"]
            contexts.append((sample[key] * std + mean).numpy())
        context = np.stack(contexts).astype(np.float32)
        predictions = []
        horizon = 4 if task == "rt" else 24
        with torch.inference_mode():
            for start in range(0, len(context), batch_size):
                tensor = torch.from_numpy(context[start:start + batch_size]).to(device)
                predictions.append(_mean_prediction(model, tensor, horizon).float().cpu().numpy())
        values[task] = np.concatenate(predictions).astype(np.float32)
        hashes[task] = sha256(adapter / "adapter_model.safetensors")
        del model, base
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return values, hashes


def _timesfm3_triplet(data, dates, issues):
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=16 if device == "cuda" else 4,
        device=device,
        cache_dir=str(ROOT / "hf_cache" / "hub"),
        local_files_only=True,
    ))
    values, hashes = {}, {
        "model": "google/timesfm-3.0-pytorch (local frozen cache)",
        "device": device,
    }
    for task in TASKS:
        axis = issues if task == "rt" else dates
        contexts = []
        covariates = []
        for value in axis:
            sample = _sample_for(data, task, value)
            price_key = "price_da" if task == "da" else "price_rt"
            if task == "rt":
                issue = pd.Timestamp(value).tz_convert("UTC")
                stop = int(data.hourly.index.searchsorted(issue, side="left"))
                target_index = pd.date_range(issue, periods=4, freq="1h", tz="UTC")
            else:
                issue_local = pd.Timestamp(value - pd.Timedelta(days=1)).tz_localize(data.timezone) + pd.Timedelta(hours=12)
                issue = issue_local.tz_convert("UTC")
                stop = int(data.hourly.index.searchsorted(issue, side="left"))
                # Synthetic 24-clock-hour query index; covariates are already in sample.
                target_index = pd.date_range(issue, periods=24, freq="1h", tz="UTC")
            context = data.hourly.iloc[stop - 168:stop][price_key].to_numpy(np.float32)
            hist_cov = sample["cal_ctx"].numpy()
            future_cov = sample["cal_tgt"].numpy()
            if task == "rt":
                past_da = data.hourly.iloc[stop - 168:stop]["price_da"].to_numpy(np.float32)
                future_da = sample["price_da_tgt_known"].numpy() * float(sample["price_da_std"]) + float(sample["price_da_mean"])
                hist_cov = np.column_stack([hist_cov, past_da])
                future_cov = np.column_stack([future_cov, future_da])
            contexts.append(context)
            covariates.append(np.concatenate([hist_cov, future_cov], axis=0).T)
        horizon = 4 if task == "rt" else 24
        outputs = list(model.predict_batch(
            contexts=contexts,
            horizon=horizon,
            past_future_covariates=covariates,
            return_quantiles=False,
            use_symmetric_averaging=False,
            make_positive=False,
        ))
        values[task] = np.stack([
            np.asarray(item.forecast, dtype=np.float32).reshape(horizon)
            for item in outputs
        ])
    return values, hashes


def _seasonal_triplet(data, dates, issues):
    values = {}
    for task in TASKS:
        axis = issues if task == "rt" else dates
        horizon = 4 if task == "rt" else 24
        rows = []
        for value in axis:
            if task == "rt":
                target = pd.date_range(value, periods=horizon, freq="1h", tz="UTC")
                price = "price_rt"
            else:
                local_noon = pd.Timestamp(value).tz_localize(data.timezone) + pd.Timedelta(hours=12)
                # Query labels 0..23; localization only identifies a deterministic
                # hourly lag anchor and never changes native settlement rows.
                target = pd.date_range(local_noon.tz_convert("UTC") - pd.Timedelta(hours=12), periods=24, freq="1h", tz="UTC")
                price = "price_da" if task == "da" else "price_rt"
            lag = target - pd.Timedelta(hours=168)
            positions = data.hourly.index.get_indexer(lag)
            if np.any(positions < 0):
                raise ValueError("lag-168历史缺失")
            rows.append(data.hourly.iloc[positions][price].to_numpy(np.float32))
        values[task] = np.stack(rows)
    return values, {"rule": "strict hourly lag-168"}


def _load_or_predict_base(name, data, dates, issues, device, batch_size):
    if name == "seasonal_lag168":
        return _seasonal_triplet(data, dates, issues)
    if name == "pooled_xgboost":
        return _xgboost_triplet(data, dates, issues)
    if name == "dlinear":
        return _dlinear_triplet(data, dates, issues, device, batch_size)
    if name == "timesfm25_lora":
        return _timesfm25_triplet(data, dates, issues, batch_size)
    if name == "timesfm3_frozen":
        return _timesfm3_triplet(data, dates, issues)
    raise ValueError(name)


def _save_forecast(path: Path, values: dict[str, np.ndarray], metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        da=values["da"].astype(np.float32),
        rt_at_da=values["rt_at_da"].astype(np.float32),
        rt=values["rt"].astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def _load_forecast(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        values = {task: archive[task].copy() for task in TASKS}
        metadata = json.loads(str(archive["metadata"].item()))
    return values, metadata


def _prepare_derived(system, cache, data, dates, issues, device, batch_size):
    def ensure(name):
        path = cache / f"{name}.npz"
        if path.exists():
            return _load_forecast(path)[0]
        if name in {"seasonal_lag168", "pooled_xgboost", "dlinear", "timesfm25_lora", "timesfm3_frozen"}:
            values, evidence = _load_or_predict_base(name, data, dates, issues, device, batch_size)
        else:
            values, evidence = _prepare_derived(
                name, cache, data, dates, issues, device, batch_size
            )
        _save_forecast(path, values, {"system": name, "evidence": evidence})
        return values

    def ensure_transformer_member(label, templates, seed):
        path = cache / f"{label}.npz"
        if path.exists():
            return _load_forecast(path)
        values, hashes = _transformer_triplet(
            data, dates, issues, templates, seed, device, batch_size
        )
        metadata = {
            "system": label, "seed": seed, "evidence": hashes,
            "austria_training": False, "austria_seed_selection": False,
        }
        _save_forecast(path, values, metadata)
        return values, metadata

    if system == "transformer_three_seed_equal":
        members, evidence = [], {}
        for seed in (0, 1, 2):
            values, metadata = ensure_transformer_member(
                f"transformer_seed{seed}", ORIGINAL_ROOTS, seed
            )
            members.append(values); evidence[f"seed{seed}"] = metadata["evidence"]
        return {task: np.mean([item[task] for item in members], axis=0) for task in TASKS}, evidence
    if system == "decision_aware_v4_final_2to1_blend":
        seed0 = ensure("decision_aware_v4_seed0")
        robust = ensure("decision_aware_v4_robust_calibrated")
        return {
            task: (2.0 / 3.0) * seed0[task] + (1.0 / 3.0) * robust[task]
            for task in TASKS
        }, {
            "rule": "prediction-layer 2:1 blend frozen before Austria scoring",
            "seed0_weight": 2.0 / 3.0,
            "robust_three_seed_weight": 1.0 / 3.0,
        }
    if system.startswith("decision_aware_v4"):
        members, evidence = [], {}
        seeds = (0,) if system == "decision_aware_v4_seed0" else (0, 1, 2)
        for seed in seeds:
            values, metadata = ensure_transformer_member(
                f"decision_aware_v4_seed{seed}_member", V4_ROOTS, seed
            )
            members.append(values); evidence[f"seed{seed}"] = metadata["evidence"]
        if system != "decision_aware_v4_robust_calibrated":
            return {task: np.mean([item[task] for item in members], axis=0) for task in TASKS}, evidence
        checkpoint_path = ROOT / "data/checkpoints/global_robust_ensemble_final_v4/calibrator.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        calibrator = EconomicEnsembleCalibrator(EnsembleCalibratorConfig(**checkpoint["calibrator_config"]))
        calibrator.load_state_dict(checkpoint["model_state"], strict=True)
        calibrator.eval()
        da_members = torch.from_numpy(np.stack([item["da"] for item in members], axis=1)).float()
        rtda_members = torch.from_numpy(np.stack([item["rt_at_da"] for item in members], axis=1)).float()
        with torch.inference_mode():
            da, rtda = calibrator.calibrate_da_pair(da_members, rtda_members)
        rt_stack = np.stack([item["rt"] for item in members], axis=1)
        weights = calibrator.weights()[2].detach().numpy()
        rt = np.einsum("m,imk->ik", weights, rt_stack)
        interpretation = calibrator.interpretation()
        local_hours = issues.tz_convert(data.timezone).hour.to_numpy()
        rt = rt * np.asarray(interpretation["rt_scale"])[local_hours, None]
        rt = rt + np.asarray(interpretation["rt_bias_usd_per_mwh"])[local_hours, None]
        evidence["calibrator"] = sha256(checkpoint_path)
        return {"da": da.numpy(), "rt_at_da": rtda.numpy(), "rt": rt}, evidence
    if system == "three_feature_tail_expert_router":
        transformer = ensure("transformer_three_seed_equal")
        dlinear = ensure("dlinear")
        cube, _ = build_feature_cube(
            transformer["da"][None], transformer["rt_at_da"][None],
            dlinear["da"][None], dlinear["rt_at_da"][None],
            np.asarray([str(value) for value in dates]),
        )
        model_path = ROOT / "data/results/tail_expert_router_v1/router_refit_2023_2024.joblib"
        router = joblib.load(model_path)
        select_hybrid = (router_probability(router, cube.reshape(len(dates), -1)) >= 0.70)
        values = {
            "da": np.where(select_hybrid[:, None], dlinear["da"], transformer["da"]),
            "rt_at_da": np.where(select_hybrid[:, None], dlinear["rt_at_da"], transformer["rt_at_da"]),
            "rt": transformer["rt"],
        }
        return values, {"router": sha256(model_path), "hybrid_selection_rate": float(select_hybrid.mean())}
    return _load_or_predict_base(system, data, dates, issues, device, batch_size)


def _metrics(target, prediction):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/decision_aware/austria_external_15min_v1.json")
    parser.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    parser.add_argument("--output-dir", default="data/results/austria_external_15min_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-days", type=int, default=0)
    args = parser.parse_args()
    requested = [item.strip() for item in args.systems.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(ALL_SYSTEMS))
    if unknown:
        raise ValueError(f"未知系统: {unknown}")
    data, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    quarter = data.evaluation_quarters(complete_settlement=True)
    if args.smoke_days:
        local_dates = pd.DatetimeIndex(quarter["timestamp_local"]).date
        keep = sorted(set(local_dates))[:args.smoke_days]
        quarter = quarter.loc[np.isin(local_dates, keep)].copy()
    if quarter["不平衡最终结算价格 (EUR/MWh)"].isna().any():
        raise ValueError("完整双结算区间仍有RT缺失")
    dates, issues, local = _forecast_axes(data, quarter)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = ROOT / args.output_dir
    cache = output_dir / "forecast_cache"
    cache.mkdir(parents=True, exist_ok=True)
    original_hash = data.raw_sha256
    summaries = []
    for system in requested:
        started = perf_counter()
        cache_path = cache / f"{system}.npz"
        if cache_path.exists() and not args.force:
            values, metadata = _load_forecast(cache_path)
            evidence = metadata.get("evidence", {})
        else:
            values, evidence = _prepare_derived(system, cache, data, dates, issues, device, args.batch_size)
            _save_forecast(cache_path, values, {
                "system": system,
                "protocol": protocol["protocol_version"],
                "austria_training": False,
                "austria_seed_selection": False,
                "evidence": evidence,
            })
        # All model outputs are in the US training currency.  Scores and
        # settlement are converted back before touching Austria truth.
        eur = {task: values[task].astype(np.float64) / data.eur_to_usd for task in TASKS}
        da_q = expand_daily_hourly_to_quarters(dates, eur["da"], local)
        rtda_q = expand_daily_hourly_to_quarters(dates, eur["rt_at_da"], local)
        rt_windows = expand_hourly_rt_to_quarter_windows(issues, eur["rt"], quarter.index)
        rt_first_q = np.asarray([value[0] for value in rt_windows])
        true_da = quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
        true_rt = quarter["不平衡最终结算价格 (EUR/MWh)"].to_numpy(np.float64)
        settlement = quarter_hour_backtest(
            da_q, rtda_q, rt_windows, true_da, true_rt, local,
            protocol["battery"],
        )
        summary = {
            "system": system,
            "protocol": protocol["protocol_version"],
            "strict_external_test": True,
            "austria_training_used": False,
            "austria_finetuning_used": False,
            "austria_seed_or_model_selection_used": False,
            "native_interval_minutes": 15,
            "days": settlement["days"],
            "quarter_rows": len(quarter),
            "forecast_metrics_eur_per_mwh": {
                "da": _metrics(true_da, da_q),
                "rt_at_da": _metrics(true_rt, rtda_q),
                "rolling_rt_executed_step": _metrics(true_rt, rt_first_q),
            },
            "settlement": settlement,
            "evidence": evidence,
            "runtime_seconds": float(perf_counter() - started),
        }
        (output_dir / f"{system}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(json.dumps({
            "event": "austria_system_complete", "system": system,
            "profit_eur_per_day": settlement["mean_daily_net_revenue_eur"],
            "da_mae": summary["forecast_metrics_eur_per_mwh"]["da"]["mae"],
            "rt_mae": summary["forecast_metrics_eur_per_mwh"]["rolling_rt_executed_step"]["mae"],
            "runtime_seconds": summary["runtime_seconds"],
        }, ensure_ascii=False), flush=True)
        if sha256(data.raw_path) != original_hash:
            raise RuntimeError("奥地利原始文件在评价过程中被修改")
    ranking = sorted([
        {"rank": 0, "system": item["system"], "mean_daily_net_revenue_eur": item["settlement"]["mean_daily_net_revenue_eur"]}
        for item in summaries
    ], key=lambda item: item["mean_daily_net_revenue_eur"], reverse=True)
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    report = {
        "protocol": protocol["protocol_version"],
        "status": "external_report_only_no_selection",
        "selection_allowed": False,
        "raw_sha256_before_and_after": original_hash,
        "evaluation_local": [str(local[0]), str(local[-1])],
        "days": len(dates),
        "quarter_rows": len(quarter),
        "ranking": ranking,
        "ineligible": protocol["ineligible_systems"],
        "systems": {item["system"]: {
            "forecast_metrics_eur_per_mwh": item["forecast_metrics_eur_per_mwh"],
            "settlement": {key: value for key, value in item["settlement"].items() if key != "daily"},
            "runtime_seconds": item["runtime_seconds"],
        } for item in summaries},
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "austria_external_complete", "ranking": ranking}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # No optimizer, fit, train, backward, or random seed exists in this entrypoint.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
