#!/usr/bin/env python
"""Evaluate the frozen Router and decision-aware DA forecasts on 365 Austria days.

September 2026 DA truth is present in the frozen Austria parquet, but its RT-price
column is empty.  The frozen DA Transformer consumes the previous 168 hours of RT
prices, so this script supplies the later-published official APCS September values
as causal history.  It never rewrites the frozen parquet and does not run settlement.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    PRICE_RT,
    expand_daily_hourly_to_quarters,
    sha256,
)
from decision_aware.dlinear import DLinearForecaster  # noqa: E402
from decision_aware.robust_ensemble import (  # noqa: E402
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)
from decision_aware.tail_expert_router import (  # noqa: E402
    build_feature_cube,
    router_probability,
)
from scripts.decision_aware.run_austria_external_15min import (  # noqa: E402
    MODEL_CLASS,
    ORIGINAL_ROOTS,
    PREDICTION_KEY,
    V4_ROOTS,
    _checkpoint_config,
    _forecast_axes,
    _load_forecast,
    _run_batched_samples,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
    group_counts,
)


DA_TASKS = ("da", "rt_at_da")
APCS_SHEET = "Werte - Values"
APCS_UTC_END = "Endzeitpunkt \n(UTC)"
APCS_IMBALANCE = "Ausgleichsenergiepreis (=Preis für Abweichung)"


def load_apcs_imbalance(path: Path) -> pd.Series:
    """Load quarter-start UTC imbalance prices from an APCS control-area workbook."""
    raw = pd.read_excel(path, sheet_name=APCS_SHEET, header=None)
    labels = raw.iloc[0].astype(str)
    try:
        utc_column = int(labels[labels == APCS_UTC_END].index[0])
        price_column = int(labels[labels == APCS_IMBALANCE].index[0])
    except IndexError as exc:
        raise ValueError(f"APCS工作簿缺少预期字段: {path}") from exc
    end_utc = pd.to_datetime(raw.iloc[4:, utc_column], errors="coerce", utc=True)
    price = pd.to_numeric(raw.iloc[4:, price_column], errors="coerce")
    frame = pd.DataFrame(
        {
            "utc": end_utc - pd.Timedelta(minutes=15),
            "price": price,
        }
    ).dropna()
    if frame.empty or frame["utc"].duplicated().any():
        raise ValueError(f"APCS工作簿时间为空或重复: {path}")
    frame = frame.sort_values("utc")
    if not (frame["utc"].diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"APCS工作簿不是连续15分钟序列: {path}")
    return frame.set_index("utc")["price"].astype(np.float64)


def extend_rt_history(
    data: AustriaExternalData,
    september: pd.Series,
) -> AustriaExternalData:
    """Return an in-memory view with missing September RT history filled."""
    quarter = data.quarter.copy()
    missing = quarter[PRICE_RT].isna()
    matched = september.reindex(quarter.index)
    if int(missing.sum()) != 2880 or int(matched.loc[missing].notna().sum()) != 2880:
        raise ValueError("APCS 2026年9月数据未完整覆盖原始文件的2,880个RT缺失点")
    # The frozen parquet stores this field to two decimals.  Match that contract.
    quarter.loc[missing, PRICE_RT] = matched.loc[missing].round(2).to_numpy()
    if quarter[PRICE_RT].isna().any():
        raise ValueError("补充后RT历史仍有缺失")
    hourly = data.hourly.copy()
    hourly["price_rt_eur"] = quarter[PRICE_RT].resample("1h").mean()
    hourly["price_rt"] = hourly["price_rt_eur"] * data.eur_to_usd
    return AustriaExternalData(
        raw_path=data.raw_path,
        raw_sha256=data.raw_sha256,
        timezone=data.timezone,
        eur_to_usd=data.eur_to_usd,
        quarter=quarter,
        hourly=hourly,
    )


def verify_overlap(data: AustriaExternalData, august: pd.Series) -> dict:
    joined = pd.DataFrame(
        {
            "frozen": data.quarter[PRICE_RT],
            "apcs": august,
        }
    ).dropna()
    if len(joined) != 2976:
        raise ValueError(f"APCS 2026年8月重叠点应为2,976，实际为{len(joined)}")
    difference = (joined["frozen"] - joined["apcs"]).abs()
    if float(difference.max()) > 0.006:
        raise ValueError("APCS不平衡价格与冻结文件口径不一致")
    return {
        "overlap_quarters": int(len(joined)),
        "max_absolute_difference_eur_per_mwh": float(difference.max()),
        "mean_absolute_difference_eur_per_mwh": float(difference.mean()),
        "interpretation": "same field; differences are explained by two-decimal rounding",
    }


def load_prefix_pair(cache_dir: Path, name: str, expected_days: int) -> dict[str, np.ndarray]:
    values, _metadata = _load_forecast(cache_dir / f"{name}.npz")
    pair = {task: values[task].astype(np.float32) for task in DA_TASKS}
    for task, value in pair.items():
        if value.shape != (expected_days, 24):
            raise ValueError(f"{name}/{task}缓存形状错误: {value.shape}")
    return pair


def append_pair(prefix: dict[str, np.ndarray], suffix: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {task: np.concatenate([prefix[task], suffix[task]], axis=0) for task in DA_TASKS}


def predict_transformer_pair(
    data: AustriaExternalData,
    dates: list,
    templates: dict[str, str],
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {}
    for task in DA_TASKS:
        path = ROOT / templates[task].format(seed=seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = _checkpoint_config(payload)
        if config.global_use_entity_embeddings or config.context_len != 168:
            raise ValueError("奥地利仅允许无ID、168小时冻结Transformer")
        model = MODEL_CLASS[task](config).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()

        def predict(batch, model=model, task=task):
            return model(batch)[PREDICTION_KEY[task]]

        output[task] = _run_batched_samples(
            data, task, dates, predict, device, batch_size
        )
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def predict_dlinear_pair(
    data: AustriaExternalData,
    dates: list,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {}
    for task in DA_TASKS:
        path = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/dlinear/{task}_seed0.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = DLinearForecaster(168, 24, 25).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        context_key = "price_da_ctx" if task == "da" else "price_rt_ctx"
        mean_key = "price_da_mean" if task == "da" else "price_rt_mean"
        std_key = "price_da_std" if task == "da" else "price_rt_std"

        def predict(batch, model=model, context_key=context_key, mean_key=mean_key, std_key=std_key):
            return (
                model(batch[context_key]) * batch[std_key].reshape(-1, 1)
                + batch[mean_key].reshape(-1, 1)
            )

        output[task] = _run_batched_samples(
            data, task, dates, predict, device, batch_size
        )
        del model, payload
    return output


def calibrate_robust(members: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    checkpoint_path = ROOT / "data/checkpoints/global_robust_ensemble_final_v4/calibrator.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    calibrator = EconomicEnsembleCalibrator(
        EnsembleCalibratorConfig(**checkpoint["calibrator_config"])
    )
    calibrator.load_state_dict(checkpoint["model_state"], strict=True)
    calibrator.eval()
    da = torch.from_numpy(np.stack([item["da"] for item in members], axis=1)).float()
    rtda = torch.from_numpy(
        np.stack([item["rt_at_da"] for item in members], axis=1)
    ).float()
    with torch.inference_mode():
        calibrated_da, calibrated_rtda = calibrator.calibrate_da_pair(da, rtda)
    return {
        "da": calibrated_da.numpy(),
        "rt_at_da": calibrated_rtda.numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--september-apcs", default="data/Austria/Statistik-RZ_202609_NEU.xlsx"
    )
    parser.add_argument(
        "--august-apcs", default="data/Austria/Statistik-RZ_202608_NEU.xlsx"
    )
    parser.add_argument(
        "--prefix-cache", default="data/results/austria_external_15min_v1/forecast_cache"
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_da_mae_365_v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    september_path = ROOT / args.september_apcs
    august_path = ROOT / args.august_apcs
    september = load_apcs_imbalance(september_path)
    august = load_apcs_imbalance(august_path)
    overlap = verify_overlap(data, august)
    extended = extend_rt_history(data, september)

    prefix_quarter = data.evaluation_quarters(complete_settlement=True)
    full_quarter = extended.evaluation_quarters(complete_settlement=False)
    prefix_dates, _prefix_issues, _prefix_local = _forecast_axes(data, prefix_quarter)
    dates, _issues, local = _forecast_axes(extended, full_quarter)
    if dates[: len(prefix_dates)] != prefix_dates or len(dates) != 365:
        raise ValueError("365天日期轴与冻结335天前缀不一致")
    suffix_dates = dates[len(prefix_dates):]
    if len(suffix_dates) != 30:
        raise ValueError(f"应补算30天，实际为{len(suffix_dates)}")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_dir = ROOT / args.prefix_cache

    original_members = []
    for seed in (0, 1, 2):
        prefix = load_prefix_pair(cache_dir, f"transformer_seed{seed}", len(prefix_dates))
        suffix = predict_transformer_pair(
            extended, suffix_dates, ORIGINAL_ROOTS, seed, device, args.batch_size
        )
        original_members.append(append_pair(prefix, suffix))
    original_equal = {
        task: np.mean([item[task] for item in original_members], axis=0)
        for task in DA_TASKS
    }

    dlinear = append_pair(
        load_prefix_pair(cache_dir, "dlinear", len(prefix_dates)),
        predict_dlinear_pair(extended, suffix_dates, device, args.batch_size),
    )

    v4_members = []
    for seed in (0, 1, 2):
        prefix = load_prefix_pair(
            cache_dir, f"decision_aware_v4_seed{seed}_member", len(prefix_dates)
        )
        suffix = predict_transformer_pair(
            extended, suffix_dates, V4_ROOTS, seed, device, args.batch_size
        )
        v4_members.append(append_pair(prefix, suffix))
    robust = calibrate_robust(v4_members)
    decision_aware = {
        task: (2.0 / 3.0) * v4_members[0][task] + (1.0 / 3.0) * robust[task]
        for task in DA_TASKS
    }

    cube, _ = build_feature_cube(
        original_equal["da"][None],
        original_equal["rt_at_da"][None],
        dlinear["da"][None],
        dlinear["rt_at_da"][None],
        np.asarray([str(value) for value in dates]),
    )
    router_path = ROOT / "data/results/tail_expert_router_v1/router_refit_2023_2024.joblib"
    router = joblib.load(router_path)
    select_hybrid = router_probability(router, cube.reshape(len(dates), -1)) >= 0.70
    router_values = {
        task: np.where(
            select_hybrid[:, None], dlinear[task], original_equal[task]
        )
        for task in DA_TASKS
    }

    realized = full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    groups = classify_days(
        local, realized, spike_floor=SPIKE_FLOOR_EUR_PER_MWH
    )
    systems = {
        "decision_aware_v4_final_2to1_blend": decision_aware,
        "three_feature_tail_expert_router": router_values,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, values in systems.items():
        da_quarter = expand_daily_hourly_to_quarters(
            dates, values["da"].astype(np.float64) / extended.eur_to_usd, local
        )
        result[name] = forecast_metrics(local, realized, da_quarter, groups)
        np.savez_compressed(
            output_dir / f"{name}.npz",
            delivery_dates=np.asarray([str(value) for value in dates]),
            da_usd_per_mwh=values["da"].astype(np.float32),
            rt_at_da_usd_per_mwh=values["rt_at_da"].astype(np.float32),
        )

    summary = {
        "protocol": "austria_da_mae_365_v1",
        "status": "complete",
        "selection_allowed": False,
        "austria_training_or_tuning_used": False,
        "evaluation": {
            "days": int(len(dates)),
            "quarters": int(len(full_quarter)),
            "local_interval": [str(local[0]), str(local[-1])],
            "mae_aggregation": "point-weighted over native 15-minute rows",
            "dst_rule": "native local days retain 92, 96, or 100 quarters",
            "group_precedence": "negative first; otherwise spike; otherwise ordinary",
            "spike_floor_eur_per_mwh": SPIKE_FLOOR_EUR_PER_MWH,
            "group_counts": group_counts(groups),
        },
        "rt_history_extension": {
            "purpose": "DA-model causal input only; no settlement is run",
            "september_source": "APCS Statistik-RZ September 2026",
            "source_url": "https://www.apcs.at/fileadmin/user_upload/APCS/Daten_Preise/RZ-Statistik/Statistik-RZ_202609_NEU.xlsx",
            "source_sha256": sha256(september_path),
            "quarters_filled": 2880,
            "rounding": "rounded to two decimals to match the frozen parquet field",
            "august_overlap_verification": overlap,
            "frozen_parquet_modified": False,
        },
        "router": {
            "hybrid_selection_days": int(select_hybrid.sum()),
            "hybrid_selection_rate": float(select_hybrid.mean()),
            "router_sha256": sha256(router_path),
        },
        "systems": result,
        "raw_parquet_sha256": protocol["raw_sha256"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
