#!/usr/bin/env python
"""Run 365-day native-quarter settlement for the two frozen project systems."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    expand_daily_hourly_to_quarters,
    expand_hourly_rt_to_quarter_windows,
    quarter_hour_backtest,
    sha256,
)
from decision_aware.robust_ensemble import (  # noqa: E402
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    extend_rt_history,
    load_apcs_imbalance,
    verify_overlap,
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
    group_counts,
    settlement_metrics,
)


SYSTEMS = (
    "decision_aware_v4_final_2to1_blend",
    "three_feature_tail_expert_router",
)


def load_energy_charts_price(path: Path) -> tuple[pd.Series, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    values = np.asarray(payload["price"], dtype=np.float64)
    if len(timestamps) != len(values) or len(values) == 0:
        raise ValueError(f"Energy-Charts价格长度错误: {path}")
    series = pd.Series(values, index=timestamps, name="price_da_eur").sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Energy-Charts时间重复: {path}")
    if not (series.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"Energy-Charts不是连续15分钟价格: {path}")
    return series, {
        "license_info": payload.get("license_info"),
        "unit": payload.get("unit"),
        "sha256": sha256(path),
    }


def verify_da_overlap(data: AustriaExternalData, september: pd.Series) -> dict:
    frozen = data.quarter["日前价格 (EUR/MWh)"].reindex(september.index)
    if frozen.isna().any() or len(frozen) != 2880:
        raise ValueError("Energy-Charts 9月DA未与冻结文件完整重叠")
    difference = (frozen - september).abs()
    if float(difference.max()) > 1e-9:
        raise ValueError("Energy-Charts DA价格与冻结文件口径不一致")
    return {
        "overlap_quarters": int(len(difference)),
        "max_absolute_difference_eur_per_mwh": float(difference.max()),
    }


def add_future_da_boundary(
    data: AustriaExternalData,
    october_first: pd.Series,
) -> AustriaExternalData:
    """Append known next-day DA prices to the hourly model interface only."""
    hourly_da = october_first.resample("1h").mean()
    expected_start = data.hourly.index[-1] + pd.Timedelta(hours=1)
    if hourly_da.index[0] != expected_start or len(hourly_da) != 24:
        raise ValueError("10月1日DA边界与冻结小时接口不连续")
    index = data.hourly.index.union(hourly_da.index)
    hourly = data.hourly.reindex(index).copy()
    hourly.loc[hourly_da.index, "price_da_eur"] = hourly_da.to_numpy()
    hourly.loc[hourly_da.index, "price_da"] = hourly_da.to_numpy() * data.eur_to_usd
    return AustriaExternalData(
        raw_path=data.raw_path,
        raw_sha256=data.raw_sha256,
        timezone=data.timezone,
        eur_to_usd=data.eur_to_usd,
        quarter=data.quarter,
        hourly=hourly,
    )


def predict_rt_suffix(
    data: AustriaExternalData,
    issues: pd.DatetimeIndex,
    template: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    path = ROOT / template.format(seed=seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(payload)
    if config.global_use_entity_embeddings or config.context_len != 168:
        raise ValueError("奥地利仅允许无ID、168小时冻结Transformer")
    model = MODEL_CLASS["rt"](config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    def predict(batch):
        return model(batch)[PREDICTION_KEY["rt"]]

    output = _run_batched_samples(data, "rt", issues, predict, device, batch_size)
    del model, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def append_rt_prefix(
    cache_dir: Path,
    name: str,
    suffix: np.ndarray,
    expected_prefix: int,
) -> np.ndarray:
    values, _metadata = _load_forecast(cache_dir / f"{name}.npz")
    prefix = values["rt"].astype(np.float32)
    if prefix.shape != (expected_prefix, 4):
        raise ValueError(f"{name} RT缓存形状错误: {prefix.shape}")
    return np.concatenate([prefix, suffix.astype(np.float32)], axis=0)


def robust_rt(
    members: list[np.ndarray],
    issues: pd.DatetimeIndex,
    timezone: str,
) -> np.ndarray:
    checkpoint_path = ROOT / "data/checkpoints/global_robust_ensemble_final_v4/calibrator.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    calibrator = EconomicEnsembleCalibrator(
        EnsembleCalibratorConfig(**checkpoint["calibrator_config"])
    )
    calibrator.load_state_dict(checkpoint["model_state"], strict=True)
    calibrator.eval()
    stack = np.stack(members, axis=1)
    weights = calibrator.weights()[2].detach().numpy()
    prediction = np.einsum("m,imk->ik", weights, stack)
    interpretation = calibrator.interpretation()
    local_hours = issues.tz_convert(timezone).hour.to_numpy()
    prediction *= np.asarray(interpretation["rt_scale"])[local_hours, None]
    prediction += np.asarray(interpretation["rt_bias_usd_per_mwh"])[local_hours, None]
    return prediction.astype(np.float32)


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
        "--september-da-check", default="data/Austria/energy_charts_at_202609.json"
    )
    parser.add_argument(
        "--october-first-da", default="data/Austria/energy_charts_at_20261001.json"
    )
    parser.add_argument(
        "--prefix-cache", default="data/results/austria_external_15min_v1/forecast_cache"
    )
    parser.add_argument(
        "--da-cache", default="data/results/austria_da_mae_365_v1"
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_settlement_365_v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    base, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    september_rt_path = ROOT / args.september_apcs
    august_rt_path = ROOT / args.august_apcs
    september_rt = load_apcs_imbalance(september_rt_path)
    august_rt = load_apcs_imbalance(august_rt_path)
    rt_overlap = verify_overlap(base, august_rt)
    extended = extend_rt_history(base, september_rt)

    september_da, september_da_meta = load_energy_charts_price(
        ROOT / args.september_da_check
    )
    october_da, october_da_meta = load_energy_charts_price(
        ROOT / args.october_first_da
    )
    da_overlap = verify_da_overlap(base, september_da)
    prediction_data = add_future_da_boundary(extended, october_da)

    prefix_quarter = base.evaluation_quarters(complete_settlement=True)
    full_quarter = extended.evaluation_quarters(complete_settlement=False)
    prefix_dates, prefix_issues, _ = _forecast_axes(base, prefix_quarter)
    dates, issues, local = _forecast_axes(extended, full_quarter)
    if len(dates) != 365 or len(issues) != 8760:
        raise ValueError("365天结算轴错误")
    suffix_issues = issues[len(prefix_issues):]
    if len(prefix_dates) != 335 or len(prefix_issues) != 8040 or len(suffix_issues) != 720:
        raise ValueError("335天冻结前缀与30天扩展轴错误")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cache_dir = ROOT / args.prefix_cache
    started = perf_counter()

    original_rt_members = []
    for seed in (0, 1, 2):
        suffix = predict_rt_suffix(
            prediction_data,
            suffix_issues,
            ORIGINAL_ROOTS["rt"],
            seed,
            device,
            args.batch_size,
        )
        original_rt_members.append(
            append_rt_prefix(cache_dir, f"transformer_seed{seed}", suffix, len(prefix_issues))
        )
    original_rt = np.mean(original_rt_members, axis=0).astype(np.float32)

    v4_rt_members = []
    for seed in (0, 1, 2):
        suffix = predict_rt_suffix(
            prediction_data,
            suffix_issues,
            V4_ROOTS["rt"],
            seed,
            device,
            args.batch_size,
        )
        v4_rt_members.append(
            append_rt_prefix(
                cache_dir,
                f"decision_aware_v4_seed{seed}_member",
                suffix,
                len(prefix_issues),
            )
        )
    decision_rt = (
        (2.0 / 3.0) * v4_rt_members[0]
        + (1.0 / 3.0) * robust_rt(v4_rt_members, issues, extended.timezone)
    ).astype(np.float32)

    rt_by_system = {
        "decision_aware_v4_final_2to1_blend": decision_rt,
        "three_feature_tail_expert_router": original_rt,
    }
    realized_da = full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    realized_rt = full_quarter["不平衡最终结算价格 (EUR/MWh)"].to_numpy(np.float64)
    if not np.isfinite(realized_rt).all():
        raise ValueError("365天结算RT真值仍不完整")
    groups = classify_days(local, realized_da, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    systems = {}
    for system in SYSTEMS:
        with np.load(ROOT / args.da_cache / f"{system}.npz", allow_pickle=False) as archive:
            da = archive["da_usd_per_mwh"].astype(np.float64) / extended.eur_to_usd
            rt_at_da = archive["rt_at_da_usd_per_mwh"].astype(np.float64) / extended.eur_to_usd
        rolling_rt = rt_by_system[system].astype(np.float64) / extended.eur_to_usd
        da_quarter = expand_daily_hourly_to_quarters(dates, da, local)
        rt_at_da_quarter = expand_daily_hourly_to_quarters(dates, rt_at_da, local)
        rt_windows = expand_hourly_rt_to_quarter_windows(issues, rolling_rt, full_quarter.index)
        settlement = quarter_hour_backtest(
            da_quarter,
            rt_at_da_quarter,
            rt_windows,
            realized_da,
            realized_rt,
            local,
            protocol["battery"],
        )
        report = {"settlement": settlement}
        systems[system] = {
            "settlement": {key: value for key, value in settlement.items() if key != "daily"},
            "by_day_group": settlement_metrics(report, groups),
        }
        (output_dir / f"{system}.json").write_text(
            json.dumps(
                {
                    "system": system,
                    "protocol": "austria_settlement_365_v1",
                    "settlement": settlement,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "system_complete",
                    "system": system,
                    "mean_daily_net_revenue_eur": settlement[
                        "mean_daily_net_revenue_eur"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "protocol": "austria_settlement_365_v1",
        "status": "complete",
        "selection_allowed": False,
        "austria_training_or_tuning_used": False,
        "evaluation": {
            "days": 365,
            "quarters": int(len(full_quarter)),
            "local_interval": [str(local[0]), str(local[-1])],
            "forecast_resolution": "frozen hourly forecasts repeated over four native quarter-hours",
            "settlement_resolution": "native 15-minute LP/MPC and cash settlement",
            "dst_rule": "native local days retain 92, 96, or 100 quarters",
            "group_counts": group_counts(groups),
        },
        "sources": {
            "september_rt": {
                "source": "APCS Statistik-RZ September 2026",
                "sha256": sha256(september_rt_path),
                "august_overlap_verification": rt_overlap,
            },
            "october_first_da_boundary": {
                "source": "Energy-Charts AT day-ahead price",
                "purpose": "known future DA input for the last three rolling-RT issue hours only",
                **october_da_meta,
                "september_exact_overlap_verification": da_overlap,
                "september_source_metadata": september_da_meta,
            },
            "frozen_parquet_modified": False,
        },
        "battery": protocol["battery"],
        "systems": systems,
        "runtime_seconds": float(perf_counter() - started),
        "raw_parquet_sha256": protocol["raw_sha256"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
