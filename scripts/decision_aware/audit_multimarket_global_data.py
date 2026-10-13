#!/usr/bin/env python
"""只读审计四市场多节点小时数据，为 global 模型冻结数据合同。

脚本不修改 ``data/markets``。默认把机器可读结果写到
``data/results/multimarket_global_data_audit.json``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
MARKET_TIMEZONES = {
    "CAISO": "America/Los_Angeles",
    "ERCOT": "America/Chicago",
    "NYISO": "America/New_York",
    "PJM": "America/New_York",
}
REQUIRED_COLUMNS = (
    "timestamp_local",
    "timestamp_utc",
    "node",
    "actual_load_mw",
    "day_ahead_price_usd_mwh",
    "real_time_price_usd_mwh",
    "wind_actual_mw",
    "solar_actual_mw",
    "hour_local",
    "is_weekend",
    "is_holiday",
)
REPEATED_FEATURE_CANDIDATES = (
    "actual_load_mw",
    "wind_actual_mw",
    "solar_actual_mw",
    "actual_temperature_2m_c",
    "actual_relative_humidity_2m_pct",
    "actual_dew_point_2m_c",
    "actual_surface_pressure_hpa",
    "actual_cloud_cover_pct",
    "actual_precipitation_mm",
    "actual_wind_speed_10m_ms",
    "actual_wind_gusts_10m_ms",
    "actual_shortwave_radiation_wm2",
    "forecast_temperature_2m_c",
    "forecast_relative_humidity_2m_pct",
    "forecast_dew_point_2m_c",
    "forecast_cloud_cover_pct",
    "forecast_precipitation_mm",
    "forecast_wind_speed_10m_ms",
    "forecast_wind_gust_10m_ms",
    "forecast_shortwave_radiation_wm2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


def _node_gap_summary(group: pd.DataFrame, timezone: str) -> dict:
    timestamps = pd.DatetimeIndex(group["timestamp_utc"].sort_values())
    deltas = timestamps.to_series().diff().dropna()
    bad = deltas[deltas != pd.Timedelta(hours=1)]
    train_start_utc = pd.Timestamp("2020-01-01").tz_localize(timezone).tz_convert("UTC")
    train_end_utc = pd.Timestamp("2023-01-01").tz_localize(timezone).tz_convert("UTC")
    train_mask = (
        (group["timestamp_utc"] >= train_start_utc)
        & (group["timestamp_utc"] < train_end_utc)
    )
    train = group.loc[train_mask]
    return {
        "rows": int(len(group)),
        "utc_start": _iso(timestamps.min()),
        "utc_end": _iso(timestamps.max()),
        "local_start": _iso(timestamps.min().tz_convert(timezone)),
        "local_end": _iso(timestamps.max().tz_convert(timezone)),
        "non_hourly_gap_count": int(len(bad)),
        "largest_gap_hours": (
            float(deltas.max() / pd.Timedelta(hours=1)) if len(deltas) else 0.0
        ),
        "internal_train_2020_2022_price_stats": {
            "day_ahead_mean": float(train["day_ahead_price_usd_mwh"].mean()),
            "day_ahead_std": float(train["day_ahead_price_usd_mwh"].std()),
            "day_ahead_min": float(train["day_ahead_price_usd_mwh"].min()),
            "day_ahead_max": float(train["day_ahead_price_usd_mwh"].max()),
            "real_time_mean": float(train["real_time_price_usd_mwh"].mean()),
            "real_time_std": float(train["real_time_price_usd_mwh"].std()),
            "real_time_min": float(train["real_time_price_usd_mwh"].min()),
            "real_time_max": float(train["real_time_price_usd_mwh"].max()),
        },
    }


def _repetition_summary(df: pd.DataFrame, column: str) -> dict:
    per_time_unique = df.groupby("timestamp_utc", sort=False)[column].nunique(
        dropna=False
    )
    repeated = per_time_unique.eq(1)
    return {
        "timestamps": int(len(per_time_unique)),
        "all_nodes_identical_timestamps": int(repeated.sum()),
        "all_nodes_identical_fraction": float(repeated.mean()),
        "max_distinct_node_values_same_timestamp": int(per_time_unique.max()),
        "classification": (
            "market_level_repeated"
            if bool(repeated.all())
            else "node_varying_or_partially_repeated"
        ),
    }


def audit_market(market: str, expected_nodes: list[str]) -> dict:
    path = (
        ROOT
        / "data"
        / "markets"
        / market
        / f"{market}_统一小时数据_20200101_20260601.parquet"
    )
    df = pd.read_parquet(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{market} 缺少必要字段: {missing_columns}")

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    local = pd.to_datetime(df["timestamp_local"], utc=True).dt.tz_convert(
        MARKET_TIMEZONES[market]
    )
    derived_local = df["timestamp_utc"].dt.tz_convert(MARKET_TIMEZONES[market])
    actual_nodes = sorted(df["node"].astype(str).unique().tolist())
    node_set_match = actual_nodes == sorted(expected_nodes)

    duplicate_rows = int(
        df.duplicated(subset=["node", "timestamp_utc"], keep=False).sum()
    )
    expected_rows = len(actual_nodes) * df["timestamp_utc"].nunique()
    missing_rates = {
        column: float(rate)
        for column, rate in df.isna().mean().items()
        if rate > 0.0
    }
    node_summaries = {
        str(node): _node_gap_summary(group, MARKET_TIMEZONES[market])
        for node, group in df.groupby("node", sort=True)
    }
    repetition = {
        column: _repetition_summary(df, column)
        for column in REPEATED_FEATURE_CANDIDATES
        if column in df.columns
    }

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "node_count": int(len(actual_nodes)),
        "nodes": actual_nodes,
        "expected_nodes": sorted(expected_nodes),
        "node_set_matches_configs_nodes_yaml": node_set_match,
        "missing_expected_nodes": sorted(set(expected_nodes) - set(actual_nodes)),
        "unexpected_nodes": sorted(set(actual_nodes) - set(expected_nodes)),
        "utc_start": _iso(df["timestamp_utc"].min()),
        "utc_end": _iso(df["timestamp_utc"].max()),
        "unique_utc_timestamps": int(df["timestamp_utc"].nunique()),
        "rectangular_panel_expected_rows": int(expected_rows),
        "is_rectangular_panel": bool(len(df) == expected_rows),
        "duplicate_node_timestamp_rows": duplicate_rows,
        "missing_rates_nonzero": missing_rates,
        "timestamp_local_matches_market_timezone": bool(
            np.array_equal(local.astype(str).to_numpy(), derived_local.astype(str).to_numpy())
        ),
        "hour_local_matches_timestamp_local": bool(
            np.array_equal(df["hour_local"].to_numpy(), local.dt.hour.to_numpy())
        ),
        "node_time_ranges_and_gaps": node_summaries,
        "feature_repetition_across_nodes": repetition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/results/multimarket_global_data_audit.json",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    nodes_path = ROOT / "configs" / "nodes.yaml"
    nodes_config = yaml.safe_load(nodes_path.read_text(encoding="utf-8"))
    markets = {}
    for market in MARKET_TIMEZONES:
        markets[market] = audit_market(market, nodes_config[market]["all"])

    report = {
        "audit_version": 1,
        "scope": "read-only four-market global data contract audit",
        "source_nodes_config": nodes_path.relative_to(ROOT).as_posix(),
        "markets": markets,
        "totals": {
            "markets": len(markets),
            "nodes": int(sum(item["node_count"] for item in markets.values())),
            "rows": int(sum(item["rows"] for item in markets.values())),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.summary_only:
        summary = {
            market: {
                key: item[key]
                for key in (
                    "node_count",
                    "rows",
                    "utc_start",
                    "utc_end",
                    "is_rectangular_panel",
                    "duplicate_node_timestamp_rows",
                    "missing_rates_nonzero",
                    "timestamp_local_matches_market_timezone",
                    "hour_local_matches_timestamp_local",
                    "node_set_matches_configs_nodes_yaml",
                )
            }
            for market, item in markets.items()
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
