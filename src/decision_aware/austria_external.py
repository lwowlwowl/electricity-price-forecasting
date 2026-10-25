"""Leakage-safe Austria 15-minute external-test data and settlement helpers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .continuous_mpc import (
    solve_fixed_da_penalty_lp_window,
    solve_predicted_da_penalty_lp_day,
)
from .policy import BESSSimulator
from .portable_normalization import normalize, robust_context_stats


PRICE_DA = "日前价格 (EUR/MWh)"
PRICE_RT = "不平衡最终结算价格 (EUR/MWh)"
TIMESTAMP = "UTC时间"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar_features(index: Iterable[pd.Timestamp]) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    hour = idx.hour.to_numpy(dtype=np.float64)
    dow = idx.dayofweek.to_numpy(dtype=np.float64)
    month = idx.month.to_numpy(dtype=np.float64)
    return np.column_stack([
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * dow / 7.0),
        np.cos(2.0 * np.pi * dow / 7.0),
        np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
        np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
        (dow >= 5.0).astype(np.float64),
        np.zeros(len(idx), dtype=np.float64),
    ]).astype(np.float32)


@dataclass(frozen=True)
class AustriaExternalData:
    raw_path: Path
    raw_sha256: str
    timezone: str
    eur_to_usd: float
    quarter: pd.DataFrame
    hourly: pd.DataFrame

    @classmethod
    def load(cls, root: Path, protocol_path: Path) -> tuple["AustriaExternalData", dict]:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["protocol_version"] != "austria_external_15min_v1":
            raise ValueError("奥地利外部测试协议版本错误")
        strict = protocol["strict_external_test"]
        if any(strict[key] for key in strict if key.endswith("_allowed")):
            raise ValueError("奥地利协议错误地允许了训练或选择")
        raw_path = root / protocol["raw_parquet"]
        digest = sha256(raw_path)
        if digest != protocol["raw_sha256"]:
            raise ValueError(f"奥地利原始文件哈希变化: {digest}")
        raw = pd.read_parquet(raw_path).copy()
        required = {
            TIMESTAMP, PRICE_DA, PRICE_RT, "实际负荷 (MW)",
            "风电实际 (MW)", "光伏实际 (MW)",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"奥地利数据缺列: {missing}")
        utc = pd.DatetimeIndex(raw[TIMESTAMP])
        if utc.tz is None:
            raise ValueError("奥地利UTC时间必须带时区")
        utc = utc.tz_convert("UTC")
        if not utc.is_unique or not utc.is_monotonic_increasing:
            raise ValueError("奥地利UTC时间必须唯一且递增")
        if not np.all(np.diff(utc.asi8) == pd.Timedelta(minutes=15).value):
            raise ValueError("奥地利原始数据不是无缺口15分钟时间线")
        raw.index = utc
        raw["timestamp_local"] = utc.tz_convert(protocol["timezone"])

        # The frozen US models operate hourly.  This is an interface view only:
        # the raw parquet is never rewritten and final settlement remains 15-minute.
        hourly = pd.DataFrame(index=raw.resample("1h").size().index)
        for target, source in {
            "price_da_eur": PRICE_DA,
            "price_rt_eur": PRICE_RT,
            "load": "实际负荷 (MW)",
            "wind": "风电实际 (MW)",
            "solar": "光伏实际 (MW)",
        }.items():
            hourly[target] = raw[source].resample("1h").mean()
        hourly["price_da"] = hourly["price_da_eur"] * float(protocol["currency_bridge"]["eur_to_usd"])
        hourly["price_rt"] = hourly["price_rt_eur"] * float(protocol["currency_bridge"]["eur_to_usd"])
        local_hour = hourly.index.tz_convert(protocol["timezone"])
        cal = calendar_features(local_hour)
        for position, name in enumerate((
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin",
            "month_cos", "is_weekend", "is_holiday",
        )):
            hourly[name] = cal[:, position]
        return cls(
            raw_path=raw_path,
            raw_sha256=digest,
            timezone=protocol["timezone"],
            eur_to_usd=float(protocol["currency_bridge"]["eur_to_usd"]),
            quarter=raw,
            hourly=hourly,
        ), protocol

    def evaluation_quarters(self, complete_settlement: bool = True) -> pd.DataFrame:
        section = self.protocol_evaluation_bounds(complete_settlement)
        local = self.quarter["timestamp_local"]
        return self.quarter.loc[(local >= section[0]) & (local < section[1])].copy()

    def protocol_evaluation_bounds(self, complete_settlement: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
        # Constants are frozen in the protocol and rechecked by the runner.
        start = pd.Timestamp("2025-10-01").tz_localize(self.timezone)
        end = pd.Timestamp("2026-09-01" if complete_settlement else "2026-10-01").tz_localize(self.timezone)
        return start, end

    def _context_positions(self, issue_utc: pd.Timestamp) -> np.ndarray:
        stop = int(self.hourly.index.searchsorted(issue_utc, side="left"))
        positions = np.arange(stop - 168, stop, dtype=np.int64)
        if positions[0] < 0 or len(positions) != 168:
            raise ValueError("奥地利历史不足168小时")
        expected = pd.date_range(
            end=issue_utc - pd.Timedelta(hours=1), periods=168, freq="1h", tz="UTC"
        )
        if not self.hourly.index[positions].equals(expected):
            raise ValueError("奥地利小时接口存在时间缺口")
        return positions

    def _base_sample(self, context_positions: np.ndarray) -> dict[str, torch.Tensor]:
        frame = self.hourly.iloc[context_positions]
        streams = {
            "price_da": frame["price_da"].to_numpy(np.float32),
            "price_rt": frame["price_rt"].to_numpy(np.float32),
            "load": frame[["load"]].to_numpy(np.float32),
            "system": frame[["wind", "solar"]].to_numpy(np.float32),
            "cal": frame[[
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin",
                "month_cos", "is_weekend", "is_holiday",
            ]].to_numpy(np.float32),
        }
        if any(not np.isfinite(value).all() for value in streams.values()):
            raise ValueError("模型起报历史包含缺失值")
        da_stats = robust_context_stats(streams["price_da"])
        rt_stats = robust_context_stats(streams["price_rt"])
        result: dict[str, torch.Tensor] = {
            "price_da_ctx": torch.from_numpy(normalize(streams["price_da"], da_stats)),
            "price_rt_ctx": torch.from_numpy(normalize(streams["price_rt"], rt_stats)),
            "price_da_mean": torch.tensor(da_stats["mean"], dtype=torch.float32),
            "price_da_std": torch.tensor(da_stats["std"], dtype=torch.float32),
            "price_rt_mean": torch.tensor(rt_stats["mean"], dtype=torch.float32),
            "price_rt_std": torch.tensor(rt_stats["std"], dtype=torch.float32),
        }
        for name in ("load", "system"):
            stats = robust_context_stats(streams[name])
            result[f"{name}_ctx"] = torch.from_numpy(normalize(streams[name], stats))
        result["cal_ctx"] = torch.from_numpy(streams["cal"])
        return result

    def day_sample(self, delivery_date: date) -> dict[str, torch.Tensor]:
        issue_local = pd.Timestamp(delivery_date - timedelta(days=1)).tz_localize(self.timezone) + pd.Timedelta(hours=12)
        issue_utc = issue_local.tz_convert("UTC")
        result = self._base_sample(self._context_positions(issue_utc))
        # Exactly 24 model queries by local clock label.  DST affects the native
        # settlement mapping, never the frozen model shape.
        naive = pd.DatetimeIndex([
            pd.Timestamp(delivery_date) + pd.Timedelta(hours=hour) for hour in range(24)
        ])
        result["cal_tgt"] = torch.from_numpy(calendar_features(naive))
        return result

    def rt_sample(self, issue_utc: pd.Timestamp) -> dict[str, torch.Tensor]:
        issue_utc = pd.Timestamp(issue_utc).tz_convert("UTC")
        result = self._base_sample(self._context_positions(issue_utc))
        target_index = pd.date_range(issue_utc, periods=4, freq="1h", tz="UTC")
        positions = self.hourly.index.get_indexer(target_index)
        if np.any(positions < 0):
            raise ValueError("rolling RT目标DA价格不完整")
        future_da = self.hourly.iloc[positions]["price_da"].to_numpy(np.float32)
        if not np.isfinite(future_da).all():
            raise ValueError("rolling RT未来已公布DA价格缺失")
        da_stats = {
            "mean": float(result["price_da_mean"]),
            "std": float(result["price_da_std"]),
        }
        result["price_da_tgt_known"] = torch.from_numpy(normalize(future_da, da_stats))
        result["cal_tgt"] = torch.from_numpy(calendar_features(target_index.tz_convert(self.timezone)))
        return result


def add_batch_dimension(sample: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0).to(device) for key, value in sample.items()}


def expand_daily_hourly_to_quarters(
    dates: list[date], hourly_predictions: np.ndarray, local_timestamps: pd.DatetimeIndex
) -> np.ndarray:
    if hourly_predictions.shape != (len(dates), 24):
        raise ValueError("日前小时预测形状错误")
    lookup = {
        (delivery_date, hour): float(hourly_predictions[row, hour])
        for row, delivery_date in enumerate(dates) for hour in range(24)
    }
    return np.asarray([lookup[(stamp.date(), stamp.hour)] for stamp in local_timestamps], dtype=np.float64)


def expand_hourly_rt_to_quarter_windows(
    hourly_issues: pd.DatetimeIndex,
    hourly_predictions: np.ndarray,
    quarter_utc: pd.DatetimeIndex,
) -> list[np.ndarray]:
    if hourly_predictions.shape != (len(hourly_issues), 4):
        raise ValueError("RT小时预测形状错误")
    issue_to_row = {stamp: row for row, stamp in enumerate(hourly_issues)}
    result: list[np.ndarray] = []
    for stamp in quarter_utc:
        issue = stamp.floor("1h")
        row = issue_to_row[issue]
        expanded = np.repeat(hourly_predictions[row], 4)
        offset = int((stamp - issue) / pd.Timedelta(minutes=15))
        result.append(expanded[offset:])
    return result


def quarter_hour_backtest(
    da_prediction: np.ndarray,
    rt_at_da_prediction: np.ndarray,
    rt_windows: list[np.ndarray],
    realized_da: np.ndarray,
    realized_rt: np.ndarray,
    local_timestamps: pd.DatetimeIndex,
    battery: dict[str, float],
) -> dict[str, Any]:
    count = len(local_timestamps)
    arrays = [da_prediction, rt_at_da_prediction, realized_da, realized_rt]
    if any(np.asarray(value).shape != (count,) for value in arrays) or len(rt_windows) != count:
        raise ValueError("15分钟回测输入长度不一致")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("15分钟回测包含非有限值")
    simulator = BESSSimulator(
        battery["power_mw"], battery["energy_mwh"], battery["efficiency_component"],
        battery["initial_soc_mwh"] / battery["energy_mwh"], dt=0.25,
        kappa=battery["degradation_eur_per_mwh"],
        soc_min=battery["soc_min_mwh"], soc_max=battery["soc_max_mwh"],
        e_cyc=battery["daily_discharge_cap_mwh"],
    )
    positions: dict[date, list[int]] = defaultdict(list)
    for index, stamp in enumerate(local_timestamps):
        positions[stamp.date()].append(index)
    ordered_dates = sorted(positions)
    plan_soc = float(battery["initial_soc_mwh"])
    plans = np.zeros(count, dtype=np.float64)
    for sample_index, delivery_date in enumerate(ordered_dates):
        idx = np.asarray(positions[delivery_date], dtype=np.int64)
        solved = solve_predicted_da_penalty_lp_day(
            da_prediction[idx], rt_at_da_prediction[idx], simulator,
            initial_plan_soc_mwh=plan_soc, initial_actual_soc_mwh=plan_soc,
            sample_index=sample_index,
        )
        plans[idx] = solved.dispatch.da_discharge_mwh - solved.dispatch.da_charge_mwh
        plan_soc = float(solved.dispatch.da_soc_path_mwh[-1])

    actual_soc = float(battery["initial_soc_mwh"])
    discharged_today = 0.0
    previous_date: date | None = None
    daily: dict[str, dict[str, float]] = {}
    soc_violations = 0
    for position, stamp in enumerate(local_timestamps):
        current_date = stamp.date()
        if current_date != previous_date:
            discharged_today = 0.0
            previous_date = current_date
            daily[str(current_date)] = {
                "da_leg": 0.0, "rt_deviation_leg": 0.0,
                "degradation_cost": 0.0, "deviation_penalty": 0.0,
                "throughput_mwh": 0.0, "net_revenue": 0.0,
            }
        forecast = np.asarray(rt_windows[position], dtype=np.float64)
        horizon = min(len(forecast), count - position)
        future_dates = local_timestamps[position:position + horizon].date
        reset = np.asarray([False] + [future_dates[i] != future_dates[i - 1] for i in range(1, horizon)])
        solved = solve_fixed_da_penalty_lp_window(
            forecast[:horizon], plans[position:position + horizon], simulator,
            initial_soc_mwh=actual_soc,
            initial_discharged_mwh=discharged_today,
            cycle_reset_mask=reset,
            sample_index=position,
        )
        discharge = float(solved.discharge_mwh[0])
        charge = float(solved.charge_mwh[0])
        actual = discharge - charge
        actual_soc += -discharge / battery["efficiency_component"] + charge * battery["efficiency_component"]
        discharged_today += discharge
        if not battery["soc_min_mwh"] - 1e-7 <= actual_soc <= battery["soc_max_mwh"] + 1e-7:
            soc_violations += 1
        deviation = actual - plans[position]
        excess = max(abs(deviation) - battery["deviation_tolerance_fraction"] * abs(plans[position]), 0.0)
        item = daily[str(current_date)]
        item["da_leg"] += plans[position] * realized_da[position]
        item["rt_deviation_leg"] += deviation * realized_rt[position]
        item["degradation_cost"] += battery["degradation_eur_per_mwh"] * (discharge + charge)
        item["deviation_penalty"] += battery["deviation_penalty_abs_rt_multiplier"] * abs(realized_rt[position]) * excess
        item["throughput_mwh"] += discharge + charge
    for item in daily.values():
        item["net_revenue"] = item["da_leg"] + item["rt_deviation_leg"] - item["degradation_cost"] - item["deviation_penalty"]
    net = np.asarray([item["net_revenue"] for item in daily.values()], dtype=np.float64)
    return {
        "days": len(daily),
        "mean_daily_net_revenue_eur": float(net.mean()),
        "median_daily_net_revenue_eur": float(np.median(net)),
        "p10_daily_net_revenue_eur": float(np.quantile(net, 0.10)),
        "positive_day_rate": float(np.mean(net > 0.0)),
        "mean_daily_throughput_mwh": float(np.mean([item["throughput_mwh"] for item in daily.values()])),
        "soc_violation_count": int(soc_violations),
        "final_soc_mwh": float(actual_soc),
        "daily": daily,
    }
