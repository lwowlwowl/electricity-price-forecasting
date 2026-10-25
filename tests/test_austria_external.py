from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    expand_daily_hourly_to_quarters,
)

PROTOCOL = ROOT / "configs/decision_aware/austria_external_15min_v1.json"


def test_protocol_is_strict_external_and_raw_hash_matches():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    strict = protocol["strict_external_test"]
    assert not any(value for key, value in strict.items() if key.endswith("_allowed"))
    raw = ROOT / protocol["raw_parquet"]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == protocol["raw_sha256"]


def test_native_quarter_timeline_and_dst_days_are_preserved():
    data, _ = AustriaExternalData.load(ROOT, PROTOCOL)
    quarter = data.evaluation_quarters(complete_settlement=True)
    delta = np.diff(quarter.index.asi8)
    assert np.all(delta == pd.Timedelta(minutes=15).value)
    counts = pd.Series(1, index=quarter["timestamp_local"]).groupby(
        quarter["timestamp_local"].dt.date
    ).sum()
    assert counts.loc[pd.Timestamp("2025-10-26").date()] == 100
    assert counts.loc[pd.Timestamp("2026-03-29").date()] == 92
    assert set(counts.unique()) == {92, 96, 100}


def test_hourly_adapter_keeps_native_dst_length():
    data, _ = AustriaExternalData.load(ROOT, PROTOCOL)
    quarter = data.evaluation_quarters(complete_settlement=True)
    local = pd.DatetimeIndex(quarter["timestamp_local"])
    dates = sorted(set(local.date))
    hourly = np.tile(np.arange(24, dtype=np.float64), (len(dates), 1))
    expanded = expand_daily_hourly_to_quarters(dates, hourly, local)
    assert expanded.shape == (len(quarter),)
    autumn = expanded[local.date == pd.Timestamp("2025-10-26").date()]
    spring = expanded[local.date == pd.Timestamp("2026-03-29").date()]
    assert len(autumn) == 100 and np.sum(autumn == 2.0) == 8
    assert len(spring) == 92 and np.sum(spring == 2.0) == 0


def test_samples_only_use_causal_history_and_fixed_shapes():
    data, _ = AustriaExternalData.load(ROOT, PROTOCOL)
    day = data.day_sample(pd.Timestamp("2025-10-01").date())
    rt = data.rt_sample(pd.Timestamp("2025-10-01T00:00:00Z"))
    assert day["price_da_ctx"].shape == (168,)
    assert day["cal_tgt"].shape == (24, 8)
    assert rt["price_rt_ctx"].shape == (168,)
    assert rt["price_da_tgt_known"].shape == (4,)
    assert rt["cal_tgt"].shape == (4, 8)
    assert all(np.isfinite(value.numpy()).all() for value in (*day.values(), *rt.values()))
