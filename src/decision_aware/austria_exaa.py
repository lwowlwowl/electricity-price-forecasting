"""Causal EXAA feature utilities for the Austria external evaluation.

EXAA publishes an earlier Austrian day-ahead auction.  Its hourly price is
known before the later day-ahead forecast gate, so it is a legitimate
``past_future`` covariate rather than a label leak.  This module only handles
parsing and time alignment; network access lives in the download script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VIENNA_TIMEZONE = "Europe/Vienna"
EXAA_COLUMN = "exaa_eur_per_mwh"


def invert_flexspot_gross_cents(gross_cents_per_kwh: np.ndarray) -> np.ndarray:
    """Recover the raw EXAA price from Salzburg AG's displayed gross tariff.

    The public tariff formula is

    ``gross = 1.2 * (raw + 0.03 * abs(raw) + 1.8)``.

    Values at the 76.32 cent/kWh retail cap cannot be inverted uniquely and
    are rejected rather than silently clipped.
    """
    gross = np.asarray(gross_cents_per_kwh, dtype=np.float64)
    if not np.isfinite(gross).all():
        raise ValueError("FlexSpot prices contain non-finite values")
    if np.any(gross >= 76.32 - 1e-9):
        raise ValueError("FlexSpot cap encountered; raw EXAA price is not identifiable")
    zero_raw_gross = 1.2 * 1.8
    raw = np.where(
        gross >= zero_raw_gross,
        (gross / 1.2 - 1.8) / 1.03,
        (gross / 1.2 - 1.8) / 0.97,
    )
    return raw


def _local_hourly_to_utc(local_naive: pd.Series) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(local_naive, dayfirst=True))
    # The source contains 23 rows on the spring transition and two ordered
    # 02:00 rows on the autumn transition.  ``ambiguous='infer'`` therefore
    # maps the two fall-back hours to distinct UTC instants without deletion.
    localized = parsed.tz_localize(VIENNA_TIMEZONE, ambiguous="infer")
    utc = localized.tz_convert("UTC")
    if not utc.is_unique or not utc.is_monotonic_increasing:
        raise ValueError("EXAA timestamps do not form a unique increasing UTC axis")
    if not (utc.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("EXAA timestamps are not continuous hourly observations")
    return utc


def read_historical_exaa_csv(path: str | Path) -> pd.DataFrame:
    """Read Salzburg AG's raw historical EXAA CSV into a UTC hourly frame."""
    source = pd.read_csv(path, sep=";", decimal=",").dropna(how="all")
    required = {"Datum und Stunde", "EXAA AT [Cent/kWh]"}
    if not required.issubset(source.columns):
        raise ValueError(f"historical EXAA CSV missing columns: {sorted(required)}")
    utc = _local_hourly_to_utc(source["Datum und Stunde"])
    raw_cents = pd.to_numeric(source["EXAA AT [Cent/kWh]"], errors="raise")
    frame = pd.DataFrame(
        {
            EXAA_COLUMN: raw_cents.to_numpy(np.float64) * 10.0,
            "source": "salzburg_ag_historical_csv",
        },
        index=utc,
    )
    frame.index.name = "timestamp_utc"
    return frame


def flexspot_records_to_hourly(records: list[dict[str, object]]) -> pd.DataFrame:
    """Convert public FlexSpot API records into raw EXAA UTC hourly prices."""
    if not records:
        raise ValueError("FlexSpot API returned no records")
    source = pd.DataFrame.from_records(records)
    required = {"date", "stunde", "preis"}
    if not required.issubset(source.columns):
        raise ValueError(f"FlexSpot records missing fields: {sorted(required)}")
    local_text = source["date"].astype(str) + " " + source["stunde"].astype(str)
    local = pd.to_datetime(local_text, format="%Y-%m-%d %H:%M")
    utc = _local_hourly_to_utc(pd.Series(local.dt.strftime("%d.%m.%Y %H:%M")))
    gross = pd.to_numeric(
        source["preis"].astype(str).str.replace(",", ".", regex=False),
        errors="raise",
    ).to_numpy(np.float64)
    raw_cents = invert_flexspot_gross_cents(gross)
    frame = pd.DataFrame(
        {
            EXAA_COLUMN: raw_cents * 10.0,
            "source": "salzburg_ag_public_flexspot_api_inverted",
        },
        index=utc,
    )
    frame.index.name = "timestamp_utc"
    return frame


def merge_exaa_hourly(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge non-overlapping hourly pieces and validate a continuous UTC axis."""
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        duplicate = combined.index[combined.index.duplicated()].unique()[0]
        raise ValueError(f"duplicate EXAA timestamp: {duplicate}")
    if not np.isfinite(combined[EXAA_COLUMN].to_numpy(np.float64)).all():
        raise ValueError("EXAA series contains non-finite prices")
    if not (
        combined.index.to_series().diff().dropna() == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("merged EXAA series is not continuous hourly data")
    return combined


def exaa_quarter_hour_covariate(
    hourly: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Repeat each causal hourly EXAA price over its four 15-minute intervals."""
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    if not target.is_unique or not target.is_monotonic_increasing:
        raise ValueError("target index must be unique and increasing")
    expanded = hourly[EXAA_COLUMN].reindex(
        target,
        method="ffill",
        tolerance=pd.Timedelta(minutes=59),
    )
    if expanded.isna().any():
        missing = expanded.index[expanded.isna()][0]
        raise ValueError(f"EXAA does not cover target timestamp {missing}")
    expanded.name = EXAA_COLUMN
    return expanded
