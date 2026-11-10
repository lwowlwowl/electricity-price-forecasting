from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_exaa import (
    EXAA_COLUMN,
    exaa_quarter_hour_covariate,
    flexspot_records_to_hourly,
    invert_flexspot_gross_cents,
)


def test_flexspot_inverse_recovers_positive_negative_and_zero_raw_prices() -> None:
    raw = np.asarray([-10.0, 0.0, 12.5])
    gross = 1.2 * (raw + 0.03 * np.abs(raw) + 1.8)
    np.testing.assert_allclose(invert_flexspot_gross_cents(gross), raw, atol=1e-12)


def test_flexspot_inverse_rejects_non_identifiable_retail_cap() -> None:
    with pytest.raises(ValueError, match="not identifiable"):
        invert_flexspot_gross_cents(np.asarray([76.32]))


def test_api_records_preserve_spring_dst_and_expand_to_quarters() -> None:
    hours = [f"{hour:02d}:00" for hour in range(24) if hour != 2]
    records = [
        {"date": "2026-03-29", "stunde": hour, "preis": "12,5400"}
        for hour in hours
    ]
    hourly = flexspot_records_to_hourly(records)
    assert len(hourly) == 23
    assert hourly.index[0] == pd.Timestamp("2026-03-28 23:00:00", tz="UTC")
    assert hourly.index[-1] == pd.Timestamp("2026-03-29 21:00:00", tz="UTC")

    target = pd.date_range(hourly.index[0], periods=4, freq="15min", tz="UTC")
    expanded = exaa_quarter_hour_covariate(hourly, target)
    assert len(expanded) == 4
    assert expanded.nunique() == 1
    assert expanded.name == EXAA_COLUMN
