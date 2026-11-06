from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import PRICE_DA, TIMESTAMP  # noqa: E402
from decision_aware.austria_mainu11_proxy import (  # noqa: E402
    PROXY_COLUMNS,
    UNAVAILABLE_MAINU11_COLUMNS,
    build_mainu11_proxy,
)


def test_proxy_uses_only_teacher_feature_families_and_exact_week_lag() -> None:
    periods = 700
    raw = pd.DataFrame({
        TIMESTAMP: pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC"),
        PRICE_DA: np.arange(periods, dtype=float),
        "实际负荷 (MW)": np.arange(periods, dtype=float) + 1000.0,
        "风电实际 (MW)": np.arange(periods, dtype=float) + 100.0,
        "光伏实际 (MW)": np.arange(periods, dtype=float) + 10.0,
        "实际10米风速 (m/s)": np.arange(periods, dtype=float) + 1.0,
        "实际短波辐射 (W/m²)": np.arange(periods, dtype=float) + 2.0,
    })
    result = build_mainu11_proxy(raw)
    assert tuple(result.columns[1:]) == PROXY_COLUMNS
    assert len(PROXY_COLUMNS) == 6
    assert len(UNAVAILABLE_MAINU11_COLUMNS) == 5
    assert result.iloc[0]["at_solar_da_proxy"] == raw.iloc[0]["光伏实际 (MW)"]
    assert result.index[0] == raw.iloc[672][TIMESTAMP]
