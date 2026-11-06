from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_chronos2_fat_proxy import (  # noqa: E402
    KNOWN_FUTURE_COLUMNS,
    PAST_ONLY_COLUMNS,
    build_chronos2_fat_proxy,
)
from decision_aware.austria_external import PRICE_DA, TIMESTAMP  # noqa: E402


def test_chronos_proxy_contains_only_teacher_feature_families() -> None:
    periods = 900
    values = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            TIMESTAMP: pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC"),
            PRICE_DA: values,
            "实际负荷 (MW)": values + 1000.0,
            "风电实际 (MW)": values + 100.0,
            "光伏实际 (MW)": values + 10.0,
            "实际气温 (°C)": values + 2.0,
            "实际10米风速 (m/s)": values + 1.0,
            "实际短波辐射 (W/m²)": values + 3.0,
            "是否节假日": np.zeros(periods),
        }
    )
    result = build_chronos2_fat_proxy(raw)
    assert tuple(result.columns[1:]) == (*KNOWN_FUTURE_COLUMNS, *PAST_ONLY_COLUMNS)
    assert result.index[0] == raw.iloc[672][TIMESTAMP]
    assert result.iloc[0]["at_solar_da_proxy"] == raw.iloc[0]["光伏实际 (MW)"]
    assert result.iloc[0]["gen_solar_lag24"] == raw.iloc[576]["光伏实际 (MW)"]

