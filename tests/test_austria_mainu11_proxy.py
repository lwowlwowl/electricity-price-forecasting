from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import PRICE_DA, TIMESTAMP  # noqa: E402
from decision_aware.austria_mainu11_proxy import (  # noqa: E402
    PROXY_COLUMNS,
    PROXY_COLUMNS_WITH_EXAA,
    UNAVAILABLE_MAINU11_COLUMNS,
    build_mainu11_proxy,
    build_mainu11_proxy_covariates,
    build_mainu11_proxy_with_exaa,
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


def test_future_covariate_extension_only_reads_week_lag() -> None:
    periods = 9 * 24 * 4
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    raw = pd.DataFrame({
        TIMESTAMP: timestamp,
        PRICE_DA: np.arange(periods, dtype=float),
        "实际负荷 (MW)": 1000.0 + np.arange(periods, dtype=float),
        "风电实际 (MW)": 100.0 + np.arange(periods, dtype=float) * 0.1,
        "光伏实际 (MW)": 50.0 + np.arange(periods, dtype=float) * 0.05,
        "实际10米风速 (m/s)": 5.0 + np.arange(periods, dtype=float) * 0.001,
        "实际短波辐射 (W/m²)": 200.0 + np.arange(periods, dtype=float) * 0.01,
    })
    covariates = build_mainu11_proxy_covariates(raw, future_quarters=8)
    assert covariates.index[-1] == timestamp[-1] + pd.Timedelta(hours=2)
    source_time = covariates.index[-1] - pd.Timedelta(days=7)
    source_row = raw.loc[raw[TIMESTAMP] == source_time].iloc[0]
    assert covariates.iloc[-1]["at_gen_da_a71_proxy"] == source_row["实际负荷 (MW)"]
    assert covariates.iloc[-1]["at_solar_da_proxy"] == source_row["光伏实际 (MW)"]


def test_exaa_is_the_only_added_mainu11_covariate() -> None:
    periods = 10 * 24 * 4
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    raw = pd.DataFrame({
        TIMESTAMP: timestamp,
        PRICE_DA: np.arange(periods, dtype=float),
        "实际负荷 (MW)": 1000.0 + np.arange(periods, dtype=float),
        "风电实际 (MW)": 100.0 + np.arange(periods, dtype=float),
        "光伏实际 (MW)": 10.0 + np.arange(periods, dtype=float),
        "实际10米风速 (m/s)": 5.0 + np.arange(periods, dtype=float),
        "实际短波辐射 (W/m²)": 200.0 + np.arange(periods, dtype=float),
    })
    hourly_index = pd.date_range(
        timestamp[0].floor("h"), timestamp[-1].ceil("h"), freq="1h", tz="UTC"
    )
    exaa = pd.DataFrame(
        {"exaa_eur_per_mwh": np.arange(len(hourly_index)), "source": "test"},
        index=hourly_index,
    )
    result = build_mainu11_proxy_with_exaa(raw, exaa)
    assert tuple(result.columns[1:]) == PROXY_COLUMNS_WITH_EXAA
    assert len(PROXY_COLUMNS_WITH_EXAA) == len(PROXY_COLUMNS) + 1
    assert result["exaa_eur_per_mwh"].notna().all()

    perturbed = raw.copy()
    perturbed[PRICE_DA] = perturbed[PRICE_DA] + 10_000.0
    perturbed_result = build_mainu11_proxy_with_exaa(perturbed, exaa)
    pd.testing.assert_frame_equal(
        result[list(PROXY_COLUMNS_WITH_EXAA)],
        perturbed_result[list(PROXY_COLUMNS_WITH_EXAA)],
    )
