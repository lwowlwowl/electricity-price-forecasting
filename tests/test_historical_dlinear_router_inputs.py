from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decision_aware.contracts import AnnualStage
from scripts.decision_aware.prepare_historical_baseline_profit_archives import prepare
from scripts.decision_aware.run_allmarket_historical_dlinear import (
    PHASES,
    _read_end_by_market,
    _training_markets,
)


def test_historical_phase_contract_never_reads_2025() -> None:
    assert PHASES["inner"]["training_years"] == [2020, 2021, 2022]
    assert PHASES["inner"]["evaluation_year"] == 2023
    assert PHASES["outer"]["training_years"] == [2020, 2021, 2022, 2023]
    assert PHASES["outer"]["evaluation_year"] == 2024
    outer_end = _read_end_by_market(AnnualStage.OUTER_VALIDATION)
    assert all(value.startswith("2025-01-01T") for value in outer_end.values())


def test_lomo_training_markets_exclude_holdout() -> None:
    assert _training_markets(None) == ["CAISO", "ERCOT", "NYISO", "PJM"]
    assert _training_markets("ERCOT") == ["CAISO", "NYISO", "PJM"]


def test_prepare_historical_profit_archives(tmp_path: Path) -> None:
    nodes = np.asarray(["N1"])
    dates = np.asarray(["2023-01-01"])
    da = np.arange(24, dtype=np.float32).reshape(1, 1, 24)
    rt = da + 1.0
    rt_windows = np.repeat(rt[..., None], 4, axis=-1)
    timestamps = np.asarray(
        [[f"2023-01-01T{hour:02d}:00:00+00:00" for hour in range(24)]]
    )
    paths = {
        "da": tmp_path / "da.npz",
        "rt_at_da": tmp_path / "rt_at_da.npz",
        "rt": tmp_path / "rt.npz",
    }
    np.savez_compressed(paths["da"], nodes=nodes, dates=dates, prediction=da, target=da)
    np.savez_compressed(
        paths["rt_at_da"],
        nodes=nodes,
        dates=dates,
        prediction=rt,
        target=rt,
        target_da=da,
    )
    np.savez_compressed(
        paths["rt"],
        nodes=nodes,
        dates=dates,
        prediction=rt_windows,
        target=rt_windows,
        executed_timestamps=timestamps,
        timezone=np.asarray("UTC"),
    )
    output = tmp_path / "profit"
    prepare("dlinear", "inner", paths, output)
    with np.load(output / "dlinear_da_inner_seed0.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert np.array_equal(archive["target_da"], da)
        assert np.array_equal(archive["target_rt"], rt)
        assert metadata["selection_allowed"] is True
        assert metadata["locked_2025_read"] is False
    with np.load(output / "dlinear_rt_inner_seed0.npz", allow_pickle=False) as archive:
        assert archive["prediction"].shape == (1, 1, 24, 4)
        assert str(archive["timezone"]) == "UTC"
