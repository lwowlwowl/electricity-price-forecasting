"""Check future-label exclusion and settlement-year/target alignment."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware import loader_v2
from scripts.decision_aware.prepare_universal_lomo_profit_archives import prepare


def test_parquet_cutoff_excludes_future_local_year_before_preparation(tmp_path, monkeypatch):
    path = tmp_path / "market.parquet"
    # Last 2024 local hour must survive even though its UTC date is Jan 1.
    times = pd.to_datetime(["2025-01-01T05:00:00Z", "2025-01-01T06:00:00Z"])
    pd.DataFrame({"timestamp_utc": times, "node": ["N", "N"], "label": [7, 99999]}).to_parquet(path)
    monkeypatch.setattr(loader_v2, "resolve_market_unified_path", lambda _: str(path))
    monkeypatch.setattr(loader_v2, "_prepare_market_node", lambda raw, *args: raw.copy())
    result = loader_v2.load_market_unified_nodes("ERCOT", ["N"], read_end_utc="2025-01-01T06:00:00Z")["N"]
    assert result["label"].tolist() == [7]
    with pytest.raises(ValueError, match="时区"):
        loader_v2.load_market_unified_nodes("ERCOT", ["N"], read_end_utc="2025-01-01")


@pytest.fixture
def outer_files(tmp_path):
    sources, templates, output = (tmp_path / name for name in ("source", "templates", "output"))
    templates.mkdir()
    for task in ("da", "rt_at_da", "rt"):
        folder = sources / f"universal_lomo_{task}_outer_frozen_v1"
        folder.mkdir(parents=True)
        target = np.ones((1, 2, 24, 4) if task == "rt" else (1, 2, 24), dtype=np.float32)
        common = {"nodes": np.array(["N"]), "dates": np.array(["2024-01-01", "2024-01-02"]), "prediction": target}
        arrays = dict(common, target=target, metadata=json.dumps({"evaluation_stage": "outer_frozen", "retrained": False}))
        if task == "rt_at_da":
            arrays["target_da"] = target
        np.savez(folder / f"ercot_{task}_v1.npz", **arrays)
        arrays = dict(common, target_rt=target)
        if task == "da":
            arrays["target_da"] = target
        if task == "rt":
            arrays.update(executed_timestamps=np.zeros((2, 24)), timezone="America/Chicago")
        np.savez(templates / f"outertemplate_{task}_outer_seed0.npz", **arrays)
    return sources, templates, output


def test_outer_conversion_preserves_frozen_provenance_and_predictions(outer_files):
    sources, templates, output = outer_files
    prepare(sources, templates, output, phase="outer")
    with np.load(output / "universal_da_outer_seed0.npz") as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["phase"] == "outer"
        assert metadata["source_metadata"]["retrained"] is False
        np.testing.assert_array_equal(archive["prediction"], np.ones((1, 2, 24)))


def test_outer_conversion_refuses_shifted_dates(outer_files):
    sources, templates, output = outer_files
    path = sources / "universal_lomo_da_outer_frozen_v1/ercot_da_v1.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["dates"] = np.array(["2023-01-01", "2023-01-02"])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="交付日"):
        prepare(sources, templates, output, phase="outer")


def test_outer_conversion_refuses_corrupted_settlement_truth(outer_files):
    sources, templates, output = outer_files
    path = templates / "outertemplate_da_outer_seed0.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["target_da"] = arrays["target_da"] + 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="真值"):
        prepare(sources, templates, output, phase="outer")
