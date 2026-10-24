import numpy as np
import pytest

from scripts.decision_aware.build_crossmarket_regret_router_v1 import (
    _copy_family,
    _economic_advantage,
    _mae_advantage,
    build_target,
)
from scripts.decision_aware.prepare_crossmarket_regret_router_v1 import _timestamps


def _task(prediction, *, rolling=False):
    days = prediction.shape[1]
    result = {
        "nodes": np.asarray(["N1"]),
        "dates": np.asarray([f"2023-01-{index + 1:02d}" for index in range(days)]),
        "prediction": prediction.astype(np.float32),
        "target_rt": np.zeros_like(prediction, dtype=np.float32),
        "metadata": np.asarray("{}"),
    }
    if rolling:
        result["executed_timestamps"] = np.full((days, 24), "2023-01-01T00:00:00+00:00")
        result["timezone"] = np.asarray("UTC")
    else:
        result["target_da"] = np.zeros_like(prediction, dtype=np.float32)
    return result


def test_timestamp_synthesis_is_24_hour_and_dst_safe():
    timestamps = _timestamps(np.asarray(["2023-01-02"]), "ERCOT")
    assert timestamps.shape == (1, 24)
    assert timestamps[0, 0].endswith("-06:00")
    with pytest.raises(ValueError, match="不是24小时交付日"):
        _timestamps(np.asarray(["2023-03-12"]), "ERCOT")


def test_route_switches_both_day_ahead_tasks_but_never_rolling_rt(tmp_path):
    transformer_da = np.zeros((1, 2, 24), dtype=np.float32)
    dlinear_da = np.full((1, 2, 24), 7.0, dtype=np.float32)
    transformer_rt = np.zeros((1, 2, 24, 4), dtype=np.float32)
    dlinear_rt = np.full((1, 2, 24, 4), 9.0, dtype=np.float32)
    record = {
        "transformer": {
            "da": _task(transformer_da),
            "rt_at_da": _task(transformer_da),
            "rt": _task(transformer_rt, rolling=True),
        },
        "dlinear": {
            "da": _task(dlinear_da),
            "rt_at_da": _task(dlinear_da),
            "rt": _task(dlinear_rt, rolling=True),
        },
    }
    mask = np.asarray([[False, True]])
    _copy_family(record, tmp_path, "route", mask, {"protocol": "test"})
    with np.load(tmp_path / "route_da_inner_seed0.npz", allow_pickle=False) as archive:
        assert np.all(archive["prediction"][0, 0] == 0.0)
        assert np.all(archive["prediction"][0, 1] == 7.0)
    with np.load(tmp_path / "route_rt_inner_seed0.npz", allow_pickle=False) as archive:
        assert np.all(archive["prediction"] == 0.0)


def test_regret_labels_have_the_documented_dlinear_minus_transformer_semantics():
    zero = np.zeros((1, 2, 24), dtype=np.float32)
    record = {
        "features": np.zeros((1, 2, 3), dtype=np.float64),
        "dates": np.asarray(["2023-01-01", "2023-01-02"]),
        "nodes": np.asarray(["N1"]),
        "transformer": {
            "da": {"prediction": zero + 2.0, "target_da": zero},
            "rt_at_da": {"prediction": zero + 2.0, "target_rt": zero},
        },
        "dlinear": {
            "da": {"prediction": zero + 1.0, "target_da": zero},
            "rt_at_da": {"prediction": zero + 1.0, "target_rt": zero},
        },
    }
    settlement = {
        "systems": {
            "da_transformer__rt_at_da_transformer__rt_transformer": {
                "nodes": {"N1": {"daily_net_revenue": {"2023-01-01": 3.0, "2023-01-02": 8.0}}}
            },
            "da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid": {
                "nodes": {"N1": {"daily_net_revenue": {"2023-01-01": 5.0, "2023-01-02": 7.0}}}
            },
        }
    }
    assert np.allclose(_economic_advantage(record, settlement), [[2.0, -1.0]])
    assert np.allclose(_mae_advantage(record), [[1.0, 1.0]])


def test_history_assisted_manifest_distinguishes_history_from_heldout_labels(tmp_path, monkeypatch):
    import scripts.decision_aware.build_crossmarket_regret_router_v1 as module

    class FakeTree:
        def __init__(self, feature_count):
            self.n_features_in_ = feature_count

    records = {}
    settlements = {}
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        task = _task(np.zeros((1, 2, 24), dtype=np.float32))
        records[market] = {
            "transformer": {"da": task, "rt_at_da": task, "rt": _task(np.zeros((1, 2, 24, 4)), rolling=True)},
            "dlinear": {"da": task, "rt_at_da": task, "rt": _task(np.zeros((1, 2, 24, 4)), rolling=True)},
            "features": np.zeros((1, 2, 3), dtype=np.float64),
            "names": ("spread_q10", "disagreement", "month_sin"),
            "dates": task["dates"],
            "nodes": task["nodes"],
        }
        settlements[market] = {
            "systems": {
                "da_transformer__rt_at_da_transformer__rt_transformer": {
                    "nodes": {"N1": {"daily_net_revenue": {"2023-01-01": 0.0, "2023-01-02": 0.0}}}
                },
                "da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid": {
                    "nodes": {"N1": {"daily_net_revenue": {"2023-01-01": 0.0, "2023-01-02": 0.0}}}
                },
            }
        }
    monkeypatch.setattr(module, "_feature_record", lambda root, market: records[market])
    monkeypatch.setattr(module, "fit_regret_tree", lambda x, y, sample_weight=None: FakeTree(x.shape[1]))
    monkeypatch.setattr(module, "predict_regret", lambda model, x: np.zeros(len(x)))
    monkeypatch.setattr(module, "regret_tree_explanation", lambda model, names: {"feature_names": list(names)})
    for market, settlement in settlements.items():
        (tmp_path / f"{market.lower()}.json").write_text(
            __import__("json").dumps(settlement), encoding="utf-8"
        )
    manifest = build_target(tmp_path, tmp_path, tmp_path / "out", "CAISO")

    assert manifest["protocol"] == "crossmarket_history_assisted_economic_regret_router_v1"
    assert manifest["dlinear_training_markets"] == ["CAISO", "ERCOT", "NYISO", "PJM"]
    assert manifest["transformer_training_markets"] == ["ERCOT", "NYISO", "PJM"]
    assert manifest["router_training_markets"] == ["ERCOT", "NYISO", "PJM"]
    assert manifest["target_market_2020_2022_history_used_for_dlinear"] is True
    assert manifest["target_market_2023_labels_used_for_router_training"] is False
