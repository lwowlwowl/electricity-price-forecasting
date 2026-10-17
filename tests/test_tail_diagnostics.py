import numpy as np
import pytest
from pathlib import Path
import sys
import json


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.tail_diagnostics import (
    fit_tail_thresholds,
    price_slice_report,
    settlement_slice_report,
    spread_slice_report,
)
from scripts.decision_aware import diagnose_tail_and_profit_instability as diagnostic


def test_thresholds_are_fit_only_from_supplied_training_values():
    da = np.arange(100.0)
    rt = da[::-1]
    result = fit_tail_thresholds(da, rt)
    assert result.da_lower_1pct == pytest.approx(0.99)
    assert result.da_upper_1pct == pytest.approx(98.01)
    assert result.rt_lower_1pct == pytest.approx(0.99)
    assert result.abs_spread_upper_1pct > result.abs_spread_upper_5pct


def test_price_and_spread_reports_expose_tail_metrics():
    thresholds = fit_tail_thresholds(
        [-10.0, 0.0, 10.0, 100.0], [-20.0, 5.0, 15.0, 80.0]
    )
    price = price_slice_report(
        [-5.0, 1.0, 8.0, 100.0],
        [-10.0, 0.0, 10.0, 100.0],
        lower_1pct=thresholds.da_lower_1pct,
        upper_1pct=thresholds.da_upper_1pct,
    )
    assert price["all"]["samples"] == 4
    assert price["negative_price"]["negative_recall"] == pytest.approx(1.0)
    assert price["all"]["mae"] == pytest.approx(2.0)

    spread = spread_slice_report(
        [10.0, -10.0], [0.0, 0.0],
        [5.0, 5.0], [0.0, 10.0], thresholds,
    )
    assert spread["all"]["direction_accuracy"] == pytest.approx(1.0)
    assert spread["all"]["magnitude_ratio"] == pytest.approx(2.0)


def test_settlement_slices_sum_the_complete_hourly_ledger():
    thresholds = fit_tail_thresholds(
        [-20.0, 0.0, 20.0, 100.0], [-10.0, 5.0, 10.0, 80.0]
    )
    hourly = {
        "realized_da_price": np.asarray([-20.0, 100.0]),
        "realized_rt_price": np.asarray([-10.0, 80.0]),
        "da_leg": np.asarray([2.0, 8.0]),
        "rt_deviation_leg": np.asarray([1.0, 3.0]),
        "degradation_cost": np.asarray([0.5, 0.5]),
        "deviation_penalty": np.asarray([0.25, 0.75]),
        "net_revenue": np.asarray([2.25, 9.75]),
    }
    report = settlement_slice_report(hourly, thresholds)
    assert report["all"]["hours"] == 2
    assert report["all"]["total_net_revenue"] == pytest.approx(12.0)
    assert report["da_negative_price"]["hours"] == 1
    assert report["da_negative_price"]["total_net_revenue"] == pytest.approx(2.25)


def test_pooled_xgboost_loader_enforces_zero_shot_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    root = tmp_path / "data/results/universal_lomo_xgboost_v1"
    root.mkdir(parents=True)
    for task in diagnostic.TASKS:
        metadata = {
            "protocol": "pooled_lomo_xgboost_equal_training_years",
            "uses_target_market_training_labels": False,
            "uses_node_id_for_prediction": False,
            "uses_market_id_for_prediction": False,
        }
        arrays = {
            "nodes": np.asarray(["N"]),
            "dates": np.asarray(["2023-01-01"]),
            "prediction": np.zeros((1, 1, 24), dtype=np.float32),
            "target": np.zeros((1, 1, 24), dtype=np.float32),
            "metadata": json.dumps(metadata),
        }
        if task == "rt_at_da":
            arrays["target_da"] = np.zeros((1, 1, 24), dtype=np.float32)
        np.savez_compressed(
            root / f"holdout_ercot_{task}_inner_seed0.npz", **arrays
        )
    result = diagnostic._pooled_xgboost_triplet("ERCOT")
    assert set(result) == set(diagnostic.TASKS)
    assert result["da"]["target_da"].shape == (1, 1, 24)


@pytest.mark.parametrize(
    ("system", "directory", "name_template", "protocol"),
    (
        (
            "timesfm3",
            "universal_lomo_timesfm3_v1",
            "ercot_{task}_v1.npz",
            "frozen_foundation_model_same_information_contract",
        ),
        (
            "timesfm25_lora",
            "universal_lomo_timesfm25_lora_v1",
            "holdout_ercot_{task}_inner_seed0.npz",
            "timesfm25_source_market_supervised_lora_target_market_zero_shot",
        ),
    ),
)
def test_tsfm_loader_enforces_protocol_and_zero_shot_contract(
    tmp_path, monkeypatch, system, directory, name_template, protocol
):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    root = tmp_path / "data/results" / directory
    root.mkdir(parents=True)
    for task in diagnostic.TASKS:
        metadata = {
            "protocol": protocol,
            "test_2025": False,
            "uses_node_id_for_prediction": False,
            "uses_market_id_for_prediction": False,
            "uses_holdout_labels_for_training": False,
            "selection_uses_holdout": False,
        }
        arrays = {
            "nodes": np.asarray(["N"]),
            "dates": np.asarray(["2023-01-01"]),
            "prediction": np.zeros((1, 1, 24), dtype=np.float32),
            "target": np.zeros((1, 1, 24), dtype=np.float32),
            "metadata": json.dumps(metadata),
        }
        if task == "rt_at_da":
            arrays["target_da"] = np.zeros((1, 1, 24), dtype=np.float32)
        np.savez_compressed(root / name_template.format(task=task), **arrays)
    result = diagnostic._tsfm_triplet("ERCOT", system)
    assert set(result) == set(diagnostic.TASKS)
    assert result["da"]["prediction"].shape == (1, 1, 24)


def test_attach_settlement_contract_requires_same_samples_and_adds_fields():
    nodes = np.asarray(["N"])
    dates = np.asarray(["2023-01-01"])
    candidate = {
        "da": {
            "nodes": nodes, "dates": dates,
            "prediction": np.ones((1, 1, 24), dtype=np.float32),
            "target_da": np.zeros((1, 1, 24), dtype=np.float32),
        },
        "rt_at_da": {
            "nodes": nodes, "dates": dates,
            "prediction": np.ones((1, 1, 24), dtype=np.float32),
            "target_rt": np.full((1, 1, 24), 2.0, dtype=np.float32),
        },
        "rt": {
            "nodes": nodes, "dates": dates,
            "prediction": np.ones((1, 1, 24, 4), dtype=np.float32),
            "target_rt": np.full((1, 1, 24, 4), 2.0, dtype=np.float32),
        },
    }
    timestamps = np.asarray([[f"2023-01-01T{hour:02d}:00:00Z" for hour in range(24)]])
    reference = {
        "da": {
            **candidate["da"],
            "target_rt": np.full((1, 1, 24), 2.0, dtype=np.float32),
        },
        "rt_at_da": dict(candidate["rt_at_da"]),
        "rt": {
            **candidate["rt"],
            "executed_timestamps": timestamps,
            "timezone": "UTC",
        },
    }
    completed = diagnostic._attach_settlement_contract(candidate, reference)
    assert np.array_equal(completed["da"]["target_rt"], reference["da"]["target_rt"])
    assert np.array_equal(completed["rt"]["executed_timestamps"], timestamps)
    assert completed["rt"]["timezone"] == "UTC"
    # 只补结算合同，不改变候选预测。
    assert np.array_equal(completed["da"]["prediction"], candidate["da"]["prediction"])

    mismatched = {task: dict(values) for task, values in candidate.items()}
    mismatched["da"]["dates"] = np.asarray(["2023-01-02"])
    with pytest.raises(ValueError, match="交付日不一致"):
        diagnostic._attach_settlement_contract(mismatched, reference)


def test_profit_aggregate_keeps_paired_daily_and_risk_fields():
    thresholds = fit_tail_thresholds(
        [-10.0, 0.0, 10.0, 20.0], [-5.0, 2.0, 8.0, 15.0]
    )

    def node(first, second):
        net = np.asarray([first, second], dtype=np.float64)
        return {
            "days": 2,
            "daily": {
                "2023-01-01": {"net_revenue": float(first)},
                "2023-01-02": {"net_revenue": float(second)},
            },
            "revenue_components": {
                "da_leg": float(net.sum()),
                "rt_deviation_leg": 0.0,
                "degradation_cost": 0.0,
                "deviation_penalty": 0.0,
            },
            "hourly": {
                "realized_da_price": np.asarray([10.0, 20.0]),
                "realized_rt_price": np.asarray([5.0, 15.0]),
                "predicted_da_price": np.asarray([9.0, 19.0]),
                "predicted_rt_at_da_price": np.asarray([4.0, 14.0]),
                "da_plan_mwh": np.asarray([1.0, 1.0]),
                "da_leg": net,
                "rt_deviation_leg": np.zeros(2),
                "degradation_cost": np.zeros(2),
                "deviation_penalty": np.zeros(2),
                "net_revenue": net,
            },
        }

    result = diagnostic._aggregate_profit(
        {"A": node(10.0, -5.0), "B": node(6.0, -1.0)}, thresholds
    )
    assert result["macro_daily_net_revenue"] == {
        "2023-01-01": 8.0, "2023-01-02": -3.0
    }
    assert result["macro_positive_day_rate"] == pytest.approx(0.5)
    assert result["macro_max_drawdown_usd_per_node"] == pytest.approx(3.0)
    assert result["worst_node_mean_daily_net_revenue"] == pytest.approx(2.5)
