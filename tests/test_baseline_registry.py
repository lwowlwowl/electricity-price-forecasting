import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baseline_registry import (  # noqa: E402
    get_baseline_stage,
    load_baseline_registry,
    verify_registered_sources,
)


def _read(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_registered_source_hashes_are_frozen_and_present():
    assert verify_registered_sources() == []


def test_2023_fair_baseline_matches_frozen_result():
    stage = get_baseline_stage("development_2023")
    source = _read(stage["strongest_fair_baseline"]["source"])
    rows = {row["system"]: row for row in source["profit_rows"]}
    assert stage["strongest_fair_baseline"]["value"] == rows[
        "xgboost_pooled"
    ]["ercot_2023_mean_daily_net_revenue"]
    assert stage["frozen_transformer_incumbent"]["value"] == rows[
        "transformer_ensemble"
    ]["ercot_2023_mean_daily_net_revenue"]


def test_2023_complete_comparators_match_frozen_results():
    stage = get_baseline_stage("development_2023")
    comparators = stage["complete_three_leg_comparators"]
    fair_rows = {
        row["system"]: row
        for row in _read("data/results/fair_xgboost_comparison_v1/summary.json")[
            "profit_rows"
        ]
    }
    assert comparators["no_id_transformer_seed0"]["value"] == fair_rows[
        "transformer"
    ]["ercot_2023_mean_daily_net_revenue"]
    assert comparators["per_node_local_xgboost"]["value"] == fair_rows[
        "xgboost_local"
    ]["ercot_2023_mean_daily_net_revenue"]

    source_system_keys = {
        "timesfm_2_5_lora": "da_timesfm25_lora__rt_at_da_timesfm25_lora__rt_timesfm25_lora",
        "lag_168_seasonal": "da_seasonal__rt_at_da_seasonal__rt_seasonal",
        "timesfm_3_frozen": "da_timesfm3_frozen__rt_at_da_timesfm3_frozen__rt_timesfm3_frozen",
    }
    for name, system_key in source_system_keys.items():
        comparator = comparators[name]
        source = _read(comparator["source"])
        assert comparator["value"] == source["systems"][system_key][
            "macro_mean_daily_revenue"
        ]


def test_2024_comparator_matches_frozen_result():
    stage = get_baseline_stage("frozen_confirmation_2024")
    source = _read(stage["strongest_available_comparator"]["source"])
    assert stage["strongest_available_comparator"]["value"] == source[
        "xgboost_usd_per_node_day"
    ]
    assert stage["frozen_transformer_incumbent"]["value"] == source[
        "universal_usd_per_node_day"
    ]


def test_2025_baselines_match_locked_results():
    registry = load_baseline_registry()
    stage = registry["stages"]["locked_report_only_2025"]
    locked = _read(stage["strongest_preregistered_pure_baseline"]["source"])
    dlinear = locked["systems"][
        "da_dlinear__rt_at_da_dlinear__rt_dlinear"
    ]["macro_mean_daily_revenue"]
    ensemble = locked["systems"][
        "da_ensemble__rt_at_da_ensemble__rt_ensemble"
    ]["macro_mean_daily_revenue"]
    router = _read(stage["strongest_existing_posttest_method"]["source"])[
        "router_macro_mean_daily_revenue"
    ]
    assert stage["strongest_preregistered_pure_baseline"]["value"] == dlinear
    assert stage["frozen_transformer_incumbent"]["value"] == ensemble
    assert stage["strongest_existing_posttest_method"]["value"] == router

    system_keys = {
        "dlinear": "da_dlinear__rt_at_da_dlinear__rt_dlinear",
        "per_node_xgboost": "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost",
        "pooled_xgboost": "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost",
        "timesfm_2_5_lora": "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25",
        "lag_168_seasonal": "da_seasonal__rt_at_da_seasonal__rt_seasonal",
        "timesfm_3_frozen": "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3",
    }
    registered = stage["complete_three_leg_preregistered_baselines"]
    for name, system_key in system_keys.items():
        assert registered[name]["value"] == locked["systems"][system_key][
            "macro_mean_daily_revenue"
        ]

    diagnostic = stage["diagnostic_not_baseline"][
        "restricted_perfect_price_reference"
    ]
    assert diagnostic["value"] == locked["systems"][
        "da_oracle__rt_at_da_oracle__rt_oracle"
    ]["macro_mean_daily_revenue"]
