#!/usr/bin/env python
"""按事前冻结的盈利主终点审计2025结论，不参与模型选择。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _block_bootstrap_ci,
)


EXPECTED_MARKETS = {"CAISO", "ERCOT", "NYISO", "PJM"}
EXPECTED_FORECAST_SYSTEMS = {
    "transformer_seed0", "transformer_seed1", "transformer_seed2",
    "transformer_ensemble", "timesfm3", "timesfm25", "pooled_xgboost",
    "per_node_xgboost", "dlinear", "seasonal", "lear",
}
EXPECTED_SETTLEMENT_LABELS = {
    "da_seed0__rt_at_da_seed0__rt_seed0",
    "da_seed1__rt_at_da_seed1__rt_seed1",
    "da_seed2__rt_at_da_seed2__rt_seed2",
    "da_ensemble__rt_at_da_ensemble__rt_ensemble",
    "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3",
    "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25",
    "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost",
    "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost",
    "da_dlinear__rt_at_da_dlinear__rt_dlinear",
    "da_seasonal__rt_at_da_seasonal__rt_seasonal",
    "da_oracle__rt_at_da_oracle__rt_oracle",
}


def classify_profit_difference(mean_difference: float, ci: list[float]) -> str:
    if mean_difference > 0.0 and ci[0] > 0.0:
        return "stable_superiority"
    if mean_difference > 0.0:
        return "positive_point_estimate_only"
    return "no_superiority"


def _load(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default=(
            "data/results/allmarket_final_2025_profit_analysis_preregistration_v1/"
            "protocol.json"
        ),
    )
    parser.add_argument(
        "--summary", default="data/results/allmarket_locked_2025_summary_v2.json"
    )
    parser.add_argument(
        "--output",
        default="data/results/allmarket_locked_2025_profit_claim_audit_v1.json",
    )
    args = parser.parse_args()
    protocol = _load(ROOT / args.protocol)
    summary = _load(ROOT / args.summary)
    if summary.get("year") != 2025 or summary.get("selection_allowed") is not False:
        raise ValueError("最终汇总不是2025不可选模结果")
    if set(summary.get("markets", {})) != EXPECTED_MARKETS:
        raise ValueError("四市场结果不完整")
    for market, market_report in summary["markets"].items():
        systems = set(market_report.get("systems", {}))
        if systems != EXPECTED_FORECAST_SYSTEMS:
            raise ValueError(f"{market}预测系统不完整: {sorted(EXPECTED_FORECAST_SYSTEMS - systems)}")

    settlement = summary.get("ercot_settlement", {})
    systems = settlement.get("systems", {})
    if set(systems) != EXPECTED_SETTLEMENT_LABELS:
        raise ValueError("ERCOT完整结算系统标签不匹配")
    endpoint = protocol["primary_economic_endpoint"]
    model_label = endpoint["settlement_label"]
    comparator_label = endpoint["comparator_settlement_label"]
    model_daily = systems[model_label]["macro_daily_net_revenue"]
    comparator_daily = systems[comparator_label]["macro_daily_net_revenue"]
    days = sorted(model_daily)
    if sorted(comparator_daily) != days:
        raise ValueError("主系统与pooled XGBoost结算日期不一致")
    difference = np.asarray(
        [model_daily[day] - comparator_daily[day] for day in days], dtype=np.float64
    )
    mean_difference = float(difference.mean())
    ci = _block_bootstrap_ci(difference)
    classification = classify_profit_difference(mean_difference, ci)
    derived = summary["ercot_settlement_derived"][comparator_label][
        "transformer_ensemble_minus_system"
    ]
    if not np.isclose(derived["mean_daily_difference"], mean_difference, atol=1e-10):
        raise ValueError("汇总中的主盈利差与逐日重算不一致")
    if not np.allclose(derived["paired_7day_block_bootstrap_95ci"], ci, atol=1e-10):
        raise ValueError("汇总中的主盈利区间与逐日重算不一致")

    report = {
        "protocol": "allmarket_locked_2025_profit_claim_audit_v1",
        "complete": True,
        "selection_allowed": False,
        "primary_endpoint": endpoint,
        "transformer_mean_daily_revenue": systems[model_label]["macro_mean_daily_revenue"],
        "comparator_mean_daily_revenue": systems[comparator_label]["macro_mean_daily_revenue"],
        "mean_daily_difference": mean_difference,
        "paired_7day_block_bootstrap_95ci": ci,
        "classification": classification,
        "mae_can_override": False,
        "report_regardless_of_outcome": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "allmarket_locked_2025_profit_claim_audit_complete",
        "classification": classification,
        "mean_daily_difference": mean_difference,
        "ci": ci,
    }))


if __name__ == "__main__":
    main()

