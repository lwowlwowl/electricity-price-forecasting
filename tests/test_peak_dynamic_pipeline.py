from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_peak_dynamic_runner_freezes_protocol() -> None:
    text = (
        ROOT / "scripts/decision_aware/run_cloud_da_peakdynamic_v1.sh"
    ).read_text(encoding="utf-8")
    assert "--initialize-from \"$BASE_CHECKPOINT\"" in text
    assert "select_peak_dynamic_checkpoint.py" in text
    assert "--mae-guard-fraction 0.02" in text
    assert "--evaluation-stage outer_frozen" in text
    assert "--no-selection-allowed" in text
    assert "--include-baselines xgboost_robust" in text
    assert "--evaluation-stage locked_2025" not in text


def test_peak_dynamic_summary_reports_profit_and_curve_shape() -> None:
    text = (
        ROOT / "scripts/decision_aware/summarize_peak_dynamic_da.py"
    ).read_text(encoding="utf-8")
    assert "daily_peak_report" in text
    assert "macro_mean_daily_revenue" in text
    assert "seven_day_block_bootstrap_95ci" in text
    assert "macro_mean_daily_revenue_components" in text
    assert "XGBOOST" in text
