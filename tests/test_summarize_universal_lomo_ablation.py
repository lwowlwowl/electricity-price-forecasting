import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_universal_lomo_ablation import (
    CURRENT,
    PURE,
    SURGICAL,
    paired,
    summarize,
)


def _system(value):
    daily = {f"d{day}": value + day for day in range(14)}
    return {
        "macro_mean_daily_revenue": sum(daily.values()) / len(daily),
        "macro_mean_daily_revenue_7day_block_bootstrap_95ci": [value, value + 14],
        "macro_daily_net_revenue": daily,
    }


def test_paired_reports_constant_daily_improvement():
    result = paired(_system(12.0), _system(10.0), seed=1)
    assert result["mean_daily_difference"] == pytest.approx(2.0)
    assert result["seven_day_block_bootstrap_95ci"] == pytest.approx([2.0, 2.0])


def test_summarize_contains_pure_and_surgical_contracts():
    names = set(PURE.values()) | set(SURGICAL.values()) | {CURRENT}
    report = {
        "selection_allowed": True,
        "systems": {name: _system(float(index)) for index, name in enumerate(names)},
    }
    result = summarize(report)
    assert set(result["pure_structures"]) == set(PURE)
    assert set(result["single_task_replacements"]) == set(SURGICAL)


def test_summarize_rejects_report_only_input():
    with pytest.raises(ValueError, match="未允许"):
        summarize({"selection_allowed": False, "systems": {}})
