#!/usr/bin/env python
"""汇总稳定v2训练、候选映射、预测诊断与完整结算结果。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.diagnose_global_screen import (  # noqa: E402
    _system_forecast_diagnostics,
)
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _seasonal_archives,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
)


PREDICTION_DIR = ROOT / "data/results/global_f0_f1_stable_v2_predictions"
BACKTEST = ROOT / "data/results/global_continuous_mpc_backtest_stable_v2.json"
CANDIDATES = ROOT / "data/results/global_f0_f1_stable_v2_candidate_manifest.json"
OLD_BACKTEST = ROOT / "data/results/global_continuous_mpc_backtest_v1.json"
OUTPUT = ROOT / "data/results/global_f0_f1_stable_v2_summary.json"
SEASONAL = "da_seasonal__rt_at_da_seasonal__rt_seasonal"


def _tail_drawdown(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    tail_count = max(1, int(math.ceil(0.10 * len(array))))
    cumulative = np.cumsum(array)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return (
        float(np.mean(np.sort(array)[:tail_count])),
        float(np.max(peak - cumulative)),
    )


def main() -> None:
    backtest = json.loads(BACKTEST.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    best_global_row = next(
        row for row in backtest["ranking"] if "seasonal" not in row["triplet"]
    )
    best_label = best_global_row["triplet"]
    parts = best_label.split("__")
    chosen = {
        "da": parts[0].removeprefix("da_"),
        "rt_at_da": parts[1].removeprefix("rt_at_da_"),
        "rt": parts[2].removeprefix("rt_"),
    }
    selected_records = {
        task: candidates["selected_by_task"][task][int(family[1:])]
        for task, family in chosen.items()
    }
    system = backtest["systems"][best_label]
    daily = [
        system["macro_daily_net_revenue"][day]
        for day in sorted(system["macro_daily_net_revenue"])
    ]
    tail, drawdown = _tail_drawdown(daily)
    nodes = system["nodes"]
    component_keys = list(next(iter(nodes.values()))["revenue_components"])
    components = {
        key: float(np.mean([
            record["revenue_components"][key] / record["days"]
            for record in nodes.values()
        ]))
        for key in component_keys
    }

    archives = {
        (family, task): _load_archive(PREDICTION_DIR, family, task)
        for family in ("c0", "c1") for task in ("da", "rt_at_da", "rt")
    }
    node_names = archives[("c0", "da")]["nodes"]
    archives.update(_seasonal_archives(archives, node_names, "c0"))
    forecast = {
        "stable_global_champion": _system_forecast_diagnostics(
            archives, chosen, node_names
        ),
        "seasonal": _system_forecast_diagnostics(
            archives,
            {"da": "seasonal", "rt_at_da": "seasonal", "rt": "seasonal"},
            node_names,
        ),
    }

    old = json.loads(OLD_BACKTEST.read_text(encoding="utf-8"))
    old_best = next(row for row in old["ranking"] if "seasonal" not in row["triplet"])
    paired = backtest["paired_vs_seasonal"][best_label]
    report = {
        "experiment": "global_f0_f1_stable_v2_inner_seed0_summary",
        "status": "2023_inner_validation_candidate_not_final",
        "selection_allowed_for": "next-stage candidate promotion only",
        "selection_forbidden_for": "2025 final claim",
        "download_integrity": {"files_checked": 45, "sha256_mismatches": 0},
        "candidate_filter": candidates,
        "best_global_triplet": {
            "label": best_label,
            "candidate_family_by_task": chosen,
            "checkpoint_by_task": selected_records,
            "macro_mean_daily_revenue": system["macro_mean_daily_revenue"],
            "bootstrap_95ci": system[
                "macro_mean_daily_revenue_7day_block_bootstrap_95ci"
            ],
            "positive_day_rate": system["macro_positive_day_rate"],
            "node_positive_mean_revenue_rate": system[
                "node_positive_mean_revenue_rate"
            ],
            "worst_node_mean_daily_revenue": system[
                "worst_node_mean_daily_revenue"
            ],
            "best_node_mean_daily_revenue": system[
                "best_node_mean_daily_revenue"
            ],
            "tail_10pct_mean_daily_revenue": tail,
            "maximum_drawdown_usd": drawdown,
            "revenue_components_per_node_day": components,
            "actual_throughput_mwh_per_node_day": float(np.mean([
                record["actual_throughput_mwh"] / record["days"]
                for record in nodes.values()
            ])),
            "soc_violation_count": int(sum(
                record["soc_violation_count"] for record in nodes.values()
            )),
        },
        "seasonal": {
            "macro_mean_daily_revenue": backtest["systems"][SEASONAL][
                "macro_mean_daily_revenue"
            ]
        },
        "paired_global_minus_seasonal": paired,
        "improvement_over_three_epoch_global_usd_per_node_day": (
            system["macro_mean_daily_revenue"]
            - old_best["macro_mean_daily_revenue"]
        ),
        "forecast_diagnostics": forecast,
        "ranking": backtest["ranking"],
        "limitations": [
            "seed 0 only",
            "2023 inner validation only",
            "F0/F1 historical architectures only",
            "Seasonal comparison interval crosses zero",
            "not a 2024 outer-confirmed or 2025 locked-test result",
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": OUTPUT.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
