#!/usr/bin/env python
"""Summarize the frozen v4 final follow-up against every existing 2025 system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = (
    "da_implicitqp_final_robust_seed0_twothirds_v4__"
    "rt_at_da_implicitqp_final_robust_seed0_twothirds_v4__"
    "rt_implicitqp_final_robust_seed0_twothirds_v4"
)


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _paired_ci(values: np.ndarray, seed: int = 20261024) -> list[float]:
    rng = np.random.default_rng(seed)
    block = 7
    starts = np.arange(len(values))
    means = np.empty(4000, dtype=np.float64)
    for draw in range(len(means)):
        pieces = []
        size = 0
        while size < len(values):
            start = int(rng.choice(starts))
            piece = values[(start + np.arange(block)) % len(values)]
            pieces.append(piece)
            size += len(piece)
        means[draw] = np.concatenate(pieces)[: len(values)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _compare(candidate: dict, baseline: dict) -> dict:
    dates = sorted(candidate["macro_daily_net_revenue"])
    if dates != sorted(baseline["macro_daily_net_revenue"]):
        raise ValueError("candidate and comparator dates differ")
    candidate_daily = np.asarray(
        [candidate["macro_daily_net_revenue"][day] for day in dates], dtype=np.float64
    )
    baseline_daily = np.asarray(
        [baseline["macro_daily_net_revenue"][day] for day in dates], dtype=np.float64
    )
    delta = candidate_daily - baseline_daily
    nodes = sorted(candidate["nodes"])
    if nodes != sorted(baseline["nodes"]):
        raise ValueError("candidate and comparator nodes differ")
    node_differences = {
        node: float(
            candidate["nodes"][node]["mean_daily_revenue"]
            - baseline["nodes"][node]["mean_daily_revenue"]
        )
        for node in nodes
    }
    remove_count = min(5, len(delta) - 1)
    keep_without_best = np.argsort(delta)[: len(delta) - remove_count]
    worst_count = max(1, int(np.ceil(0.10 * len(delta))))
    baseline_worst = np.argsort(baseline_daily)[:worst_count]
    return {
        "candidate_mean": float(candidate_daily.mean()),
        "baseline_mean": float(baseline_daily.mean()),
        "mean_difference": float(delta.mean()),
        "relative_improvement_fraction": float(delta.mean() / baseline_daily.mean()),
        "paired_7day_block_bootstrap_95ci": _paired_ci(delta),
        "positive_day_fraction": float(np.mean(delta > 0.0)),
        "improved_nodes": int(sum(value > 0.0 for value in node_differences.values())),
        "node_count": len(nodes),
        "node_differences": node_differences,
        "difference_after_removing_5_best_candidate_days": float(
            delta[keep_without_best].mean()
        ),
        "difference_on_baseline_worst_10pct_days": float(
            delta[baseline_worst].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-settlement",
        default="data/results/global_implicit_qp_final_v4/settlement_2025.json",
    )
    parser.add_argument(
        "--locked-systems",
        default="data/results/allmarket_locked_2025_all_continuous_mpc_v2.json",
    )
    parser.add_argument(
        "--router-settlement",
        default="data/results/tail_expert_router_v1/followup_2025_mpc_v1.json",
    )
    parser.add_argument("--candidate-label", default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--output",
        default="data/results/global_implicit_qp_final_v4/summary_2025.json",
    )
    args = parser.parse_args()

    candidate_report = _load(args.candidate_settlement)
    locked_report = _load(args.locked_systems)
    router_report = _load(args.router_settlement)
    candidate = candidate_report["systems"][args.candidate_label]
    comparators = dict(locked_report["systems"])
    router_label, router = next(iter(router_report["systems"].items()))
    comparators[router_label] = router
    comparisons = {
        label: _compare(candidate, baseline)
        for label, baseline in comparators.items()
    }
    ranking = sorted(
        [
            {
                "system": args.candidate_label,
                "macro_mean_daily_revenue": candidate["macro_mean_daily_revenue"],
            },
            *[
                {
                    "system": label,
                    "macro_mean_daily_revenue": value["macro_mean_daily_revenue"],
                }
                for label, value in comparators.items()
            ],
        ],
        key=lambda item: item["macro_mean_daily_revenue"],
        reverse=True,
    )
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index
    result = {
        "status": "locked_2025_followup_complete_no_post_score_selection",
        "claim_boundary": (
            "2025 artifacts pre-existed before this candidate; this is a frozen "
            "follow-up confirmation, not a pristine first opening of 2025"
        ),
        "candidate": args.candidate_label,
        "candidate_macro_mean_daily_revenue": candidate[
            "macro_mean_daily_revenue"
        ],
        "candidate_components": candidate[
            "macro_mean_daily_revenue_components"
        ],
        "candidate_total_soc_violation_count": candidate[
            "total_soc_violation_count"
        ],
        "comparisons": comparisons,
        "ranking": ranking,
        "selection_allowed": False,
        "post_score_model_changes_allowed": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "final_v4_summary_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "candidate_revenue": result["candidate_macro_mean_daily_revenue"],
        "rank": next(
            item["rank"] for item in ranking if item["system"] == args.candidate_label
        ),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
