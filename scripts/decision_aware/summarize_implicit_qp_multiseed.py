#!/usr/bin/env python
"""Summarize fixed implicit-QP seeds and their preregistered ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def _block_ci(
    values: np.ndarray, *, block: int = 7, draws: int = 5000, seed: int = 20261024
) -> list[float]:
    generator = np.random.default_rng(seed)
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    block_count = int(np.ceil(n / block))
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = generator.choice(starts, size=block_count, replace=True)
        sample = np.concatenate([values[start : start + block] for start in selected])[:n]
        means[index] = sample.mean()
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _comparison(candidate: dict, baseline: dict, *, seed: int) -> dict:
    dates = sorted(set(candidate["macro_daily_net_revenue"]) & set(
        baseline["macro_daily_net_revenue"]
    ))
    if not dates:
        raise ValueError("candidate and baseline have no common dates")
    difference = np.asarray([
        candidate["macro_daily_net_revenue"][day]
        - baseline["macro_daily_net_revenue"][day]
        for day in dates
    ])
    node_differences = {
        node: candidate["nodes"][node]["mean_daily_revenue"]
        - baseline["nodes"][node]["mean_daily_revenue"]
        for node in sorted(baseline["nodes"])
    }
    keep = np.ones(len(difference), dtype=bool)
    keep[np.argsort(difference)[-5:]] = False
    ci = _block_ci(difference, seed=seed)
    return {
        "macro_mean_daily_revenue": candidate["macro_mean_daily_revenue"],
        "baseline_macro_mean_daily_revenue": baseline["macro_mean_daily_revenue"],
        "mean_daily_difference": float(difference.mean()),
        "paired_7day_block_bootstrap_95ci": ci,
        "difference_after_removing_5_best_delta_days": float(difference[keep].mean()),
        "improved_nodes": int(sum(value > 0.0 for value in node_differences.values())),
        "node_count": len(node_differences),
        "node_differences": node_differences,
        "passes_point_estimate": bool(difference.mean() > 0.0),
        "passes_paired_interval": bool(ci[0] > 0.0),
        "passes_remove_top5": bool(difference[keep].mean() > 0.0),
    }


def summarize(
    report: dict,
    seed_families: tuple[str, ...],
    ensemble_family: str,
    baseline_family: str,
) -> dict:
    if len(seed_families) != 3:
        raise ValueError("exactly three preregistered seed families are required")
    systems = report["systems"]
    baseline = systems[_label(baseline_family)]
    comparisons = {
        family: _comparison(systems[_label(family)], baseline, seed=20261024 + index)
        for index, family in enumerate(seed_families)
    }
    ensemble = _comparison(
        systems[_label(ensemble_family)], baseline, seed=20261034
    )
    revenues = np.asarray([
        comparisons[family]["macro_mean_daily_revenue"] for family in seed_families
    ])
    return {
        "experiment": "implicit_qp_fixed_multiseed_validation_v1",
        "status": "2023_inner_fixed_method_validation",
        "selection_year": 2023,
        "2024_used": False,
        "2025_used": False,
        "seed_selection_by_profit": False,
        "frozen_method": {
            "scope": "decoder_head",
            "decision_weight": 0.60,
            "learning_rate": 2e-5,
            "quadratic_regularization": 0.25,
            "barrier_temperature": 0.10,
            "epochs": 3,
            "episodes_per_node": 2,
        },
        "seed_families": list(seed_families),
        "individual_seeds": comparisons,
        "across_seed_mean_daily_revenue": float(revenues.mean()),
        "across_seed_sample_std_daily_revenue": float(revenues.std(ddof=1)),
        "fixed_equal_weight_ensemble_family": ensemble_family,
        "fixed_equal_weight_ensemble": ensemble,
        "baseline_family": baseline_family,
        "source": report.get("experiment", "unknown"),
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Implicit-QP fixed multiseed validation", "",
        "> 2023 inner only; method frozen before seed 1/2; no profit-based seed selection.", "",
        "| System | Revenue (USD/node-day) | Delta vs XGBoost | Paired 95% CI | Improved nodes | Remove-best-5 delta |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for family, item in result["individual_seeds"].items():
        ci = item["paired_7day_block_bootstrap_95ci"]
        lines.append(
            f"| {family} | {item['macro_mean_daily_revenue']:.4f} | "
            f"{item['mean_daily_difference']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{item['improved_nodes']}/{item['node_count']} | "
            f"{item['difference_after_removing_5_best_delta_days']:+.4f} |"
        )
    item = result["fixed_equal_weight_ensemble"]
    ci = item["paired_7day_block_bootstrap_95ci"]
    lines.append(
        f"| fixed equal-weight ensemble | {item['macro_mean_daily_revenue']:.4f} | "
        f"{item['mean_daily_difference']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
        f"{item['improved_nodes']}/{item['node_count']} | "
        f"{item['difference_after_removing_5_best_delta_days']:+.4f} |"
    )
    lines.extend((
        "", f"Across-seed mean: {result['across_seed_mean_daily_revenue']:.4f}.",
        f"Across-seed sample standard deviation: {result['across_seed_sample_std_daily_revenue']:.4f}.",
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/results/global_implicit_qp_multiseed_continuous_mpc_v1.json",
    )
    parser.add_argument(
        "--seed-families",
        default=(
            "implicitqp_decoder_balanced_seed0,"
            "implicitqp_decoder_balanced_seed1,"
            "implicitqp_decoder_balanced_seed2"
        ),
    )
    parser.add_argument(
        "--ensemble-family", default="implicitqp_decoder_balanced_ensemble"
    )
    parser.add_argument("--baseline-family", default="xgbpooled")
    parser.add_argument(
        "--output-dir",
        default="data/results/global_implicit_qp_multiseed_summary_v1",
    )
    args = parser.parse_args()
    seed_families = tuple(
        value.strip() for value in args.seed_families.split(",") if value.strip()
    )
    report = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    result = summarize(report, seed_families, args.ensemble_family, args.baseline_family)
    result["source_path"] = args.input
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "event": "implicit_qp_multiseed_summary_ready",
        "seed_mean": result["across_seed_mean_daily_revenue"],
        "seed_std": result["across_seed_sample_std_daily_revenue"],
        "ensemble": result["fixed_equal_weight_ensemble"]["macro_mean_daily_revenue"],
    }))


if __name__ == "__main__":
    main()
