#!/usr/bin/env python
"""Summarize strict 2023 cross-market regret routing with economic endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.global_data import MARKET_ORDER  # noqa: E402
from scripts.decision_aware.evaluate_global_continuous_mpc import _block_bootstrap_ci  # noqa: E402


FAMILIES = (
    "transformer", "static_hybrid", "economic_regret_tree",
    "mae_regret_tree", "economic_no_tail_tree",
)


def _label(family: str) -> str:
    return f"da_{family}__rt_at_da_{family}__rt_{family}"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _daily(system: dict) -> tuple[list[str], np.ndarray]:
    mapping = system["macro_daily_net_revenue"]
    dates = sorted(mapping)
    return dates, np.asarray([mapping[date] for date in dates], dtype=np.float64)


def _forecast_metrics(candidate_root: Path, market: str, family: str) -> dict:
    directory = candidate_root / market.lower()
    da = _load_npz(directory / f"{family}_da_inner_seed0.npz")
    rt = _load_npz(directory / f"{family}_rt_at_da_inner_seed0.npz")
    da_error = da["prediction"].astype(np.float64) - da["target_da"].astype(np.float64)
    rt_error = rt["prediction"].astype(np.float64) - rt["target_rt"].astype(np.float64)
    predicted_spread = da["prediction"].astype(np.float64) - rt["prediction"].astype(np.float64)
    actual_spread = da["target_da"].astype(np.float64) - rt["target_rt"].astype(np.float64)
    return {
        "da_mae": float(np.mean(np.abs(da_error))),
        "rt_at_da_mae": float(np.mean(np.abs(rt_error))),
        "spread_direction_accuracy": float(np.mean(np.sign(predicted_spread) == np.sign(actual_spread))),
        "spread_amplitude_mae": float(np.mean(np.abs(predicted_spread - actual_spread))),
    }


def _remove_top(values: np.ndarray, count: int) -> float | None:
    if len(values) <= count:
        return None
    indices = np.argsort(values)[-count:]
    keep = np.ones(len(values), dtype=bool)
    keep[indices] = False
    return float(values[keep].mean())


def _node_improvement(candidate: dict, baseline: dict) -> dict:
    rows = {}
    for node in sorted(candidate["nodes"]):
        candidate_daily = candidate["nodes"][node]["daily_net_revenue"]
        baseline_daily = baseline["nodes"][node]["daily_net_revenue"]
        dates = sorted(candidate_daily)
        difference = np.asarray([
            candidate_daily[date] - baseline_daily[date] for date in dates
        ], dtype=np.float64)
        rows[node] = float(difference.mean())
    return {
        "positive_nodes": int(sum(value > 0 for value in rows.values())),
        "node_count": len(rows),
        "worst_node_difference": float(min(rows.values())),
        "best_node_difference": float(max(rows.values())),
        "per_node_mean_difference": rows,
    }


def _tail_attribution(candidate_root: Path, market: str, settlement: dict, baseline_label: str) -> dict:
    directory = candidate_root / market.lower()
    truth = _load_npz(directory / "transformer_da_inner_seed0.npz")
    rt_truth = _load_npz(directory / "transformer_rt_at_da_inner_seed0.npz")
    nodes = truth["nodes"].astype(str)
    dates = truth["dates"].astype(str)
    true_da = truth["target_da"].astype(np.float64)
    true_rt = rt_truth["target_rt"].astype(np.float64)
    q99 = float(np.quantile(true_da, 0.99))
    ramp_threshold = float(np.quantile(np.abs(np.diff(true_da, axis=-1)), 0.99))
    spread_threshold = float(np.quantile(np.abs(true_da - true_rt), 0.95))
    flags = {
        "negative_price": np.any((true_da < 0.0) | (true_rt < 0.0), axis=-1),
        "positive_spike_q99": np.any(true_da >= q99, axis=-1),
        "rapid_ramp_q99": np.any(np.abs(np.diff(true_da, axis=-1)) >= ramp_threshold, axis=-1),
        "extreme_da_rt_spread_q95": np.any(np.abs(true_da - true_rt) >= spread_threshold, axis=-1),
    }
    flags["ordinary"] = ~np.logical_or.reduce(list(flags.values()))
    candidate = settlement["systems"][_label("economic_regret_tree")]
    baseline = settlement["systems"][baseline_label]
    values = []
    group_flags = {name: [] for name in flags}
    date_to_index = {date: index for index, date in enumerate(dates)}
    for node_index, node in enumerate(nodes):
        left = candidate["nodes"][node]["daily_net_revenue"]
        right = baseline["nodes"][node]["daily_net_revenue"]
        for date in sorted(left):
            values.append(float(left[date]) - float(right[date]))
            day_index = date_to_index[date]
            for name, mask in flags.items():
                group_flags[name].append(bool(mask[node_index, day_index]))
    improvement = np.asarray(values, dtype=np.float64)
    result = {}
    for name, selected_values in group_flags.items():
        selected = np.asarray(selected_values, dtype=bool)
        result[name] = {
            "node_day_count": int(selected.sum()),
            "mean_profit_difference": float(improvement[selected].mean()) if selected.any() else None,
            "total_profit_difference": float(improvement[selected].sum()),
            "share_of_total_improvement": (
                float(improvement[selected].sum() / improvement.sum())
                if abs(float(improvement.sum())) > 1e-12 else None
            ),
        }
    return {
        "subsets_are_nonexclusive_except_ordinary": True,
        "thresholds": {
            "da_q99": q99,
            "absolute_da_ramp_q99": ramp_threshold,
            "absolute_da_rt_spread_q95": spread_threshold,
        },
        "groups": result,
    }


def _market_equal_bootstrap(market_differences: dict[str, np.ndarray], draws: int = 4000) -> list[float]:
    rng = np.random.default_rng(20261021)
    results = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means = []
        for values in market_differences.values():
            pieces = []
            while sum(len(piece) for piece in pieces) < len(values):
                start = int(rng.integers(0, len(values)))
                pieces.append(values[(start + np.arange(7)) % len(values)])
            means.append(float(np.concatenate(pieces)[:len(values)].mean()))
        results[draw] = np.mean(means)
    return [float(np.quantile(results, 0.025)), float(np.quantile(results, 0.975))]


def summarize(
    candidate_root: Path,
    settlement_root: Path,
    *,
    history_assisted: bool = False,
) -> dict:
    markets = {}
    main_differences = {}
    main_positive = 0
    economic_beats_mae = 0
    uses_tail_all = True
    for market in MARKET_ORDER:
        settlement = json.loads((settlement_root / f"{market.lower()}.json").read_text(encoding="utf-8"))
        manifest = json.loads((candidate_root / market.lower() / "manifest.json").read_text(encoding="utf-8"))
        systems = settlement["systems"]
        static_labels = [_label("transformer"), _label("static_hybrid")]
        baseline_label = max(static_labels, key=lambda value: systems[value]["macro_mean_daily_revenue"])
        baseline = systems[baseline_label]
        baseline_dates, baseline_daily = _daily(baseline)
        rows = {}
        for family in FAMILIES:
            system = systems[_label(family)]
            dates, daily = _daily(system)
            if dates != baseline_dates:
                raise ValueError(f"{market} {family}结算日期不一致")
            difference = daily - baseline_daily
            rows[family] = {
                "mean_daily_net_revenue": system["macro_mean_daily_revenue"],
                "difference_vs_best_static": float(difference.mean()),
                "difference_7day_block_bootstrap_95ci": _block_bootstrap_ci(difference, seed=20261021),
                "difference_without_top_1_day": _remove_top(difference, 1),
                "difference_without_top_5_days": _remove_top(difference, 5),
                "difference_without_top_10_days": _remove_top(difference, 10),
                "revenue_components": system["macro_mean_daily_revenue_components"],
                "mean_daily_throughput_mwh": system["macro_mean_daily_actual_throughput_mwh"],
                "forecast": _forecast_metrics(candidate_root, market, family),
                "node_robustness": _node_improvement(system, baseline),
            }
            if family == "economic_regret_tree":
                main_differences[market] = difference
        main = rows["economic_regret_tree"]
        if main["difference_vs_best_static"] > 0:
            main_positive += 1
        if main["mean_daily_net_revenue"] >= rows["mae_regret_tree"]["mean_daily_net_revenue"]:
            economic_beats_mae += 1
        used = manifest["models"]["economic_regret_tree"]["features"]
        no_tail = set(manifest["no_tail_feature_names"])
        uses_tail = any(name not in no_tail for name in used)
        uses_tail_all = uses_tail_all and uses_tail
        markets[market] = {
            "best_static_family": baseline_label,
            "systems": rows,
            "router": manifest["models"]["economic_regret_tree"],
            "economic_router_uses_tail_or_spread_feature": uses_tail,
            "tail_profit_attribution": _tail_attribution(candidate_root, market, settlement, baseline_label),
        }
    market_means = [float(value.mean()) for value in main_differences.values()]
    market_equal = float(np.mean(market_means))
    ci = _market_equal_bootstrap(main_differences)
    success = {
        "positive_vs_best_static_at_least_3_of_4": main_positive >= 3,
        "market_equal_profit_difference_positive": market_equal > 0,
        "economic_not_worse_than_mae_router_at_least_3_of_4": economic_beats_mae >= 3,
        "tail_or_spread_feature_used_in_every_fold": uses_tail_all,
    }
    success["main_method_passes_pre_registered_core_gate"] = all(success.values())
    return {
        "experiment": (
            "crossmarket_history_assisted_economic_regret_router_v1"
            if history_assisted
            else "strict_crossmarket_lomo_economic_regret_router_v1"
        ),
        "deployment_setting": "history_assisted_adaptation" if history_assisted else "strict_zero_shot_experts",
        "year": 2023,
        "primary_endpoint": "complete LP/MPC net profit under standardized research contract",
        "locked_2025_read": False,
        "information_boundary": (
            {
                "transformer_training_markets": "three source markets, excluding the evaluated market",
                "dlinear_training_markets": "all four markets using 2020-2022 history",
                "router_training_markets": "three source markets using 2023 node-day outcomes",
                "evaluated_market_2023_labels_used_for_router_training": False,
            }
            if history_assisted
            else {
                "transformer_training_markets": "three source markets, excluding the evaluated market",
                "dlinear_training_markets": "three source markets, excluding the evaluated market",
                "router_training_markets": "three source markets using 2023 node-day outcomes",
                "evaluated_market_2023_labels_used_for_router_training": False,
            }
        ),
        "markets": markets,
        "market_equal_summary": {
            "mean_daily_profit_difference_vs_best_static": market_equal,
            "hierarchical_7day_block_bootstrap_95ci": ci,
            "positive_markets": main_positive,
            "economic_router_beats_or_ties_mae_router_markets": economic_beats_mae,
        },
        "pre_registered_gate": success,
        "claim": (
            "history_assisted_gate_passed"
            if history_assisted and success["main_method_passes_pre_registered_core_gate"]
            else "history_assisted_gate_failed_stop_router_research"
            if history_assisted
            else "strict_crossmarket_gate_passed"
            if success["main_method_passes_pre_registered_core_gate"]
            else "strict_crossmarket_gate_failed_route_must_not_be_tuned_on_held_markets"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", default="data/results/crossmarket_regret_router_candidates_v1")
    parser.add_argument("--settlement-root", default="data/results/crossmarket_regret_router_settlement_v1")
    parser.add_argument("--output", default="data/results/crossmarket_regret_router_summary_v1.json")
    parser.add_argument("--history-assisted", action="store_true")
    args = parser.parse_args()
    result = summarize(
        ROOT / args.candidate_root,
        ROOT / args.settlement_root,
        history_assisted=args.history_assisted,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "crossmarket_regret_router_summary_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "market_equal_summary": result["market_equal_summary"],
        "pre_registered_gate": result["pre_registered_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
