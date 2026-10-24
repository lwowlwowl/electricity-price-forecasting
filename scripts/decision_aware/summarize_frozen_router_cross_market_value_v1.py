#!/usr/bin/env python
"""Summarize standardized cross-market value of the frozen expert router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MARKETS = ("CAISO", "ERCOT", "NYISO", "PJM")
LABELS = {
    "transformer": "da_transformer__rt_at_da_transformer__rt_transformer",
    "dlinear": "da_dlinear__rt_at_da_dlinear__rt_dlinear",
    "static_hybrid": "da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid",
    "frozen_router": "da_frozen_router__rt_at_da_frozen_router__rt_frozen_router",
}


def _block_sample_mean(values: np.ndarray, rng: np.random.Generator) -> float:
    block = 7
    pieces: list[np.ndarray] = []
    while sum(len(piece) for piece in pieces) < len(values):
        start = int(rng.integers(len(values)))
        pieces.append(values[(start + np.arange(block)) % len(values)])
    return float(np.mean(np.concatenate(pieces)[:len(values)]))


def _market_equal_ci(series: list[np.ndarray], *, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(20261020)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        values[draw] = np.mean([_block_sample_mean(item, rng) for item in series])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", default="data/results/tail_expert_router_cross_market_value_v1"
    )
    parser.add_argument(
        "--tail-audit", default="data/results/tail_expert_router_cross_market_2025_v1.json"
    )
    parser.add_argument(
        "--output", default="data/results/tail_expert_router_cross_market_summary_v1.json"
    )
    args = parser.parse_args()

    tail = json.loads((ROOT / args.tail_audit).read_text(encoding="utf-8"))
    result = {
        "protocol": "frozen_ercot_router_cross_market_transfer_2025_v1",
        "status": "fixed_posthoc_transfer_audit_complete",
        "selection_allowed": False,
        "profit_interpretation": {
            "ERCOT": "existing frozen market claim",
            "CAISO_NYISO_PJM": "standardized research contract; not realized market profit",
        },
        "markets": {},
    }
    difference_series: list[np.ndarray] = []
    for market in MARKETS:
        value = json.loads(
            (ROOT / args.input_root / f"{market.lower()}.json").read_text(encoding="utf-8")
        )
        systems = value["systems"]
        means = {
            name: float(systems[label]["macro_mean_daily_revenue"])
            for name, label in LABELS.items()
        }
        best_static = max(
            (name for name in ("transformer", "dlinear", "static_hybrid")),
            key=means.__getitem__,
        )
        router = systems[LABELS["frozen_router"]]
        baseline = systems[LABELS[best_static]]
        days = sorted(router["macro_daily_net_revenue"])
        if days != sorted(baseline["macro_daily_net_revenue"]):
            raise ValueError(f"{market}路由与静态基线结算日期不一致")
        difference = np.asarray([
            router["macro_daily_net_revenue"][day]
            - baseline["macro_daily_net_revenue"][day]
            for day in days
        ], dtype=np.float64)
        difference_series.append(difference)
        node_difference = {
            node: float(
                router["nodes"][node]["mean_daily_revenue"]
                - baseline["nodes"][node]["mean_daily_revenue"]
            )
            for node in router["nodes"]
        }
        result["markets"][market] = {
            "standardized_mean_daily_value": means,
            "best_static_system": best_static,
            "router_minus_best_static": float(difference.mean()),
            "router_improved_nodes": int(sum(value > 0.0 for value in node_difference.values())),
            "node_count": int(len(node_difference)),
            "router_components": router["macro_mean_daily_revenue_components"],
            "best_static_components": baseline["macro_mean_daily_revenue_components"],
            "dlinear_route_share": float(tail["markets"][market]["route"]["dlinear_share"]),
            "tail_metrics": tail["markets"][market]["systems"]["frozen_router"],
        }

    router_market_equal = float(np.mean([
        item["standardized_mean_daily_value"]["frozen_router"]
        for item in result["markets"].values()
    ]))
    best_market_equal = float(np.mean([
        max(
            item["standardized_mean_daily_value"][name]
            for name in ("transformer", "dlinear", "static_hybrid")
        )
        for item in result["markets"].values()
    ]))
    positive_markets = int(sum(
        item["router_minus_best_static"] > 0.0 for item in result["markets"].values()
    ))
    result["market_equal"] = {
        "frozen_router": router_market_equal,
        "best_static_per_market": best_market_equal,
        "difference": router_market_equal - best_market_equal,
        "paired_7day_block_bootstrap_95ci": _market_equal_ci(difference_series),
        "positive_markets": positive_markets,
        "markets": len(MARKETS),
        "preregistered_point_estimate_gate_passed": bool(
            router_market_equal > best_market_equal and positive_markets >= 3
        ),
        "statistically_significant_at_95pct": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "frozen_router_cross_market_summary_complete",
        "output": output.relative_to(ROOT).as_posix(),
        **result["market_equal"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
