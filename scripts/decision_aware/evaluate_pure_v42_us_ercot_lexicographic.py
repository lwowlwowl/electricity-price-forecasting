#!/usr/bin/env python
"""ERCOT settlement audit with an explicit lexicographic v4 tie-break."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.continuous_mpc import locked_continuous_mpc_backtest  # noqa: E402
from decision_aware.global_evaluation import align_node_task_predictions  # noqa: E402
from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _aggregate,
    _compact_node,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


def _inputs(directory: Path, family: str, node_index: int) -> dict:
    archives = {
        task: _load_archive(directory, family, task, "inner")
        for task in ("da", "rt_at_da", "rt")
    }
    return align_node_task_predictions(
        _node_record(archives["da"], node_index),
        _node_record(archives["rt_at_da"], node_index),
        _node_record(archives["rt"], node_index),
    )


def _worker(
    config_path: str,
    base_dir: str,
    candidate_dir: str,
    node: str,
    node_index: int,
) -> tuple[str, dict, dict, dict]:
    cfg = PilotConfig.from_yaml(config_path)
    base_input = _inputs(Path(base_dir), "v4_official", node_index)
    candidate_input = _inputs(
        Path(candidate_dir), "pure_v42_memory_locked", node_index
    )
    base_report = locked_continuous_mpc_backtest(
        base_input["da_price_forecasts"],
        base_input["rt_at_da_price_forecasts"],
        base_input["rt_price_forecasts"],
        base_input["realized_da_price"],
        base_input["realized_rt_price"],
        base_input["local_timestamps"],
        cfg,
        include_hourly_details=True,
    )
    plans: dict[object, list[float | None]] = defaultdict(lambda: [None] * 24)
    for row in base_report["hourly"]:
        day = np.datetime64(row["delivery_date"]).astype(object)
        plans[day][int(row["hour_local"])] = float(row["da_plan_mwh"])
    locked_plans = {
        day: np.asarray(values, dtype=np.float64)
        for day, values in plans.items()
    }
    if any(not np.isfinite(value).all() for value in locked_plans.values()):
        raise ValueError(f"{node}: incomplete v4 plan map")
    candidate_report = locked_continuous_mpc_backtest(
        candidate_input["da_price_forecasts"],
        candidate_input["rt_at_da_price_forecasts"],
        candidate_input["rt_price_forecasts"],
        candidate_input["realized_da_price"],
        candidate_input["realized_rt_price"],
        candidate_input["local_timestamps"],
        cfg,
        locked_da_plans=locked_plans,
    )
    base_compact = _compact_node(base_report)
    candidate_compact = _compact_node(candidate_report)
    dates = sorted(base_compact["daily_net_revenue"])
    delta = np.asarray(
        [
            candidate_compact["daily_net_revenue"][day]
            - base_compact["daily_net_revenue"][day]
            for day in dates
        ],
        dtype=np.float64,
    )
    audit = {
        "mean_daily_revenue_delta": float(delta.mean()),
        "maximum_absolute_daily_revenue_delta": float(np.max(np.abs(delta))),
        "same_plan_digest": bool(
            base_compact["plan_digest"] == candidate_compact["plan_digest"]
        ),
        "same_action_digest": bool(
            base_compact["action_digest"] == candidate_compact["action_digest"]
        ),
    }
    return node, base_compact, candidate_compact, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", default="data/results/pure_v42_memory_us_ercot_predictions_v1"
    )
    parser.add_argument(
        "--candidate-dir",
        default="data/results/pure_v42_memory_locked_us_ercot_predictions_v1",
    )
    parser.add_argument(
        "--projection-summary",
        default="data/results/pure_v42_memory_locked_us_ercot_predictions_v1/summary.json",
    )
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        default="data/results/pure_v42_memory_lexicographic_us_ercot_v1.json",
    )
    args = parser.parse_args()

    projection = json.loads(
        (ROOT / args.projection_summary).read_text(encoding="utf-8")
    )
    if projection["maximum_target_objective_gap"] > 1e-6:
        raise ValueError("projection did not prove the v4 plan primary-optimal")

    base_directory = ROOT / args.base_dir
    with np.load(
        base_directory / "v4_official_da_inner_seed0.npz", allow_pickle=False
    ) as archive:
        nodes = archive["nodes"].astype(str).tolist()
    base_nodes: dict[str, dict] = {}
    candidate_nodes: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _worker,
                str((ROOT / args.config).resolve()),
                str(base_directory.resolve()),
                str((ROOT / args.candidate_dir).resolve()),
                node,
                index,
            ): node
            for index, node in enumerate(nodes)
        }
        for count, future in enumerate(as_completed(futures), start=1):
            node, base, candidate, audit = future.result()
            base_nodes[node] = base
            candidate_nodes[node] = candidate
            audits[node] = audit
            print(
                json.dumps(
                    {"event": "node_complete", "completed": count, "node": node}
                ),
                flush=True,
            )

    base = _aggregate(base_nodes)
    candidate = _aggregate(candidate_nodes)
    report = {
        "experiment": "pure_v42_memory_lexicographic_us_ercot_v1",
        "status": "2023_inner_controlled_audit",
        "lexicographic_contract": (
            "Primary: strengthened-price joint LP. Secondary: on its optimal "
            "face select the frozen-v4 plan. The inverse-LP KKT audit proves "
            "that plan is primary-optimal."
        ),
        "forecast_metrics": {
            "v4_mae_usd_per_mwh": projection["base_mae_usd_per_mwh"],
            "strengthened_locked_mae_usd_per_mwh": projection[
                "locked_candidate_mae_usd_per_mwh"
            ],
            "relative_change_pct": float(
                100.0
                * (
                    projection["locked_candidate_mae_usd_per_mwh"]
                    / projection["base_mae_usd_per_mwh"]
                    - 1.0
                )
            ),
        },
        "systems": {"v4": base, "strengthened_v4": candidate},
        "paired_audit": {
            "mean_daily_revenue_delta": float(
                candidate["macro_mean_daily_revenue"]
                - base["macro_mean_daily_revenue"]
            ),
            "maximum_absolute_daily_revenue_delta": float(
                max(
                    item["maximum_absolute_daily_revenue_delta"]
                    for item in audits.values()
                )
            ),
            "all_plan_digests_equal": bool(
                all(item["same_plan_digest"] for item in audits.values())
            ),
            "all_action_digests_equal": bool(
                all(item["same_action_digest"] for item in audits.values())
            ),
            "nodes": audits,
        },
        "projection_kkt_audit": {
            key: projection[key]
            for key in (
                "maximum_target_objective_gap",
                "maximum_stationarity_residual",
            )
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = dict(report)
    compact["systems"] = {
        key: {item: value for item, value in system.items() if item != "nodes"}
        for key, system in report["systems"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
