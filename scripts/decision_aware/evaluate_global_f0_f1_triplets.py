#!/usr/bin/env python
"""在完整2023双结算合同下评价8个ERCOT Global F0/F1三元组。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.backtest_dual import locked_dual_backtest_by_node  # noqa: E402
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_evaluation import (  # noqa: E402
    align_node_task_predictions,
    compact_backtest_report,
    enumerate_fusion_triplets,
)


def _load_archive(
    directory: Path, family: str, task: str, phase: str = "inner"
) -> dict:
    path = directory / f"{family}_{task}_{phase}_seed0.npz"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("family") != family or metadata.get("task") != task:
            raise ValueError(f"预测档案元数据错误: {path}")
        if metadata.get("phase", phase) != phase:
            raise ValueError(f"预测档案phase错误: {path}")
        result = {
            "path": path,
            "metadata": metadata,
            "nodes": archive["nodes"].astype(str),
            "dates": archive["dates"].astype(str),
            "prediction": archive["prediction"].astype(np.float32),
            "target_rt": archive["target_rt"].astype(np.float32),
        }
        if "target_da" in archive.files:
            result["target_da"] = archive["target_da"].astype(np.float32)
        if "executed_timestamps" in archive.files:
            result["executed_timestamps"] = archive[
                "executed_timestamps"
            ].astype(str)
        if "timezone" in archive.files:
            result["timezone"] = str(archive["timezone"].item())
    return result


def _node_record(archive: dict, node_index: int) -> dict:
    keys = (
        "dates", "prediction", "target_da", "target_rt",
        "executed_timestamps", "timezone",
    )
    record = {}
    for key in keys:
        if key not in archive:
            continue
        value = archive[key]
        if key in {"dates", "executed_timestamps", "timezone"}:
            record[key] = value
        else:
            record[key] = value[node_index]
    return record


def _forecast_summary(archives: dict) -> dict:
    reports = {}
    for (family, task), archive in archives.items():
        error = archive["prediction"].astype(np.float64) - archive[
            "target_rt" if task != "da" else "target_da"
        ].astype(np.float64)
        reports[f"{family}_{task}"] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "elements": int(error.size),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument(
        "--prediction-dir", default="data/results/global_f0_f1_predictions_v1"
    )
    parser.add_argument(
        "--coordination-modes",
        default="rt_only,follow_da,plan_track_topk",
    )
    parser.add_argument(
        "--output", default="data/results/global_f0_f1_triplet_backtest_v1.json"
    )
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    modes = tuple(
        value.strip() for value in args.coordination_modes.split(",") if value.strip()
    )
    allowed_modes = {"rt_only", "follow_da", "plan_track_topk", "lookahead_mpc"}
    unknown = sorted(set(modes) - allowed_modes)
    if unknown:
        raise ValueError(f"未知coordination mode: {unknown}")
    directory = ROOT / args.prediction_dir
    archives = {
        (family, task): _load_archive(directory, family, task)
        for family in ("f0", "f1")
        for task in ("da", "rt_at_da", "rt")
    }
    reference_nodes = archives[("f0", "da")]["nodes"]
    for key, archive in archives.items():
        if not np.array_equal(reference_nodes, archive["nodes"]):
            raise ValueError(f"{key}的节点顺序不一致")
    # 同一任务的F0/F1必须共享完全相同的真值和时间合同。
    for task in ("da", "rt_at_da", "rt"):
        left, right = archives[("f0", task)], archives[("f1", task)]
        if not np.array_equal(left["dates"], right["dates"]):
            raise ValueError(f"{task}的F0/F1日期不一致")
        for key in ("target_da", "target_rt", "executed_timestamps"):
            if key in left and not np.array_equal(left[key], right[key]):
                raise ValueError(f"{task}的F0/F1字段{key}不一致")

    combinations = enumerate_fusion_triplets()
    systems = {}
    started_all = perf_counter()
    for combination in combinations:
        label = (
            f"da_{combination['da']}__rt_at_da_{combination['rt_at_da']}"
            f"__rt_{combination['rt']}"
        )
        node_inputs = {}
        for node_index, node in enumerate(reference_nodes):
            node_inputs[str(node)] = align_node_task_predictions(
                _node_record(archives[(combination["da"], "da")], node_index),
                _node_record(
                    archives[(combination["rt_at_da"], "rt_at_da")], node_index
                ),
                _node_record(archives[(combination["rt"], "rt")], node_index),
            )
        systems[label] = {"fusion": combination, "modes": {}}
        for mode in modes:
            started = perf_counter()
            report = locked_dual_backtest_by_node(
                node_inputs,
                cfg,
                da_k_charge=cfg.topk_k_charge,
                da_k_discharge=cfg.topk_k_discharge,
                rt_k_charge=cfg.rt_topk_k_charge,
                rt_k_discharge=cfg.rt_topk_k_discharge,
                coordination_mode=mode,
                use_deviation_penalty=True,
            )
            compact = compact_backtest_report(report)
            compact["elapsed_seconds"] = perf_counter() - started
            systems[label]["modes"][mode] = compact
            print(json.dumps({
                "event": "triplet_mode_complete",
                "triplet": label,
                "mode": mode,
                "macro_mean_daily_revenue": compact[
                    "macro_mean_daily_revenue"
                ],
                "seconds": compact["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)

    rankings = {}
    for mode in modes:
        rankings[mode] = sorted(
            (
                {
                    "rank": 0,
                    "triplet": label,
                    "macro_mean_daily_revenue": item["modes"][mode][
                        "macro_mean_daily_revenue"
                    ],
                }
                for label, item in systems.items()
            ),
            key=lambda item: item["macro_mean_daily_revenue"],
            reverse=True,
        )
        for rank, item in enumerate(rankings[mode], start=1):
            item["rank"] = rank

    report = {
        "experiment": "global_f0_f1_triplet_backtest_v1",
        "status": "diagnostic_seed0_three_epoch",
        "selection_allowed": False,
        "experiment_contract": FORMAL_EXPERIMENT_CONTRACT.to_dict(),
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "market": "ERCOT",
        "nodes": reference_nodes.tolist(),
        "common_evaluation_days": int(
            len(set(archives[("f0", "da")]["dates"]) & set(
                archives[("f0", "rt")]["dates"]
            ))
        ),
        "split": "2023 inner validation only",
        "settlement": {
            "deviation_penalty_enabled": True,
            "deviation_tolerance_fraction": 0.03,
            "deviation_excess_multiplier_abs_rt_price": 2.0,
            "degradation_usd_per_actual_throughput_mwh": cfg.bess_kappa,
            "terminal_soc_monetized": False,
            "soc_scope": "independent_per_node; continuous within node segments",
        },
        "policy_warning": (
            "These are existing discrete HardTopK/lookahead diagnostics, not the "
            "future formal continuous DA+RT MPC contract."
        ),
        "forecast_metrics": _forecast_summary(archives),
        "systems": systems,
        "rankings": rankings,
        "runtime_seconds": perf_counter() - started_all,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "saved",
        "path": output.relative_to(ROOT).as_posix(),
        "seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
