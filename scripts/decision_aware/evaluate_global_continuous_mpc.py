#!/usr/bin/env python
"""并行评价ERCOT多节点预测驱动连续DA-MILP/RT-MPC。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.continuous_mpc import locked_continuous_mpc_backtest  # noqa: E402
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_evaluation import (  # noqa: E402
    align_node_task_predictions,
    enumerate_fusion_triplets,
)
from scripts.decision_aware.evaluate_global_f0_f1_triplets import (  # noqa: E402
    _load_archive,
    _node_record,
)


BASELINE_TASK_MODELS = {
    "xgboost_robust": {
        "da": "xgboost_absolute",
        "rt_at_da": "xgboost_pseudohuber",
        "rt": "xgboost_absolute",
    },
}


def _label(combination: dict[str, str]) -> str:
    return (
        f"da_{combination['da']}__rt_at_da_{combination['rt_at_da']}"
        f"__rt_{combination['rt']}"
    )


def _tail_and_drawdown(daily: dict[str, dict]) -> tuple[float, float]:
    values = np.asarray([daily[key]["net_revenue"] for key in sorted(daily)])
    tail_count = max(1, int(math.ceil(0.10 * len(values))))
    tail = float(np.mean(np.sort(values)[:tail_count]))
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    drawdown = float(np.max(running_peak - cumulative))
    return tail, drawdown


def _compact_node(report: dict) -> dict:
    tail, drawdown = _tail_and_drawdown(report["daily"])
    return {
        "days": report["days"],
        "mean_daily_revenue": report["mean_daily_revenue"],
        "positive_day_rate": report["positive_day_rate"],
        "tail_10pct_mean_daily_revenue": tail,
        "maximum_drawdown_usd": drawdown,
        "total_revenue": report["total_revenue"],
        "actual_throughput_mwh": report["actual_throughput_mwh"],
        "soc_violation_count": report["soc_violation_count"],
        "revenue_components": report["revenue_components"],
        "segments": report["segments"],
        "state_resets_due_to_gaps": report["state_resets_due_to_gaps"],
        "solver_calls": report["solver_calls"],
        "actual_final_soc_mwh": report["actual_state"]["final_soc_mwh"],
        "plan_final_soc_mwh": report["plan_feasibility"]["final_soc_mwh"],
        "action_digest": report["actual_state"]["action_digest"],
        "plan_digest": report["da_plan_action_digest"],
        "daily_net_revenue": {
            key: value["net_revenue"] for key, value in report["daily"].items()
        },
    }


def _worker(config_path: str, label: str, node: str, node_input: dict) -> tuple:
    # 避免每个HiGHS进程再自行扩张成大量线程。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = PilotConfig.from_yaml(config_path)
    report = locked_continuous_mpc_backtest(
        node_input["da_price_forecasts"],
        node_input["rt_at_da_price_forecasts"],
        node_input["rt_price_forecasts"],
        node_input["realized_da_price"],
        node_input["realized_rt_price"],
        node_input["local_timestamps"],
        cfg,
    )
    return label, node, _compact_node(report)


def _process_safe(value):
    """把共享CPU Tensor改成普通NumPy数组。

    PyTorch在向ProcessPool传递Tensor时会为每个storage创建文件描述符。
    年度多节点评价会因此超过Linux的open-files上限。求解器本来就
    只需要CPU数值，所以在提交进程前转成NumPy，不改变任何预测或真值。
    """
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _process_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_process_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_process_safe(item) for item in value)
    return value


def _block_bootstrap_ci(values, *, seed: int = 20261008, draws: int = 2000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    block = 7
    starts = np.arange(len(array))
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pieces = []
        while sum(len(piece) for piece in pieces) < len(array):
            start = int(rng.choice(starts))
            pieces.append(array[(start + np.arange(block)) % len(array)])
        means[draw] = np.mean(np.concatenate(pieces)[:len(array)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _aggregate(nodes: dict[str, dict]) -> dict:
    ordered_nodes = sorted(nodes)
    dates = sorted(nodes[ordered_nodes[0]]["daily_net_revenue"])
    if any(sorted(nodes[node]["daily_net_revenue"]) != dates for node in ordered_nodes):
        raise ValueError("多节点日收益日期不一致")
    matrix = np.asarray([
        [nodes[node]["daily_net_revenue"][day] for day in dates]
        for node in ordered_nodes
    ])
    macro_daily = matrix.mean(axis=0)
    node_means = matrix.mean(axis=1)
    ordered_daily = np.sort(macro_daily)
    total_revenue = float(np.sum(macro_daily))
    top_5_count = min(5, len(ordered_daily))
    top_10_count = min(10, len(ordered_daily))
    component_names = (
        "da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty",
    )
    macro_components_per_day = {
        name: float(np.mean([
            nodes[node]["revenue_components"][name] / nodes[node]["days"]
            for node in ordered_nodes
        ]))
        for name in component_names
    }
    reconstructed = (
        macro_components_per_day["da_leg"]
        + macro_components_per_day["rt_deviation_leg"]
        - macro_components_per_day["degradation_cost"]
        - macro_components_per_day["deviation_penalty"]
    )
    return {
        "node_count": len(nodes),
        "days": len(dates),
        "macro_mean_daily_revenue": float(macro_daily.mean()),
        "macro_median_daily_revenue": float(np.median(macro_daily)),
        "macro_p10_daily_revenue": float(np.quantile(macro_daily, 0.10)),
        "macro_p90_daily_revenue": float(np.quantile(macro_daily, 0.90)),
        "macro_min_daily_revenue": float(np.min(macro_daily)),
        "macro_max_daily_revenue": float(np.max(macro_daily)),
        "top_5_days_total_revenue_share": (
            float(np.sum(ordered_daily[-top_5_count:]) / total_revenue)
            if abs(total_revenue) > 1e-12 else None
        ),
        "top_10_days_total_revenue_share": (
            float(np.sum(ordered_daily[-top_10_count:]) / total_revenue)
            if abs(total_revenue) > 1e-12 else None
        ),
        "macro_mean_daily_revenue_without_top_5_days": (
            float(np.mean(ordered_daily[:-top_5_count]))
            if len(ordered_daily) > top_5_count else None
        ),
        "macro_mean_daily_actual_throughput_mwh": float(np.mean([
            nodes[node]["actual_throughput_mwh"] / nodes[node]["days"]
            for node in ordered_nodes
        ])),
        "macro_mean_daily_revenue_components": macro_components_per_day,
        "macro_mean_daily_revenue_reconstructed": float(reconstructed),
        "macro_mean_daily_revenue_7day_block_bootstrap_95ci": _block_bootstrap_ci(
            macro_daily
        ),
        "macro_positive_day_rate": float(np.mean(macro_daily > 0.0)),
        "node_positive_mean_revenue_rate": float(np.mean(node_means > 0.0)),
        "total_soc_violation_count": int(sum(
            nodes[node]["soc_violation_count"] for node in ordered_nodes
        )),
        "worst_node_mean_daily_revenue": float(node_means.min()),
        "best_node_mean_daily_revenue": float(node_means.max()),
        "macro_daily_net_revenue": dict(zip(dates, macro_daily.tolist())),
        "nodes": nodes,
    }


def _node_baseline_archives(
    reference: dict,
    nodes: np.ndarray,
    baseline: str,
    template_family: str = "f0",
    phase: str = "inner",
) -> dict:
    """把逐节点基线预测装入统一年度档案；真值必须逐元素吻合。"""
    result = {}
    for task in ("da", "rt_at_da", "rt"):
        template = reference[(template_family, task)]
        stored_model = BASELINE_TASK_MODELS.get(baseline, {}).get(task, baseline)
        predictions = []
        for node in nodes:
            path = (
                ROOT / "data" / "results" / "global_baselines_v1"
                / f"{task}_{phase}" / f"{stored_model}_{node}_seed0.npz"
            )
            with np.load(path, allow_pickle=False) as archive:
                prediction = archive["prediction"].astype(np.float32)
                target = archive["target"].astype(np.float32)
            expected_target = (
                template["target_da"] if task == "da" else template["target_rt"]
            )[len(predictions)]
            if task == "rt" and target.ndim == 2:
                target = target.reshape(expected_target.shape)
                prediction = prediction.reshape(expected_target.shape)
            if target.shape != expected_target.shape or not np.allclose(
                target, expected_target, rtol=0.0, atol=1e-5
            ):
                raise ValueError(
                    f"{baseline}真值与统一档案不一致: {task}/{node}"
                )
            predictions.append(prediction)
        item = dict(template)
        item["prediction"] = np.stack(predictions)
        item["metadata"] = {
            "family": baseline,
            "task": task,
            "stored_model": stored_model,
        }
        result[(baseline, task)] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument("--prediction-dir", default="data/results/global_f0_f1_predictions_v1")
    parser.add_argument("--families", default="f0,f1")
    parser.add_argument("--phase", choices=("inner", "outer"), default="inner")
    parser.add_argument("--selection-allowed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--nodes", default="all")
    parser.add_argument("--triplets", default="all")
    parser.add_argument("--include-seasonal", action="store_true")
    parser.add_argument(
        "--include-baselines",
        default="",
        help="可选：逗号分隔的完整逐节点基线，如seasonal,xgboost；仅评价各自纯三元组",
    )
    parser.add_argument(
        "--cross-baselines",
        default="",
        help="可选：把基线当作任务家族，与families做跨任务组合诊断",
    )
    parser.add_argument(
        "--status",
        default="inner_validation_candidate_not_final",
        help="写入结果的阶段标签；不得把内部验证结果误标为最终测试",
    )
    parser.add_argument("--output", default="data/results/global_continuous_mpc_backtest_v1.json")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    directory = ROOT / args.prediction_dir
    families = tuple(
        value.strip().lower() for value in args.families.split(",") if value.strip()
    )
    if not families or len(set(families)) != len(families):
        raise ValueError("families必须非空且不能重复")
    archives = {
        (family, task): _load_archive(directory, family, task, args.phase)
        for family in families for task in ("da", "rt_at_da", "rt")
    }
    reference_nodes = archives[(families[0], "da")]["nodes"]
    baselines = [
        value.strip().lower()
        for value in args.include_baselines.split(",")
        if value.strip()
    ]
    if args.include_seasonal and "seasonal" not in baselines:
        baselines.append("seasonal")
    if len(set(baselines)) != len(baselines):
        raise ValueError("include-baselines不能重复")
    cross_baselines = tuple(
        value.strip().lower()
        for value in args.cross_baselines.split(",") if value.strip()
    )
    if len(set(cross_baselines)) != len(cross_baselines):
        raise ValueError("cross-baselines不能重复")
    if set(cross_baselines) & set(families):
        raise ValueError("cross-baselines不能与families重名")
    for baseline in dict.fromkeys([*baselines, *cross_baselines]):
        archives.update(
            _node_baseline_archives(
                archives, reference_nodes, baseline, families[0], args.phase
            )
        )

    if args.nodes == "all":
        selected_nodes = reference_nodes.tolist()
    else:
        requested = [value.strip() for value in args.nodes.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(reference_nodes))
        if unknown:
            raise ValueError(f"未知节点: {unknown}")
        selected_nodes = requested

    combinations = list(enumerate_fusion_triplets((*families, *cross_baselines)))
    combinations.extend(
        {"da": baseline, "rt_at_da": baseline, "rt": baseline}
        for baseline in baselines
    )
    if args.triplets != "all":
        requested_labels = {value.strip() for value in args.triplets.split(",") if value.strip()}
        combinations = [item for item in combinations if _label(item) in requested_labels]
        missing = requested_labels - {_label(item) for item in combinations}
        if missing:
            raise ValueError(f"未知三元组: {sorted(missing)}")

    node_to_index = {str(node): index for index, node in enumerate(reference_nodes)}
    jobs = []
    for combination in combinations:
        label = _label(combination)
        for node in selected_nodes:
            index = node_to_index[node]
            node_input = align_node_task_predictions(
                _node_record(archives[(combination["da"], "da")], index),
                _node_record(archives[(combination["rt_at_da"], "rt_at_da")], index),
                _node_record(archives[(combination["rt"], "rt")], index),
            )
            jobs.append((label, node, _process_safe(node_input)))

    started = perf_counter()
    completed: dict[str, dict[str, dict]] = {_label(item): {} for item in combinations}
    config_path = str((ROOT / args.config).resolve())
    if args.workers == 1:
        iterator = (
            _worker(config_path, label, node, node_input)
            for label, node, node_input in jobs
        )
        for count, (label, node, report) in enumerate(iterator, start=1):
            completed[label][node] = report
            print(json.dumps({
                "event": "continuous_node_complete",
                "completed": count,
                "total": len(jobs),
                "triplet": label,
                "node": node,
                "mean_daily_revenue": report["mean_daily_revenue"],
            }, ensure_ascii=False), flush=True)
    else:
        # Windows下过多spawn进程会重复加载PyTorch/CUDA DLL；默认由调用者将
        # 进程数限制在本机页面文件可承受的范围。线程模式保留作低内存回退。
        executor_class = (
            ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        )
        with executor_class(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_worker, config_path, label, node, node_input): (label, node)
                for label, node, node_input in jobs
            }
            for count, future in enumerate(as_completed(futures), start=1):
                label, node, report = future.result()
                completed[label][node] = report
                print(json.dumps({
                    "event": "continuous_node_complete",
                    "completed": count,
                    "total": len(jobs),
                    "triplet": label,
                    "node": node,
                    "mean_daily_revenue": report["mean_daily_revenue"],
                }, ensure_ascii=False), flush=True)

    systems = {label: _aggregate(nodes) for label, nodes in completed.items()}
    ranking = sorted(
        ({"rank": 0, "triplet": label,
          "macro_mean_daily_revenue": value["macro_mean_daily_revenue"]}
         for label, value in systems.items()),
        key=lambda item: item["macro_mean_daily_revenue"], reverse=True,
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    comparisons = {}
    seasonal_label = "da_seasonal__rt_at_da_seasonal__rt_seasonal"
    if seasonal_label in systems:
        baseline = systems[seasonal_label]["macro_daily_net_revenue"]
        for label, system in systems.items():
            if label == seasonal_label:
                continue
            difference = np.asarray([
                system["macro_daily_net_revenue"][day] - baseline[day]
                for day in sorted(baseline)
            ])
            comparisons[label] = {
                "mean_daily_revenue_difference_vs_seasonal": float(difference.mean()),
                "7day_block_bootstrap_95ci": _block_bootstrap_ci(
                    difference, seed=20261009
                ),
            }

    paired_vs_baselines = {}
    for baseline in dict.fromkeys([*baselines, *cross_baselines]):
        baseline_label = _label({
            "da": baseline, "rt_at_da": baseline, "rt": baseline,
        })
        if baseline_label not in systems:
            continue
        baseline_daily = systems[baseline_label]["macro_daily_net_revenue"]
        paired_vs_baselines[baseline] = {}
        for label, system in systems.items():
            if label == baseline_label:
                continue
            difference = np.asarray([
                system["macro_daily_net_revenue"][day] - baseline_daily[day]
                for day in sorted(baseline_daily)
            ])
            paired_vs_baselines[baseline][label] = {
                "mean_daily_revenue_difference": float(difference.mean()),
                "7day_block_bootstrap_95ci": _block_bootstrap_ci(
                    difference, seed=20261009
                ),
            }

    report = {
        "experiment": "global_predicted_continuous_da_lp_rt_mpc_v1",
        "status": args.status,
        "selection_allowed": args.selection_allowed,
        "experiment_contract": FORMAL_EXPERIMENT_CONTRACT.to_dict(),
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "split": (
            "2023 inner validation only"
            if args.phase == "inner"
            else "2024 outer validation only"
        ),
        "nodes": selected_nodes,
        "policy": {
            "da": (
                "daily predicted DA-minus-RT-at-DA spread direction followed "
                "by joint penalty-aware continuous LP; plan locked"
            ),
            "rt": "hourly predicted-price fixed-DA continuous LP; four-hour horizon; execute first action",
            "truth_access": "realized DA/RT only after action for settlement",
            "deviation_penalty_enabled": True,
            "degradation_usd_per_actual_throughput_mwh": cfg.bess_kappa,
            "terminal_soc_monetized_in_settlement": False,
            "mpc_terminal_opportunity_value": "last visible predicted RT price; decision-only anti-myopia term",
        },
        "systems": systems,
        "ranking": ranking,
        "paired_vs_seasonal": comparisons,
        "paired_vs_baselines": paired_vs_baselines,
        "runtime_seconds": perf_counter() - started,
        "workers": args.workers,
        "executor": args.executor,
        "included_pure_baselines": baselines,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "continuous_evaluation_saved",
        "path": output.relative_to(ROOT).as_posix(),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
