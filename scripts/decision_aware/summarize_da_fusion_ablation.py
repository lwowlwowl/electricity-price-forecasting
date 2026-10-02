#!/usr/bin/env python
"""汇总DA或RT-at-DA融合消融，并只按验证期完整双结算收益选结构。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (0, 1, 2)


def _bootstrap_difference(values: np.ndarray, samples: int = 5000,
                          block: int = 7, seed: int = 20260929) -> list[float]:
    rng = np.random.default_rng(seed)
    block = min(block, len(values))
    blocks = math.ceil(len(values) / block)
    last_start = len(values) - block + 1
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        starts = rng.integers(0, last_start, size=blocks)
        sample = np.concatenate([values[start:start + block] for start in starts])
        means[index] = sample[:len(values)].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def _best_seed(runs: dict, mode: str) -> int:
    """只用验证期完整双结算收益，在胜出结构内部选择部署seed。"""
    return max(
        SEEDS,
        key=lambda seed: runs[(mode, seed)]["validation"]["full_dual"][
            "mean_daily_revenue"
        ],
    )


def _normalize_checkpoint(value: str) -> str:
    return value.replace("\\", "/")


def _select_frozen_upstream(result_dir: Path, task: str, prefix: str,
                            panel: list[str]) -> dict:
    """只在滚动RT实验真正冻结过的上游checkpoint中选择一个。"""
    if not panel:
        raise ValueError(f"滚动RT结果没有冻结的{task} checkpoint面板")

    candidates = []
    for checkpoint in panel:
        normalized = _normalize_checkpoint(checkpoint)
        parent_name = Path(normalized).parent.name
        if "_seed" not in parent_name:
            raise ValueError(f"无法从checkpoint识别结构和seed: {checkpoint}")
        mode_code, seed_text = parent_name.rsplit("_seed", 1)
        seed = int(seed_text)
        run_path = result_dir / f"{prefix}_{mode_code}_seed{seed}.json"
        report = json.loads(run_path.read_text(encoding="utf-8"))
        if _normalize_checkpoint(report["checkpoint"]) != normalized:
            raise ValueError(f"冻结checkpoint与结果文件不匹配: {checkpoint}")
        candidates.append(report)

    selected = max(
        candidates,
        key=lambda report: report["validation"]["full_dual"][
            "mean_daily_revenue"
        ],
    )
    return {
        "fusion_mode": selected["fusion_mode"],
        "seed": int(selected["seed"]),
        "checkpoint": selected["checkpoint"],
        "selection_basis": (
            "highest stage validation revenue among upstream checkpoints "
            "actually frozen in the rolling-RT experiment"
        ),
    }


def _write_selected_system(result_dir: Path, rt_runs: dict,
                           rt_winner: str) -> Path:
    """记录不做笛卡尔积回选的单checkpoint组合及其已暴露期结果。"""
    rt_seed = _best_seed(rt_runs, rt_winner)
    rt_report = rt_runs[(rt_winner, rt_seed)]
    frozen = rt_report["frozen_system"]
    da_panel = list(frozen.get("da_checkpoints") or [])
    if not da_panel and frozen.get("da_checkpoint"):
        da_panel = [frozen["da_checkpoint"]]
    rt_at_da_panel = list(frozen.get("rt_at_da_checkpoints") or [])
    if not rt_at_da_panel and frozen.get("rt_at_da_checkpoint"):
        rt_at_da_panel = [frozen["rt_at_da_checkpoint"]]

    selections = {
        "da": _select_frozen_upstream(
            result_dir, "DA", "da_fusion", da_panel
        ),
        "rt_at_da": _select_frozen_upstream(
            result_dir, "RT-at-DA", "rt_at_da_fusion", rt_at_da_panel
        ),
    }
    selections["rt"] = {
        "fusion_mode": rt_winner,
        "seed": rt_seed,
        "checkpoint": rt_report["checkpoint"],
        "selection_basis": "highest stage validation revenue within selected mode",
    }

    da_checkpoint = _normalize_checkpoint(selections["da"]["checkpoint"])
    rt_da_checkpoint = _normalize_checkpoint(
        selections["rt_at_da"]["checkpoint"]
    )
    reference_key = f"DA={da_checkpoint}|RT-at-DA={rt_da_checkpoint}"
    validation_refs = rt_report["validation"]["full_dual"][
        "per_reference_mean_daily_revenue"
    ]
    report_refs = rt_report["test_report_only"]["full_dual"][
        "per_reference_mean_daily_revenue"
    ]
    if reference_key not in validation_refs or reference_key not in report_refs:
        raise KeyError(f"滚动RT结果缺少最终参考组合: {reference_key}")

    selected = {
        "experiment": "selected fusion structures and deployment checkpoints",
        "selection_rule": (
            "select structure by three-seed mean validation revenue, then select "
            "one checkpoint by validation revenue within that structure; no joint grid"
        ),
        "models": selections,
        "coordination_mode": "rt_only",
        "validation_full_dual_mean_daily_revenue": float(
            validation_refs[reference_key]
        ),
        "report_only_full_dual_mean_daily_revenue": float(
            report_refs[reference_key]
        ),
        "report_only_warning": (
            "2026-01 through 2026-06 was exposed by earlier experiments and is not "
            "a fresh final test"
        ),
        "reference_key": reference_key,
    }
    output = result_dir / "fusion_selected_system_summary.json"
    output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("da", "rt_at_da", "rt"), default="da"
    )
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    modes = {
        "da": ("F0", "F1", "F2"),
        "rt_at_da": ("F0", "F1", "F1b", "F2"),
        "rt": ("F0", "F1", "F1b"),
    }[args.task]
    prefix = {
        "da": "da_fusion",
        "rt_at_da": "rt_at_da_fusion",
        "rt": "rt_fusion",
    }[args.task]
    experiment_label = {
        "da": "DA F0/F1/F2 fusion ablation; three seeds",
        "rt_at_da": "RT-at-DA F0/F1/F1b/F2 confirmation ablation; three seeds",
        "rt": "Rolling RT F0/F1/F1b confirmation ablation; three seeds",
    }[args.task]

    result_dir = Path(args.results_dir)
    runs = {}
    for mode in modes:
        for seed in SEEDS:
            path = result_dir / f"{prefix}_{mode.lower()}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["fusion_mode"] != mode or int(report["seed"]) != seed:
                raise ValueError(f"结果身份不匹配: {path}")
            if report.get("task", "da") != args.task:
                raise ValueError(f"结果任务不匹配: {path}")
            if report["loss"] != "pure Huber only; alpha=1 beta=0":
                raise ValueError(f"{path}不是纯Huber消融")
            runs[(mode, seed)] = report

    split_contracts = {
        json.dumps(report["split"], sort_keys=True, ensure_ascii=False)
        for report in runs.values()
    }
    # 兼容脚本扩展参考面板字段之前生成的旧结果：新增但为空的列表不改变
    # 实际冻结系统，不能因此把同一实验合同误判为不同。
    frozen_contracts = {
        json.dumps(
            {
                key: value
                for key, value in report["frozen_system"].items()
                if value not in (None, "", [])
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for report in runs.values()
    }
    effective_batches = {report["effective_batch_size"] for report in runs.values()}
    if len(split_contracts) != 1 or len(frozen_contracts) != 1:
        raise ValueError("不同run的数据切分或冻结RT合同不一致")
    if effective_batches != {32}:
        raise ValueError(f"有效batch size不一致: {effective_batches}")

    parameter_counts = {
        mode: runs[(mode, 0)]["parameter_count"] for mode in modes
    }
    parameter_ratio = max(parameter_counts.values()) / min(parameter_counts.values())
    if parameter_ratio >= 1.10:
        raise ValueError(f"参数预算差距超过10%: {parameter_counts}")

    mode_summaries = {}
    for mode in modes:
        reports = [runs[(mode, seed)] for seed in SEEDS]
        mode_summaries[mode] = {
            "validation_full_dual_revenue": _stats([
                report["validation"]["full_dual"]["mean_daily_revenue"]
                for report in reports
            ]),
            "validation_mae": _stats([
                report["validation"]["forecast"]["mae"] for report in reports
            ]),
            "validation_lower_tail_10pct": _stats([
                report["validation"]["full_dual"]["lower_tail_10pct_mean"]
                for report in reports
            ]),
            "validation_max_drawdown": _stats([
                report["validation"]["full_dual"]["max_drawdown"]
                for report in reports
            ]),
            "test_report_only_revenue": _stats([
                report["test_report_only"]["full_dual"]["mean_daily_revenue"]
                for report in reports
            ]),
            "parameter_count": parameter_counts[mode],
            "training_seconds": _stats([
                report["compute"]["seconds"] for report in reports
            ]),
            "peak_cuda_memory_mb": _stats([
                report["compute"]["peak_cuda_memory_mb"] for report in reports
            ]),
        }

    winner = max(
        modes,
        key=lambda mode: mode_summaries[mode]["validation_full_dual_revenue"]["mean"],
    )
    paired = {}
    winner_daily = []
    for seed in SEEDS:
        winner_daily.append(
            runs[(winner, seed)]["validation"]["full_dual"]["daily_net_revenue"]
        )
    for other in modes:
        if other == winner:
            continue
        other_daily = [
            runs[(other, seed)]["validation"]["full_dual"]["daily_net_revenue"]
            for seed in SEEDS
        ]
        dates = sorted(set.intersection(*[
            set(mapping) for mapping in winner_daily + other_daily
        ]))
        winner_values = np.mean([
            [mapping[date] for date in dates] for mapping in winner_daily
        ], axis=0)
        other_values = np.mean([
            [mapping[date] for date in dates] for mapping in other_daily
        ], axis=0)
        difference = winner_values - other_values
        paired[f"{winner}_minus_{other}"] = {
            "mean_daily_difference": float(difference.mean()),
            "weekly_block_bootstrap_95ci": _bootstrap_difference(difference),
            "paired_days": len(dates),
        }

    summary = {
        "experiment": experiment_label,
        "task": args.task,
        "selection_rule": "highest mean frozen full-dual validation revenue across seeds",
        "test_role": "report-only and never used to choose the winner",
        "fairness": {
            "loss": "pure Huber for every run",
            "effective_batch_size": 32,
            "parameter_counts": parameter_counts,
            "max_to_min_parameter_ratio": parameter_ratio,
            "same_split": True,
            "same_frozen_rt_system": True,
        },
        "modes": mode_summaries,
        "selected_by_validation": winner,
        "paired_validation_differences": paired,
        "run_files": {
            f"{mode}_seed{seed}": f"data/results/{prefix}_{mode.lower()}_seed{seed}.json"
            for mode in modes for seed in SEEDS
        },
    }
    default_outputs = {
        "da": "data/results/da_fusion_ablation_summary.json",
        "rt_at_da": "data/results/rt_at_da_fusion_ablation_summary.json",
        "rt": "data/results/rt_fusion_ablation_summary.json",
    }
    output = Path(args.output or default_outputs[args.task])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_output = None
    if args.task == "rt":
        selected_output = _write_selected_system(result_dir, runs, winner)
    print(json.dumps({
        "selected_by_validation": winner,
        "validation_revenues": {
            mode: mode_summaries[mode]["validation_full_dual_revenue"]
            for mode in modes
        },
        "output": str(output),
        "selected_system_output": (
            str(selected_output) if selected_output is not None else None
        ),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
