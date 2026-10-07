#!/usr/bin/env python
"""汇总偏差罚金版冻结验证结果与一次性report-only披露。

本脚本不训练、不推理，也不改变模型选择；它只读取已经生成的正式JSON，
检查关键合同后生成一份机器可读的最终闭环摘要。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式结果: {relative_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "members": values,
        "seed_count": len(values),
    }


def _assert_final_contract(validation: dict, report: dict) -> None:
    contract = validation["contract"]
    if validation.get("evaluation_split") != "validation":
        raise ValueError("冻结选模表不是validation")
    if validation.get("split_role") != "model_selection_and_validation":
        raise ValueError("冻结选模表的切分角色不正确")
    if contract["evaluation_days"] != 182:
        raise ValueError("冻结验证表不是182日")
    if not contract["deviation_penalty_enabled"]:
        raise ValueError("冻结验证表没有启用偏差罚金")
    if float(contract["bess"]["degradation_usd_per_mwh"]) != 5.7:
        raise ValueError("冻结验证表的κ不是5.7")
    if report.get("evaluation_split") != "report_only":
        raise ValueError("最终基线表不是report_only")
    if not report.get("report_only_explicitly_confirmed"):
        raise ValueError("最终基线表未显式确认report-only披露")
    if report["contract"]["evaluation_days"] != 150:
        raise ValueError("最终report-only表不是150日")
    if not report["contract"]["deviation_penalty_enabled"]:
        raise ValueError("最终report-only表没有启用偏差罚金")
    if float(report["contract"]["bess"]["degradation_usd_per_mwh"]) != 5.7:
        raise ValueError("最终report-only表的κ不是5.7")


def _safe_row(label: str, validation: dict, report: dict) -> dict:
    if label in ("xgboost", "gru", "transformer_v2"):
        val = validation["seed_aggregates"][label]["mean_daily_revenue"]
        rep = report["seed_aggregates"][label]["mean_daily_revenue"]
        return {
            "system": label,
            "policy_family": "safe_da_only_follow_da",
            "validation_mean_daily_revenue": val["mean"],
            "validation_seed_sample_std": val["sample_std"],
            "report_only_mean_daily_revenue": rep["mean"],
            "report_only_seed_sample_std": rep["sample_std"],
        }
    val = validation["systems"][label]
    rep = report["systems"][label]
    return {
        "system": label,
        "policy_family": "safe_da_only_follow_da",
        "validation_mean_daily_revenue": val["mean_daily_revenue"],
        "validation_seed_sample_std": 0.0,
        "report_only_mean_daily_revenue": rep["mean_daily_revenue"],
        "report_only_seed_sample_std": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "data/results/penalty_experiment_final_closure_kappa57.json"
        ),
    )
    args = parser.parse_args()

    validation_path = (
        "data/results/joint_system_baseline_comparison_validation_"
        "penalty_kappa57_final.json"
    )
    report_path = (
        "data/results/joint_system_baseline_comparison_report_only_"
        "penalty_kappa57_final.json"
    )
    validation = _load(validation_path)
    report = _load(report_path)
    _assert_final_contract(validation, report)

    rows = [
        _safe_row(label, validation, report)
        for label in (
            "seasonal", "xgboost", "gru", "transformer_huber",
            "transformer_v2",
        )
    ]

    active_train_paths = [
        f"data/results/joint_decision_aware_v3_penalty_active_kappa57_seed{seed}.json"
        for seed in range(3)
    ]
    active_report_paths = [
        "data/results/joint_decision_aware_v3_penalty_active_kappa57_"
        f"seed{seed}_report_only_final.json"
        for seed in range(3)
    ]
    active_train = [_load(path) for path in active_train_paths]
    active_report = [_load(path) for path in active_report_paths]
    for seed, item in enumerate(active_report):
        if item.get("split") != "report_only" or item.get("days") != 150:
            raise ValueError(f"active seed{seed}不是完整report-only评估")
        if not item.get("report_only_explicitly_confirmed"):
            raise ValueError(f"active seed{seed}未确认最终披露")
        if float(item["contract"]["bess_kappa"]) != 5.7:
            raise ValueError(f"active seed{seed}的κ不是5.7")
        if not item["contract"]["use_deviation_penalty"]:
            raise ValueError(f"active seed{seed}没有启用偏差罚金")

    active_validation_values = [
        float(item["best_validation"]["full_dual"]["mean_daily_revenue"])
        for item in active_train
    ]
    active_report_values = [
        float(item["metrics"]["mean_daily_revenue"])
        for item in active_report
    ]
    active_validation = _mean_std(active_validation_values)
    active_report_summary = _mean_std(active_report_values)
    rows.append({
        "system": "transformer_v3_penalty_active_joint",
        "policy_family": "active_spread_plan_track_topk",
        "validation_mean_daily_revenue": active_validation["mean"],
        "validation_seed_sample_std": active_validation["sample_std"],
        "report_only_mean_daily_revenue": active_report_summary["mean"],
        "report_only_seed_sample_std": active_report_summary["sample_std"],
    })

    active_epoch0 = _load(
        "data/results/joint_system_active_epoch0_report_only_penalty_"
        "kappa57_final.json"
    )
    active_epoch0_validation = float(
        active_train[0]["baseline_validation"]["full_dual"]
        ["mean_daily_revenue"]
    )
    active_epoch0_report = float(
        active_epoch0["systems"]["transformer_huber"]["mean_daily_revenue"]
    )
    rows.append({
        "system": "transformer_v3_penalty_active_epoch0",
        "policy_family": "active_spread_plan_track_topk",
        "validation_mean_daily_revenue": active_epoch0_validation,
        "validation_seed_sample_std": 0.0,
        "report_only_mean_daily_revenue": active_epoch0_report,
        "report_only_seed_sample_std": 0.0,
    })

    validation_winner = max(
        rows, key=lambda row: row["validation_mean_daily_revenue"]
    )
    report_best = max(
        rows, key=lambda row: row["report_only_mean_daily_revenue"]
    )
    payload = {
        "experiment": "penalty_experiment_final_closure_kappa57",
        "status": "closed",
        "model_selection_frozen_before_report_only": True,
        "report_only_used_for_training_tuning_or_reselection": False,
        "primary_selection_split": "validation_2025-07-01_to_2025-12-30_182_days",
        "final_disclosure_split": "report_only_2026-01-01_to_2026-05-31_150_days",
        "contract": {
            "market": "ERCOT",
            "node": "LZ_LCRA",
            "degradation_usd_per_mwh": 5.7,
            "deviation_penalty_enabled": True,
            "deviation_penalty_formula": (
                "2*abs(RT_price)*max(abs(u_RT_actual-u_DA)-"
                "0.03*abs(u_DA),0)"
            ),
            "continuous_soc": True,
            "terminal_soc_monetized": False,
        },
        "comparison_rows": rows,
        "selection_frozen_on_validation": {
            "winner": validation_winner["system"],
            "mean_daily_revenue": validation_winner[
                "validation_mean_daily_revenue"
            ],
            "note": "report-only不会改变这项预先冻结的选择",
        },
        "report_only_disclosure": {
            "numerical_best": report_best["system"],
            "mean_daily_revenue": report_best[
                "report_only_mean_daily_revenue"
            ],
            "active_joint_three_seed": active_report_summary,
            "active_epoch0_mean_daily_revenue": active_epoch0_report,
            "active_joint_minus_active_epoch0": (
                active_report_summary["mean"] - active_epoch0_report
            ),
        },
        "conclusion": (
            "偏差罚金版已完成与旧版相同的冻结验证、三seed和最终"
            "report-only闭环。安全策略验证冠军按预先规则冻结；active联合"
            "微调未形成样本外改善，不能据report-only回头调参。"
        ),
        "artifacts": {
            "safe_validation": validation_path,
            "safe_report_only": report_path,
            "active_training": active_train_paths,
            "active_report_only": active_report_paths,
            "active_epoch0_report_only": (
                "data/results/joint_system_active_epoch0_report_only_"
                "penalty_kappa57_final.json"
            ),
        },
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({
        "event": "penalty_experiment_closed",
        "output": str(output),
        "validation_winner": validation_winner["system"],
        "report_only_numerical_best": report_best["system"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
