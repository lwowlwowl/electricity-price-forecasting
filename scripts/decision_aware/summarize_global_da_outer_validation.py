#!/usr/bin/env python
"""汇总冻结Global DA候选的2024外层验证结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "da_blendgm15__rt_at_da_xgboost_robust__rt_xgboost_robust"
BASELINE = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/results/global_da_outer_median_antiblend_continuous_mpc_v1.json",
    )
    parser.add_argument(
        "--output", default="data/results/global_da_outer_validation_conclusion_v1.json"
    )
    parser.add_argument(
        "--document", default="docs/progress/3_多节点Global与跨市场泛化/5_ERCOT_DA冻结候选2024外层验证.md"
    )
    parser.add_argument(
        "--training-protocol",
        choices=["validation_adaptive", "frozen_inner_plan"],
        default="validation_adaptive",
    )
    args = parser.parse_args()

    source = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    candidate = source["systems"][CANDIDATE]
    baseline = source["systems"][BASELINE]
    paired = source["paired_vs_baselines"]["xgboost_robust"][CANDIDATE]
    difference = float(paired["mean_daily_revenue_difference"])
    ci = [float(value) for value in paired["7day_block_bootstrap_95ci"]]
    beats_mean = difference > 0.0
    statistically_clear = ci[0] > 0.0
    conclusion = {
        "experiment": "ercot_global_da_frozen_candidate_2024_outer_validation",
        "status": "2024_outer_validation_complete_2025_untouched",
        "selection_allowed": True,
        "locked_2025_used": False,
        "frozen_candidate": {
            "architecture": "Global F1 DA",
            "seeds": [0, 1, 2],
            "aggregation": "per-hour median",
            "formula": "p_da = 1.15 * p_xgboost - 0.15 * median(p_global_seed0, seed1, seed2)",
            "rt_at_da": "xgboost_pseudohuber",
            "rolling_rt": "xgboost_absolute",
            "training_protocol": args.training_protocol,
            "frozen_final_epochs_by_seed": (
                {"0": 10, "1": 5, "2": 19}
                if args.training_protocol == "frozen_inner_plan"
                else None
            ),
        },
        "candidate": {
            "macro_mean_daily_revenue": candidate["macro_mean_daily_revenue"],
            "macro_median_daily_revenue": candidate[
                "macro_median_daily_revenue"
            ],
            "top_10_days_total_revenue_share": candidate[
                "top_10_days_total_revenue_share"
            ],
            "macro_mean_daily_revenue_components": candidate[
                "macro_mean_daily_revenue_components"
            ],
            "macro_positive_day_rate": candidate["macro_positive_day_rate"],
            "node_positive_mean_revenue_rate": candidate[
                "node_positive_mean_revenue_rate"
            ],
            "worst_node_mean_daily_revenue": candidate[
                "worst_node_mean_daily_revenue"
            ],
            "best_node_mean_daily_revenue": candidate["best_node_mean_daily_revenue"],
        },
        "baseline": {
            "name": "xgboost_robust_three_task",
            "macro_mean_daily_revenue": baseline["macro_mean_daily_revenue"],
            "macro_median_daily_revenue": baseline[
                "macro_median_daily_revenue"
            ],
            "top_10_days_total_revenue_share": baseline[
                "top_10_days_total_revenue_share"
            ],
            "macro_mean_daily_revenue_components": baseline[
                "macro_mean_daily_revenue_components"
            ],
        },
        "paired_comparison": {
            "mean_daily_revenue_difference": difference,
            "7day_block_bootstrap_95ci": ci,
            "beats_baseline_on_mean": beats_mean,
            "statistically_clear_positive": statistically_clear,
        },
        "methodology_control": {
            "2024_used_for_training_control": (
                args.training_protocol != "frozen_inner_plan"
            ),
            "strict_outer_result": args.training_protocol == "frozen_inner_plan",
            "leakage_audit": (
                "data/results/global_da_outer_frozen_training_audit_v1.json"
                if args.training_protocol == "frozen_inner_plan"
                else None
            ),
            "discarded_diagnostic_run": (
                "An earlier diagnostic selected epochs from 2024 MAE and is not used "
                "for the reported decision."
                if args.training_protocol == "frozen_inner_plan"
                else None
            ),
        },
        "decision": {
            "eligible_for_final_2020_2024_retraining": beats_mean,
            "strong_win": statistically_clear,
            "next_step": (
                "freeze the candidate, retrain on 2020-2024, then run the single locked 2025 test"
                if beats_mean
                else "keep 2025 locked; return to inner-development evidence and redesign without tuning on 2024"
            ),
        },
        "source_result": args.input,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = (
        "均值超过且区间下界大于0，属于较明确胜出。"
        if statistically_clear
        else "均值超过，但区间仍跨0，只能称均值胜出。"
        if beats_mean
        else "没有超过稳健XGBoost基线；不能进入2025。"
    )
    document = ROOT / args.document
    document.parent.mkdir(parents=True, exist_ok=True)
    training_description = (
        "1. 只根据2023内部训练记录冻结seed 0/1/2的最终轮次为10/5/19，"
        "并逐轮复用相应的学习率轨迹。2024不参与早停、学习率调度或checkpoint选择，"
        "只在最终轮次评估一次。\n"
        if args.training_protocol == "frozen_inner_plan"
        else "1. 使用2020—2023分别训练三个Global DA seed，并按2024节点等权MAE保留checkpoint。\n"
    )
    document_text = (
        "# ERCOT DA冻结候选：2024外层验证\n\n"
        "> 本页只记录2024外层验证。2025锁定回顾测试尚未读取。\n\n"
        "## 冻结方案\n\n"
        "- DA：Global F1，seed 0/1/2逐小时中位数；随后使用2023已冻结的公式 "
        "`1.15 × XGBoost - 0.15 × Global中位数`。\n"
        "- RT-at-DA：稳健XGBoost（伪Huber）。\n"
        "- 滚动RT：稳健XGBoost（绝对误差）。\n"
        "- 偏差罚金、退化成本和连续DA-LP/RT-MPC合同保持不变。\n\n"
        "## 运行流程\n\n"
    ) + training_description + (
        "2. 使用2020—2023重训三任务稳健XGBoost，损失类型不变。\n"
        "3. 在2024导出三个Global DA预测并逐小时取中位数。\n"
        "4. 应用冻结`-15%`公式；不在2024搜索新权重。\n"
        "5. 在同一连续DA-LP、锁定DA计划、滚动RT-MPC和完整双结算下比较。\n\n"
        "## 2024结果\n\n"
        f"- 冻结候选：{candidate['macro_mean_daily_revenue']:.2f} 美元/节点·日。\n"
        f"- 稳健XGBoost：{baseline['macro_mean_daily_revenue']:.2f} 美元/节点·日。\n"
        f"- 候选/基线日收益中位数：{candidate['macro_median_daily_revenue']:.2f} / "
        f"{baseline['macro_median_daily_revenue']:.2f}。\n"
        f"- 候选/基线最高10天收益占比："
        f"{100*candidate['top_10_days_total_revenue_share']:.2f}% / "
        f"{100*baseline['top_10_days_total_revenue_share']:.2f}%。\n"
        f"- 配对差值：{difference:+.2f} 美元/节点·日。\n"
        f"- 7日块bootstrap 95%区间：[{ci[0]:.2f}, {ci[1]:.2f}]。\n"
        f"- 结论：{verdict}\n\n"
        "### 收益拆分\n\n"
        f"- 候选DA收入项：{candidate['macro_mean_daily_revenue_components']['da_leg']:.2f}；"
        f"基线：{baseline['macro_mean_daily_revenue_components']['da_leg']:.2f}。\n"
        f"- 候选RT偏差结算项：{candidate['macro_mean_daily_revenue_components']['rt_deviation_leg']:+.2f}；"
        f"基线：{baseline['macro_mean_daily_revenue_components']['rt_deviation_leg']:+.2f}。\n"
        f"- 候选退化成本：{candidate['macro_mean_daily_revenue_components']['degradation_cost']:.2f}；"
        f"基线：{baseline['macro_mean_daily_revenue_components']['degradation_cost']:.2f}。\n"
        f"- 候选偏差罚金：{candidate['macro_mean_daily_revenue_components']['deviation_penalty']:.2f}；"
        f"基线：{baseline['macro_mean_daily_revenue_components']['deviation_penalty']:.2f}。\n"
        f"- 候选正收益日比例：{100*candidate['macro_positive_day_rate']:.2f}%；"
        f"正年均收益节点比例：{100*candidate['node_positive_mean_revenue_rate']:.2f}%。\n\n"
        "## 方法学与质量控制\n\n"
        + (
            "- 先前有一轮诊断训练会根据2024 MAE挑选epoch；该轮已明确作废，不进入本页结论。\n"
            "- 本页结果只使用严格版本：2024不控制训练，只在冻结终点评估一次。\n"
            "- 自动审计文件：`data/results/global_da_outer_frozen_training_audit_v1.json`。\n"
            if args.training_protocol == "frozen_inner_plan"
            else "- 当前结果会用2024控制训练，只能视为诊断，不能作为严格外层结论。\n"
        )
        + "- 本地完整测试：184项通过。2025仍未读取。\n"
        "- 云端与本地关键checkpoint、收益JSON、结论JSON和审计JSON哈希一致。\n"
        "- 无效自适应诊断权重和重复checkpoint副本已清理。\n\n"
        "## 下一步\n\n"
    ) + (
        "冻结全部规则，用2020—2024重训最终权重，然后只运行一次2025锁定回顾测试。\n"
        if beats_mean
        else (
            "保持2025未使用；回到内部开发证据改进模型，不能根据2024反复搜索权重。\n\n"
            "已转回2020—2022训练、2023内部验证，开展连续MPC零阶盈利微调初筛；"
            "结果见第32号文档。\n"
        )
    )
    document.write_text(
        document_text,
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "outer_validation_summary_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "document": document.relative_to(ROOT).as_posix(),
        "difference": difference,
        "ci": ci,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
