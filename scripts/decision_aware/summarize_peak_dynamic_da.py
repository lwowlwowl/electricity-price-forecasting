#!/usr/bin/env python
"""汇总DA局部峰值幅度后训练的2023开发与2024冻结检查。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.tail_diagnostics import daily_peak_report  # noqa: E402
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _forecast_report,
    _load_simple,
    _simple_triplet,
    _thresholds,
)
from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (  # noqa: E402
    circular_block_ci,
)


CURRENT = "da_current__rt_at_da_current__rt_current"
CANDIDATE = "da_peak_dynamic__rt_at_da_current__rt_current"
XGBOOST = (
    "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
)


def _paired(candidate: dict, current: dict) -> dict:
    dates = sorted(current["macro_daily_net_revenue"])
    if dates != sorted(candidate["macro_daily_net_revenue"]):
        raise ValueError("候选与当前系统的结算日期不一致")
    difference = np.asarray([
        candidate["macro_daily_net_revenue"][date]
        - current["macro_daily_net_revenue"][date]
        for date in dates
    ], dtype=np.float64)
    return {
        "mean_daily_difference": float(difference.mean()),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=20261020
        ),
    }


def _phase_report(
    phase: str,
    candidate_path: Path,
    settlement_path: Path,
) -> dict:
    current = _simple_triplet("ERCOT", phase)
    candidate_da = _load_simple(candidate_path, "da")
    for key in ("nodes", "dates"):
        if not np.array_equal(candidate_da[key], current["da"][key]):
            raise ValueError(f"{phase}候选DA的{key}与当前系统不一致")
    if not np.allclose(
        candidate_da["target"], current["da"]["target"], rtol=0, atol=1e-5
    ):
        raise ValueError(f"{phase}候选DA真值与当前系统不一致")
    candidate_triplet = dict(current)
    candidate_triplet["da"] = candidate_da
    thresholds = _thresholds("ERCOT", current["da"]["nodes"], phase)
    current_forecast = _forecast_report(current, thresholds)
    candidate_forecast = _forecast_report(candidate_triplet, thresholds)
    current_peak = daily_peak_report(
        current["da"]["prediction"], current["da"]["target"],
        positive_threshold=thresholds.da_upper_1pct,
    )
    candidate_peak = daily_peak_report(
        candidate_da["prediction"], candidate_da["target"],
        positive_threshold=thresholds.da_upper_1pct,
    )
    settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
    if phase == "outer" and settlement.get("selection_allowed") is not False:
        raise ValueError("2024冻结结算不得允许选模")
    systems = settlement["systems"]
    current_profit = systems[CURRENT]
    candidate_profit = systems[CANDIDATE]
    xgboost_profit = systems.get(XGBOOST)
    return {
        "phase": phase,
        "year": 2023 if phase == "inner" else 2024,
        "selection_allowed": phase == "inner",
        "positive_threshold": thresholds.da_upper_1pct,
        "forecast": {
            "current": current_forecast,
            "peak_dynamic": candidate_forecast,
        },
        "daily_peak_on_true_spike_days": {
            "current": current_peak,
            "peak_dynamic": candidate_peak,
        },
        "profit": {
            "current": current_profit,
            "peak_dynamic": candidate_profit,
            "xgboost_robust": xgboost_profit,
            "paired_vs_current": _paired(candidate_profit, current_profit),
            "paired_vs_xgboost": (
                _paired(candidate_profit, xgboost_profit)
                if xgboost_profit is not None else None
            ),
        },
        "source_paths": {
            "candidate_da": candidate_path.relative_to(ROOT).as_posix(),
            "settlement": settlement_path.relative_to(ROOT).as_posix(),
        },
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# DA局部峰值幅度后训练结果", "",
        "> 训练和选模只使用三个源市场2020—2023；ERCOT 2023/2024均不参与checkpoint选择，2025未读取。",
        "",
        f"源市场选定epoch：{report['source_market_selection']['epoch']}；"
        f"普通MAE守门上限：{report['source_market_selection']['mae_limit']:.4f}。",
    ]
    for key in ("inner_2023", "outer_2024"):
        item = report[key]
        c_da = item["forecast"]["current"]["da"]
        p_da = item["forecast"]["peak_dynamic"]["da"]
        c_peak = item["daily_peak_on_true_spike_days"]["current"]
        p_peak = item["daily_peak_on_true_spike_days"]["peak_dynamic"]
        c_profit = item["profit"]["current"]
        p_profit = item["profit"]["peak_dynamic"]
        x_profit = item["profit"]["xgboost_robust"]
        paired = item["profit"]["paired_vs_current"]
        ci = paired["seven_day_block_bootstrap_95ci"]
        lines.extend([
            "", f"## ERCOT {item['year']}", "",
            "| 指标 | 原Transformer | 峰值动态后训练 |", "|---|---:|---:|",
            f"| 全时段DA MAE | {_fmt(c_da['all']['mae'])} | {_fmt(p_da['all']['mae'])} |",
            f"| 普通时段DA MAE | {_fmt(c_da['ordinary']['mae'])} | {_fmt(p_da['ordinary']['mae'])} |",
            f"| 最高1% DA MAE | {_fmt(c_da['highest_1pct']['mae'])} | {_fmt(p_da['highest_1pct']['mae'])} |",
            f"| 正尖峰召回 | {_fmt(c_da['highest_1pct']['spike_recall'], 3)} | {_fmt(p_da['highest_1pct']['spike_recall'], 3)} |",
            f"| 日峰值±1小时命中 | {_fmt(c_peak['within_1h_rate'], 3)} | {_fmt(p_peak['within_1h_rate'], 3)} |",
            f"| 预测/真实日峰值 | {_fmt(c_peak['predicted_daily_peak_to_true_peak_ratio'], 3)} | {_fmt(p_peak['predicted_daily_peak_to_true_peak_ratio'], 3)} |",
            f"| 日均净收益 | {_fmt(c_profit['macro_mean_daily_revenue'])} | {_fmt(p_profit['macro_mean_daily_revenue'])} |",
            "",
            f"净收益配对差：{paired['mean_daily_difference']:+.2f} USD/节点·日；"
            f"7日块95%区间[{ci[0]:.2f}, {ci[1]:.2f}]。",
        ])
        if x_profit is not None:
            x_paired = item["profit"]["paired_vs_xgboost"]
            x_ci = x_paired["seven_day_block_bootstrap_95ci"]
            systems = [
                ("原Transformer", c_profit),
                ("峰值动态后训练", p_profit),
                ("逐节点稳健XGBoost", x_profit),
            ]
            lines.extend([
                "", "### 收益分量与强基线", "",
                "| 系统 | DA计划腿 | RT修正腿 | 退化成本 | 偏差罚金 | 净收益 |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for label, system in systems:
                component = system["macro_mean_daily_revenue_components"]
                lines.append(
                    f"| {label} | {component['da_leg']:.2f} | "
                    f"{component['rt_deviation_leg']:.2f} | "
                    f"{component['degradation_cost']:.2f} | "
                    f"{component['deviation_penalty']:.2f} | "
                    f"{system['macro_mean_daily_revenue']:.2f} |"
                )
            lines.extend([
                "",
                f"候选相对XGBoost：{x_paired['mean_daily_difference']:+.2f} "
                f"USD/节点·日；7日块95%区间[{x_ci[0]:.2f}, {x_ci[1]:.2f}]。",
            ])
    lines.extend([
        "", "## 解释边界", "",
        "本实验只改变DA曲线的局部峰值幅度训练目标。即使幅度指标改善，只有完整净收益同时改善，才可称为经济上有效；否则保留为机制性负结果，不调权重继续追逐。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-candidate", required=True)
    parser.add_argument("--outer-candidate", required=True)
    parser.add_argument("--inner-settlement", required=True)
    parser.add_argument("--outer-settlement", required=True)
    parser.add_argument("--selection-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    selection = json.loads((ROOT / args.selection_summary).read_text(encoding="utf-8"))
    report = {
        "experiment": "da_local_peak_excursion_post_training_v1",
        "status": "mechanism_experiment_not_locked_2025",
        "single_changed_mechanism": "daily max-minus-median excursion loss",
        "source_market_selection": selection,
        "inner_2023": _phase_report(
            "inner", ROOT / args.inner_candidate, ROOT / args.inner_settlement
        ),
        "outer_2024": _phase_report(
            "outer", ROOT / args.outer_candidate, ROOT / args.outer_settlement
        ),
        "test_2025_read": False,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "event": "peak_dynamic_summary_complete",
        "selected_epoch": selection["epoch"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
