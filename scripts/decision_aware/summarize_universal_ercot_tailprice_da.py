#!/usr/bin/env python
"""汇总ERCOT留出折DA价格尾部加权的预注册单因素实验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _forecast_report,
    _load_simple,
    _simple_triplet,
    _thresholds,
)
from decision_aware.config import PilotConfig  # noqa: E402
from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (  # noqa: E402
    circular_block_ci,
)


CURRENT = "da_current__rt_at_da_current__rt_current"
CANDIDATE = "da_tail_price_da__rt_at_da_current__rt_current"


def _require_matching_contract(candidate: dict, current: dict) -> None:
    for key in ("nodes", "dates", "target"):
        left = candidate[key]
        right = current[key]
        if key in {"nodes", "dates"}:
            equal = np.array_equal(left, right)
        else:
            equal = np.allclose(left, right, rtol=0.0, atol=1e-5)
        if not equal:
            raise ValueError(f"候选DA与当前DA的{key}不一致")
    metadata = candidate.get("metadata", {})
    if metadata.get("protocol") != "leave_one_market_out_zero_shot":
        raise ValueError("候选DA必须是LOMO零样本结果")
    if metadata.get("final_profit_claim_allowed") is not False:
        raise ValueError("候选DA不得被标记为最终盈利结果")
    if metadata.get("holdout_market") != "ERCOT":
        raise ValueError("候选DA留出市场必须是ERCOT")
    if metadata.get("uses_node_id_for_prediction") is not False:
        raise ValueError("候选DA必须是无ID模型")
    if metadata.get("uses_market_id_for_prediction") is not False:
        raise ValueError("候选DA必须是无市场ID模型")
    if metadata.get("training_markets") != ["CAISO", "NYISO", "PJM"]:
        raise ValueError("候选DA训练市场合同不匹配")


def _validate_single_factor_config(config_path: Path) -> None:
    cfg = PilotConfig.from_yaml(config_path)
    if cfg.global_use_entity_embeddings:
        raise ValueError("尾部候选配置不得启用实体embedding")
    if tuple(cfg.global_training_markets) != ("CAISO", "NYISO", "PJM"):
        raise ValueError("尾部候选配置的源市场不匹配")
    expected = {
        "global_use_critical_event_weighting": True,
        "global_event_price_lower_quantile": 0.01,
        "global_event_price_upper_quantile": 0.99,
        "global_event_price_weight": 1.0,
        "global_event_spread_weight": 0.0,
        "global_event_sign_reversal_weight": 0.0,
        "global_event_rt_first_step_weight": 0.0,
        "global_event_max_weight": 2.0,
    }
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"尾部候选配置{key}不符合预注册值")


def paired_profit(candidate: dict, current: dict) -> dict:
    candidate_daily = candidate["macro_daily_net_revenue"]
    current_daily = current["macro_daily_net_revenue"]
    dates = sorted(current_daily)
    if dates != sorted(candidate_daily):
        raise ValueError("候选与当前系统的共同交付日不一致")
    difference = np.asarray(
        [candidate_daily[day] - current_daily[day] for day in dates],
        dtype=np.float64,
    )
    return {
        "mean_daily_difference": float(difference.mean()),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=20261017
        ),
    }


def acceptance_decision(
    current_da: dict,
    candidate_da: dict,
    current_profit: float,
    candidate_profit: float,
) -> dict:
    """冻结的继续扩展门槛；不读取2024或2025。"""
    current_tail = float(current_da["highest_1pct"]["mae"])
    candidate_tail = float(candidate_da["highest_1pct"]["mae"])
    current_ordinary = float(current_da["ordinary"]["mae"])
    candidate_ordinary = float(candidate_da["ordinary"]["mae"])
    current_negative = current_da["negative_price"]["negative_recall"]
    candidate_negative = candidate_da["negative_price"]["negative_recall"]
    negative_not_worse = (
        True
        if current_negative is None
        else candidate_negative is not None
        and float(candidate_negative) >= float(current_negative)
    )
    checks = {
        "highest_1pct_da_mae_improves": candidate_tail < current_tail,
        "ordinary_da_mae_within_2pct": (
            candidate_ordinary <= current_ordinary * 1.02
        ),
        "negative_price_recall_not_worse": negative_not_worse,
        "mean_daily_profit_improves": candidate_profit > current_profit,
    }
    return {
        "checks": checks,
        "eligible_to_expand": all(checks.values()),
        "rule": (
            "Only expand beyond this ERCOT seed-0 pilot when all four frozen checks "
            "pass; the paired interval is reported but is not a post-hoc tuning target."
        ),
    }


def summarize(
    candidate_da_path: Path,
    settlement_path: Path,
    config_path: Path,
) -> dict:
    _validate_single_factor_config(config_path)
    current = _simple_triplet("ERCOT", "inner")
    candidate_da = _load_simple(candidate_da_path, "da")
    _require_matching_contract(candidate_da, current["da"])
    candidate = dict(current)
    candidate["da"] = candidate_da
    thresholds = _thresholds("ERCOT", current["da"]["nodes"], "inner")
    current_forecast = _forecast_report(current, thresholds)
    candidate_forecast = _forecast_report(candidate, thresholds)

    settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
    if settlement.get("split") != "2023 inner validation only":
        raise ValueError("结算报告不是2023 inner开发口径")
    systems = settlement["systems"]
    current_profit = systems[CURRENT]
    candidate_profit = systems[CANDIDATE]
    paired = paired_profit(candidate_profit, current_profit)
    acceptance = acceptance_decision(
        current_forecast["da"],
        candidate_forecast["da"],
        float(current_profit["macro_mean_daily_revenue"]),
        float(candidate_profit["macro_mean_daily_revenue"]),
    )
    return {
        "experiment": "universal_lomo_ercot_da_tailprice_single_factor_v1",
        "status": "2023_inner_development_pilot_not_final_test",
        "test_2024_used_for_selection": False,
        "test_2025_read": False,
        "single_changed_mechanism": (
            "double the normalized DA MAE weight only for each source node's "
            "training-period lowest/highest 1% price targets"
        ),
        "thresholds": thresholds.to_dict(),
        "forecast": {
            "current": current_forecast,
            "tail_price_da": candidate_forecast,
        },
        "profit": {
            "current": current_profit,
            "tail_price_da": candidate_profit,
            "paired_vs_current": paired,
        },
        "acceptance": acceptance,
        "source_paths": {
            "candidate_da": str(candidate_da_path),
            "settlement": str(settlement_path),
            "config": str(config_path),
        },
    }


def markdown(report: dict) -> str:
    current_da = report["forecast"]["current"]["da"]
    candidate_da = report["forecast"]["tail_price_da"]["da"]
    current_profit = report["profit"]["current"]["macro_mean_daily_revenue"]
    candidate_profit = report["profit"]["tail_price_da"]["macro_mean_daily_revenue"]
    paired = report["profit"]["paired_vs_current"]
    interval = paired["seven_day_block_bootstrap_95ci"]
    checks = report["acceptance"]["checks"]
    rows = [
        ("DA全时段MAE", current_da["all"]["mae"], candidate_da["all"]["mae"]),
        ("DA普通时段MAE", current_da["ordinary"]["mae"], candidate_da["ordinary"]["mae"]),
        ("DA最高1% MAE", current_da["highest_1pct"]["mae"], candidate_da["highest_1pct"]["mae"]),
        ("DA尖峰召回率", current_da["highest_1pct"]["spike_recall"], candidate_da["highest_1pct"]["spike_recall"]),
        ("DA负价召回率", current_da["negative_price"]["negative_recall"], candidate_da["negative_price"]["negative_recall"]),
    ]
    lines = [
        "# ERCOT无ID DA尾部价格加权：单因素试验", "",
        "> 只使用2023 inner开发口径；2024不参与选择，2025未读取。", "",
        "| 指标 | 当前DA | 尾部加权DA |", "|---|---:|---:|",
    ]
    for label, left, right in rows:
        left_text = "-" if left is None else f"{left:.4f}"
        right_text = "-" if right is None else f"{right:.4f}"
        lines.append(f"| {label} | {left_text} | {right_text} |")
    lines.extend((
        "", "## 完整收益", "",
        f"当前系统：{current_profit:.2f} USD/节点·日；尾部加权DA系统：{candidate_profit:.2f}。",
        f"配对差：{paired['mean_daily_difference']:+.2f}，7日块95%区间"
        f"[{interval[0]:.2f}, {interval[1]:.2f}]。", "",
        "## 预注册门槛", "",
    ))
    for name, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    lines.extend((
        "",
        "只有四项全部通过，才扩展到RT-at-DA或其他LOMO折；否则作为负结果停止。",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/decision_aware/"
            "universal_lomo_holdout_ercot_da_tailprice_v1.yaml"
        ),
    )
    parser.add_argument(
        "--candidate-da",
        default=(
            "data/results/universal_lomo_ercot_tailprice_da_predictions_v1/"
            "tail_price_da_da_inner_seed0.npz"
        ),
    )
    parser.add_argument(
        "--settlement",
        default=(
            "data/results/"
            "universal_lomo_ercot_tailprice_da_continuous_mpc_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/universal_lomo_ercot_tailprice_da_summary_v1",
    )
    args = parser.parse_args()
    result = summarize(
        ROOT / args.candidate_da,
        ROOT / args.settlement,
        ROOT / args.config,
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        markdown(result), encoding="utf-8"
    )
    print(json.dumps({
        "event": "tailprice_da_summary_complete",
        "eligible_to_expand": result["acceptance"]["eligible_to_expand"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
