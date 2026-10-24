#!/usr/bin/env python
"""Summarize frozen three-seed DA peak-dynamic replication and stability gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.tail_diagnostics import (  # noqa: E402
    daily_peak_report,
    event_detection_report,
    price_slice_report,
)
from scripts.decision_aware.diagnose_tail_and_profit_instability import (  # noqa: E402
    _load_simple,
    _thresholds,
)
from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (  # noqa: E402
    circular_block_ci,
)


def _paired(candidate: dict, original: dict, *, seed: int) -> dict:
    dates = sorted(original["macro_daily_net_revenue"])
    if dates != sorted(candidate["macro_daily_net_revenue"]):
        raise ValueError("候选与原系统逐日收益日期不一致")
    values = np.asarray([
        candidate["macro_daily_net_revenue"][day]
        - original["macro_daily_net_revenue"][day]
        for day in dates
    ], dtype=float)
    return {
        "mean_daily_difference": float(values.mean()),
        "median_daily_difference": float(np.median(values)),
        "positive_difference_day_rate": float(np.mean(values > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            values, block=7, draws=5000, seed=seed
        ),
    }


def _event_report(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    report = event_detection_report(
        prediction, target, threshold=threshold, direction="above", tolerance_hours=0
    )
    exact = report["exact"]
    truth = np.asarray(target) >= threshold
    predicted = np.asarray(prediction) >= threshold
    false_positives = int(np.sum(predicted & ~truth))
    non_events = int(np.sum(~truth))
    return {
        **report,
        "exact_precision": exact["precision"],
        "exact_recall": exact["recall"],
        "exact_f1": exact["f1"],
        "false_positive_rate": false_positives / non_events if non_events else None,
    }


def _prediction_paths(phase: str, kind: str, seed: str) -> Path:
    if kind == "candidate":
        if seed == "median":
            return ROOT / (
                "data/results/universal_lomo_ercot_da_peakdynamic_ensemble_v1/"
                f"peak_da_{phase}_median.npz"
            )
        return ROOT / (
            "data/results/universal_lomo_ercot_da_peakdynamic_predictions_v1/"
            f"peak_dynamic_da_{phase}_seed{seed}.npz"
        )
    if seed == "median":
        return ROOT / (
            "data/results/universal_lomo_ercot_da_peakdynamic_ensemble_v1/"
            f"original_da_{phase}_median.npz"
        )
    if seed == "0":
        suffix = "_outer_frozen" if phase == "outer" else ""
        return ROOT / f"data/results/universal_lomo_da{suffix}_v1/ercot_da_v1.npz"
    if phase == "inner":
        return ROOT / f"data/results/universal_lomo_multiseed_v1/ercot_da_inner_seed{seed}.npz"
    return ROOT / (
        "data/results/universal_lomo_multiseed_outer_v1/"
        f"ercot_da_outer_seed{seed}.npz"
    )


def _system_labels(seed: str) -> tuple[str, str]:
    token = "median" if seed == "median" else f"seed{seed}"
    original = f"da_original_{token}__rt_at_da_original_{token}__rt_original_{token}"
    candidate = f"da_peak_dynamic_{token}__rt_at_da_original_{token}__rt_original_{token}"
    return original, candidate


def _one(phase: str, seed: str, settlement: dict, threshold: float) -> dict:
    original_da = _load_simple(_prediction_paths(phase, "original", seed), "da")
    candidate_da = _load_simple(_prediction_paths(phase, "candidate", seed), "da")
    for key in ("nodes", "dates"):
        if not np.array_equal(original_da[key], candidate_da[key]):
            raise ValueError(f"{phase}/{seed}的{key}不一致")
    if not np.allclose(original_da["target"], candidate_da["target"], rtol=0, atol=1e-5):
        raise ValueError(f"{phase}/{seed}真值不一致")
    lower = _thresholds("ERCOT", original_da["nodes"], phase).da_lower_1pct
    original_slice = price_slice_report(
        original_da["prediction"], original_da["target"],
        lower_1pct=lower, upper_1pct=threshold,
    )
    candidate_slice = price_slice_report(
        candidate_da["prediction"], candidate_da["target"],
        lower_1pct=lower, upper_1pct=threshold,
    )
    original_peak = daily_peak_report(
        original_da["prediction"], original_da["target"], positive_threshold=threshold
    )
    candidate_peak = daily_peak_report(
        candidate_da["prediction"], candidate_da["target"], positive_threshold=threshold
    )
    original_label, candidate_label = _system_labels(seed)
    systems = settlement["systems"]
    if original_label not in systems or candidate_label not in systems:
        raise ValueError(f"结算缺少{original_label}或{candidate_label}")
    return {
        "seed": seed,
        "forecast": {
            "original": original_slice,
            "peak_dynamic": candidate_slice,
        },
        "daily_peak_on_true_spike_days": {
            "original": original_peak,
            "peak_dynamic": candidate_peak,
        },
        "positive_spike_event": {
            "original": _event_report(
                original_da["prediction"], original_da["target"], threshold
            ),
            "peak_dynamic": _event_report(
                candidate_da["prediction"], candidate_da["target"], threshold
            ),
        },
        "profit": {
            "original": systems[original_label],
            "peak_dynamic": systems[candidate_label],
            "paired_candidate_minus_original": _paired(
                systems[candidate_label], systems[original_label],
                seed=20261020 + (10 if seed == "median" else int(seed)),
            ),
        },
        "paths": {
            "original_da": _prediction_paths(phase, "original", seed).relative_to(ROOT).as_posix(),
            "candidate_da": _prediction_paths(phase, "candidate", seed).relative_to(ROOT).as_posix(),
        },
    }


def _phase(phase: str, settlement_path: Path) -> dict:
    settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
    if phase == "outer" and settlement.get("selection_allowed") is not False:
        raise ValueError("2024结算必须禁止选模")
    seed0 = _load_simple(_prediction_paths(phase, "original", "0"), "da")
    threshold = _thresholds("ERCOT", seed0["nodes"], phase).da_upper_1pct
    return {
        "phase": phase,
        "year": 2023 if phase == "inner" else 2024,
        "positive_threshold": threshold,
        "runs": {
            seed: _one(phase, seed, settlement, threshold)
            for seed in ("0", "1", "2", "median")
        },
        "settlement_path": settlement_path.relative_to(ROOT).as_posix(),
    }


def _gate(inner: dict, outer: dict) -> dict:
    consistent = []
    for seed in ("0", "1", "2"):
        inner_delta = inner["runs"][seed]["profit"]["paired_candidate_minus_original"]["mean_daily_difference"]
        outer_delta = outer["runs"][seed]["profit"]["paired_candidate_minus_original"]["mean_daily_difference"]
        if inner_delta > 0 and outer_delta > 0:
            consistent.append(int(seed))
    ensemble_ci = {
        "2023": inner["runs"]["median"]["profit"]["paired_candidate_minus_original"]["seven_day_block_bootstrap_95ci"],
        "2024": outer["runs"]["median"]["profit"]["paired_candidate_minus_original"]["seven_day_block_bootstrap_95ci"],
    }
    passed = len(consistent) >= 2 and all(interval[0] > 0 for interval in ensemble_ci.values())
    return {
        "consistent_positive_seeds_both_years": consistent,
        "consistent_positive_seed_count": len(consistent),
        "median_ensemble_ci": ensemble_ci,
        "passed": passed,
        "allowed_claim": (
            "方法具有初步训练稳定性"
            if passed else "seed 0有效，但方法对初始化敏感"
        ),
        "stop_and_do_not_retune": not passed,
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# DA局部峰值幅度三seed稳健性结果", "",
        "> 所有seed完整报告；固定逐小时中位数集成；不按ERCOT结果选seed；未读取2025。", "",
    ]
    for phase_key in ("inner_2023", "outer_2024"):
        phase = report[phase_key]
        lines.extend([
            f"## ERCOT {phase['year']}", "",
            "| 运行 | 候选普通MAE | 候选最高1% MAE | 候选峰值幅度保留率 | 候选真实峰值时刻幅度 | 候选尖峰召回 | 原收益 | 候选收益 | 收益差95%区间 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for seed in ("0", "1", "2", "median"):
            item = phase["runs"][seed]
            forecast = item["forecast"]["peak_dynamic"]
            peak = item["daily_peak_on_true_spike_days"]["peak_dynamic"]
            event = item["positive_spike_event"]["peak_dynamic"]
            profit = item["profit"]
            ci = profit["paired_candidate_minus_original"]["seven_day_block_bootstrap_95ci"]
            label = "固定中位数" if seed == "median" else f"seed {seed}"
            lines.append(
                f"| {label} | {_fmt(forecast['ordinary']['mae'])} | "
                f"{_fmt(forecast['highest_1pct']['mae'])} | "
                f"{_fmt(peak['predicted_daily_peak_to_true_peak_ratio'], 3)} | "
                f"{_fmt(peak['prediction_at_true_peak_to_true_peak_ratio'], 3)} | "
                f"{_fmt(event['exact_recall'], 3)} | "
                f"{profit['original']['macro_mean_daily_revenue']:.2f} | "
                f"{profit['peak_dynamic']['macro_mean_daily_revenue']:.2f} | "
                f"[{ci[0]:.2f}, {ci[1]:.2f}] |"
            )
        lines.append("")
    gate = report["stability_gate"]
    lines.extend([
        "## 冻结门槛结论", "",
        f"- 两年方向均为正的seed：{gate['consistent_positive_seeds_both_years']}（{gate['consistent_positive_seed_count']}/3）。",
        f"- 固定中位数集成2023区间：{gate['median_ensemble_ci']['2023']}。",
        f"- 固定中位数集成2024区间：{gate['median_ensemble_ci']['2024']}。",
        f"- 结论：**{gate['allowed_claim']}**。",
        "- 已改善与尚未解决必须分开表述；绝对正尖峰召回若仍为0，不得写成解决了尖峰预测。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-settlement", required=True)
    parser.add_argument("--outer-settlement", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    inner = _phase("inner", ROOT / args.inner_settlement)
    outer = _phase("outer", ROOT / args.outer_settlement)
    report = {
        "experiment": "da_peak_dynamic_multiseed_stability_v1",
        "pre_registered_median_ensemble": "elementwise hourly median of seeds 0,1,2",
        "seed_selection_by_ercot": False,
        "test_2025_read": False,
        "inner_2023": inner,
        "outer_2024": outer,
        "stability_gate": _gate(inner, outer),
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "event": "peak_dynamic_multiseed_summary_complete",
        "passed": report["stability_gate"]["passed"],
        "claim": report["stability_gate"]["allowed_claim"],
        "output_dir": args.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
