#!/usr/bin/env python
"""运行CAISO/NYISO/PJM三折正尖峰概率门控(P-E)可用性审计。

硬隔离：不加载ERCOT，不读取2025，不拟合P-A，不修改价格，不运行结算。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.positive_spike_day_audit import (  # noqa: E402
    DAY_FEATURE_NAMES,
    bootstrap_pr_auc_difference,
    coefficient_rows,
    fit_sparse_event_model,
    make_daily_bundle,
    node_event_thresholds,
    quantile_calibration,
)
from decision_aware.positive_spike_probability_gate import (  # noqa: E402
    THRESHOLD_GRID,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_platt_calibrator,
    metrics_with_optional_threshold,
    node_equal_metrics,
    pe_market_gate,
    select_fpr_budget_threshold,
)


MARKETS = ("CAISO", "NYISO", "PJM")
FIT_YEARS = (2020, 2021)
CALIBRATION_YEAR = 2022
EVALUATION_YEAR = 2023
FEATURE_CACHE = ROOT / "data/results/negative_price_adapter_v1/feature_cache"
OUTPUT = ROOT / "data/results/positive_spike_probability_gate_lomo_v1"


def _load_bundle(market: str, split: str) -> dict:
    if market not in MARKETS:
        raise ValueError(f"禁止加载非预注册市场: {market}")
    path = FEATURE_CACHE / f"{market.lower()}_{split}_features_v2.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    dates = np.asarray(bundle["dates"]).astype(str)
    years = {int(value[:4]) for value in dates}
    if any(year >= 2025 for year in years):
        raise RuntimeError(f"{path}包含禁止读取的2025或以后日期")
    expected = {2020, 2021, 2022} if split == "train" else {2023}
    if years != expected:
        raise RuntimeError(f"{path}年份异常: {sorted(years)}，预期{sorted(expected)}")
    return bundle


def _combine(parts: list[tuple[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([part.x for _, part in parts]),
        np.concatenate([part.y for _, part in parts]),
        np.concatenate([
            np.full(len(part.y), market, dtype="U8") for market, part in parts
        ]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csvs(report: dict) -> None:
    with (OUTPUT / "node_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = (
            "holdout_market", "node", "samples", "events", "event_rate", "pr_auc",
            "pr_auc_lift", "precision", "recall", "f1", "false_positive_rate",
            "brier", "constant_event_rate_brier", "brier_improvement_vs_constant",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for market in MARKETS:
            for row in report["folds"][market]["node_metrics"]:
                writer.writerow({key: market if key == "holdout_market" else row.get(key) for key in fields})

    with (OUTPUT / "calibration.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = (
            "holdout_market", "bin", "samples", "mean_probability", "observed_rate",
            "minimum_probability", "maximum_probability",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for market in MARKETS:
            for row in report["folds"][market]["calibration_bins"]:
                writer.writerow({"holdout_market": market, **row})

    with (OUTPUT / "coefficients.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("holdout_market", "feature", "standardized_coefficient", "nonzero")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for market in MARKETS:
            for row in report["folds"][market]["coefficients"]:
                writer.writerow({"holdout_market": market, **row})

    with (OUTPUT / "threshold_search.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("holdout_market", "threshold", "feasible", "source_market", "fpr", "recall")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for holdout in MARKETS:
            for candidate in report["folds"][holdout]["threshold_search"]:
                for source_market, metric in candidate["by_market"].items():
                    writer.writerow({
                        "holdout_market": holdout,
                        "threshold": candidate["threshold"],
                        "feasible": candidate["feasible"],
                        "source_market": source_market,
                        "fpr": metric["false_positive_rate"],
                        "recall": metric["recall"],
                    })


def _write_plot(report: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for axis, market in zip(axes, MARKETS, strict=True):
        rows = report["folds"][market]["calibration_bins"]
        axis.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="ideal")
        axis.plot(
            [row["mean_probability"] for row in rows],
            [row["observed_rate"] for row in rows],
            marker="o", color="#2f6bff", linewidth=1.8, label="P-E",
        )
        axis.set_title(market)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[-1].legend(loc="upper left", frameon=False)
    figure.suptitle("P-E probability calibration (2023 holdout)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "calibration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _write_markdown(report: dict) -> None:
    lines = [
        "# 正尖峰概率门控P-E三折可用性审计", "",
        "> 样本单位为节点—交付日；只使用CAISO、NYISO、PJM；ERCOT未读取；2025未读取；P-A未拟合。",
        "", "## 三折结果", "",
        "| 留出市场 | 阈值 | PR-AUC/率 | Brier改善(总体/节点等权) | FPR(总体/节点等权) | Recall | ECE | 通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for market in MARKETS:
        fold = report["folds"][market]
        overall, equal = fold["overall_metrics"], fold["node_equal_metrics"]
        lines.append(
            f"| {market} | {_fmt(fold['selected_threshold'])} | {_fmt(overall['pr_auc_lift'])}× | "
            f"{_fmt(overall['brier_improvement_vs_constant'])}/{_fmt(equal['brier_improvement_vs_constant'])} | "
            f"{_fmt(overall['false_positive_rate'])}/{_fmt(equal['false_positive_rate'])} | "
            f"{_fmt(overall['recall'])} | {fold['ece']:.3f} | {'是' if fold['gate']['passed'] else '否'} |"
        )
    gate = report["gate"]
    lines.extend((
        "", "## 冻结结论", "",
        f"- 通过市场：{', '.join(gate['passed_markets']) if gate['passed_markets'] else '无'}；",
        f"- 失败市场：{', '.join(gate['failed_markets']) if gate['failed_markets'] else '无'}；",
        f"- P-E三折硬门：**{'通过' if gate['pe_passed'] else '失败'}**；",
        f"- P-A是否允许拟合：**{'是' if gate['allow_pa_fit'] else '否'}**；",
        f"- ERCOT是否允许运行：**否**（P-A尚未通过且本脚本没有ERCOT入口）。", "",
        gate["required_action"], "",
    ))
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = perf_counter()
    source = {
        market: {split: _load_bundle(market, split) for split in ("train", "val")}
        for market in MARKETS
    }
    thresholds = {
        market: node_event_thresholds(source[market]["train"]) for market in MARKETS
    }
    daily = {
        market: {
            "fit": make_daily_bundle(source[market]["train"], thresholds[market], years=FIT_YEARS),
            "calibrate": make_daily_bundle(source[market]["train"], thresholds[market], years=(CALIBRATION_YEAR,)),
            "evaluate": make_daily_bundle(source[market]["val"], thresholds[market], years=(EVALUATION_YEAR,)),
        }
        for market in MARKETS
    }

    folds = {}
    for fold_index, holdout in enumerate(MARKETS):
        training_markets = [market for market in MARKETS if market != holdout]
        fit_x, fit_y, fit_markets = _combine([
            (market, daily[market]["fit"]) for market in training_markets
        ])
        base = fit_sparse_event_model(fit_x, fit_y, fit_markets)
        cal_x, cal_y, cal_markets = _combine([
            (market, daily[market]["calibrate"]) for market in training_markets
        ])
        calibrator = fit_platt_calibrator(base.decision_function(cal_x), cal_y, cal_markets)
        calibrated_source = calibrator.predict(base.decision_function(cal_x))
        selected_threshold, threshold_search = select_fpr_budget_threshold(
            calibrated_source, cal_y, cal_markets,
            candidates=THRESHOLD_GRID, source_fpr_budget=0.05,
        )

        evaluation = daily[holdout]["evaluate"]
        probability = calibrator.predict(base.decision_function(evaluation.x))
        overall = metrics_with_optional_threshold(probability, evaluation.y, selected_threshold)
        equal, node_rows = node_equal_metrics(
            probability, evaluation.y, evaluation.node_ids, evaluation.nodes,
            selected_threshold,
        )
        bootstrap = bootstrap_pr_auc_difference(
            probability, evaluation.y, evaluation.dates,
            repeats=2000, seed=20261020 + fold_index,
        )
        calibration_bins = quantile_calibration(probability, evaluation.y)
        gate = pe_market_gate(overall, equal, bootstrap, selected_threshold)
        coefficients = coefficient_rows(base)
        folds[holdout] = {
            "holdout_market": holdout,
            "training_markets": training_markets,
            "sample_unit": "node-delivery-day",
            "fit_years": list(FIT_YEARS),
            "calibration_and_threshold_year": CALIBRATION_YEAR,
            "evaluation_year": EVALUATION_YEAR,
            "selected_threshold": selected_threshold,
            "threshold_grid": {"start": 0.01, "stop": 0.99, "step": 0.01, "count": 99},
            "threshold_search": threshold_search,
            "platt": {"slope": calibrator.slope, "intercept": calibrator.intercept},
            "overall_metrics": overall,
            "node_equal_metrics": equal,
            "node_metrics": node_rows,
            "pr_auc_minus_event_rate_bootstrap": bootstrap,
            "calibration_bins": calibration_bins,
            "ece": expected_calibration_error(calibration_bins),
            "evaluation_calibration": calibration_slope_intercept(probability, evaluation.y),
            "coefficients": coefficients,
            "base_intercept": float(base.named_steps["event"].intercept_[0]),
            "nonzero_coefficients": int(sum(row["nonzero"] for row in coefficients)),
            "gate": gate,
            "event_price_thresholds_by_node": {
                str(node): float(value)
                for node, value in zip(evaluation.nodes, thresholds[holdout], strict=True)
            },
        }
        print(json.dumps({
            "event": "pe_fold_complete", "holdout": holdout,
            "threshold": selected_threshold, "overall": overall,
            "node_equal": equal, "gate": gate,
        }, ensure_ascii=False), flush=True)

    passed = [market for market in MARKETS if folds[market]["gate"]["passed"]]
    failed = [market for market in MARKETS if not folds[market]["gate"]["passed"]]
    pe_passed = len(passed) >= 2
    gate = {
        "rule": "at least 2/3 markets pass every frozen P-E check",
        "passed_markets": passed,
        "failed_markets": failed,
        "pe_passed": pe_passed,
        "allow_pa_fit": pe_passed,
        "allow_ercot": False,
        "required_action": (
            "P-E通过；只允许按文档68首次拟合并审计P-A，仍不得运行ERCOT。"
            if pe_passed else
            "P-E失败；停止，不拟合P-A，不创建或运行ERCOT结算入口。"
        ),
    }
    report = {
        "experiment": "positive_spike_probability_gate_lomo_v1",
        "status": "complete",
        "sample_unit": "node-delivery-day",
        "markets": list(MARKETS),
        "uses_ercot": False,
        "ercot_files_read": [],
        "test_2025_read": False,
        "pa_fitted": False,
        "ercot_settlement_run": False,
        "minimum_trigger_price_rule": "max(original_peak, q99_node + 1 USD/MWh)",
        "model": {
            "base": "StandardScaler + L1 LogisticRegression(C=0.1)",
            "calibration": "shared source-market Platt slope/intercept",
            "source_fpr_budget": 0.05,
            "threshold_grid": list(THRESHOLD_GRID),
            "features": list(DAY_FEATURE_NAMES),
        },
        "folds": folds,
        "gate": gate,
        "input_files": {
            f"{market}_{split}": {
                "path": str((FEATURE_CACHE / f"{market.lower()}_{split}_features_v2.npz").relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(FEATURE_CACHE / f"{market.lower()}_{split}_features_v2.npz"),
            }
            for market in MARKETS for split in ("train", "val")
        },
        "runtime_seconds": perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_csvs(report)
    _write_plot(report)
    _write_markdown(report)
    artifacts = [OUTPUT / name for name in (
        "summary.json", "summary.md", "node_metrics.csv", "calibration.csv",
        "coefficients.csv", "threshold_search.csv", "calibration_curve.png",
    )]
    (OUTPUT / "completed_outputs.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in artifacts) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "pe_audit_complete", "gate": gate,
        "runtime_seconds": report["runtime_seconds"],
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
