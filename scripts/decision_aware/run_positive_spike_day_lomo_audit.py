#!/usr/bin/env python
"""运行CAISO/NYISO/PJM正尖峰日级留一市场可预测性审计。

ERCOT不被加载；2025被硬拒绝。脚本只训练稀疏逻辑回归审计器，不包含幅度模型。
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
    binary_metrics,
    bootstrap_pr_auc_difference,
    coefficient_rows,
    fit_sparse_event_model,
    make_daily_bundle,
    node_event_thresholds,
    quantile_calibration,
    select_macro_f1_threshold,
)


MARKETS = ("CAISO", "NYISO", "PJM")
FIT_YEARS = (2020, 2021)
THRESHOLD_YEAR = 2022
EVALUATION_YEAR = 2023
FEATURE_CACHE = ROOT / "data/results/negative_price_adapter_v1/feature_cache"
OUTPUT = ROOT / "data/results/positive_spike_day_lomo_audit_v1"


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
    if split == "train" and not years.issubset({2020, 2021, 2022}):
        raise RuntimeError(f"训练缓存年份异常: {sorted(years)}")
    if split == "val" and years != {2023}:
        raise RuntimeError(f"验证缓存必须严格为2023: {sorted(years)}")
    return bundle


def _combine(parts: list[tuple[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([part.x for _, part in parts])
    y = np.concatenate([part.y for _, part in parts])
    markets = np.concatenate([
        np.full(len(part.y), market, dtype="U8") for market, part in parts
    ])
    return x, y, markets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csvs(report: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "calibration.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "holdout_market", "bin", "samples", "mean_probability", "observed_rate",
            "minimum_probability", "maximum_probability",
        ))
        writer.writeheader()
        for market in MARKETS:
            for row in report["folds"][market]["calibration"]:
                writer.writerow({"holdout_market": market, **row})


def _write_calibration_plot(report: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for axis, market in zip(axes, MARKETS, strict=True):
        rows = report["folds"][market]["calibration"]
        predicted = [row["mean_probability"] for row in rows]
        observed = [row["observed_rate"] for row in rows]
        axis.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="ideal")
        axis.plot(predicted, observed, marker="o", color="#2f6bff", linewidth=1.8, label="LOMO")
        axis.set_title(market)
        axis.set_xlim(0, 0.65)
        axis.set_ylim(0, 0.65)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed event rate")
    axes[-1].legend(loc="upper left", frameon=False)
    figure.suptitle("Positive-spike day probability calibration (2023 holdout)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "calibration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    with (OUTPUT / "coefficients.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "holdout_market", "feature", "standardized_coefficient", "nonzero",
        ))
        writer.writeheader()
        for market in MARKETS:
            for row in report["folds"][market]["coefficients"]:
                writer.writerow({"holdout_market": market, **row})


def _write_markdown(report: dict) -> None:
    lines = [
        "# 正尖峰日级跨市场可预测性审计结果", "",
        "> 三折仅使用CAISO、NYISO、PJM；模型拟合为2020—2021，概率阈值只由训练市场2022选择，留出市场2023只评价；ERCOT未读取；2025未读取。",
        "", "## 留一市场结果", "",
        "| 留出市场 | 事件日/样本 | 事件率 | PR-AUC | 相对事件率 | Precision | Recall | F1 | Brier | 误报率 | ΔPR-AUC 95%CI | 明显通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for market in MARKETS:
        fold = report["folds"][market]
        metric = fold["metrics"]
        ci = fold["pr_auc_minus_event_rate_bootstrap"]["ci95"]
        lines.append(
            f"| {market} | {metric['events']}/{metric['samples']} | {metric['event_rate']:.3f} | "
            f"{metric['pr_auc']:.3f} | {metric['pr_auc_lift']:.2f}× | {metric['precision']:.3f} | "
            f"{metric['recall']:.3f} | {metric['f1']:.3f} | {metric['brier']:.3f} | "
            f"{metric['false_positive_rate']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{'是' if fold['clearly_above_baseline'] else '否'} |"
        )
    gate = report["gate"]
    lines.extend((
        "", "## 冻结门槛", "",
        f"- 明显通过市场：{', '.join(gate['passed_markets']) if gate['passed_markets'] else '无'}；",
        f"- 未明显通过市场：{', '.join(gate['failed_markets']) if gate['failed_markets'] else '无'}；",
        f"- 判定：**{gate['conclusion']}**；",
        f"- 是否允许预注册概率事件头与条件幅度分布：**{'是' if gate['allow_next_model_preregistration'] else '否'}**。",
        "", "## 可解释系数", "",
    ))
    for market in MARKETS:
        fold = report["folds"][market]
        nonzero = [row for row in fold["coefficients"] if row["nonzero"]]
        nonzero.sort(key=lambda row: -abs(row["standardized_coefficient"]))
        lines.extend((
            f"### 留出{market}", "",
            f"非零系数：{len(nonzero)}/{len(DAY_FEATURE_NAMES)}；截距：{fold['intercept']:+.4f}。", "",
        ))
        lines.extend(
            f"- `{row['feature']}`: {row['standardized_coefficient']:+.4f}"
            for row in nonzero
        )
        if not nonzero:
            lines.append("- 无非零系数。")
        lines.append("")
    lines.extend((
        "## 校准", "",
        "每折10个等频概率箱见`calibration.csv`。这些结果只用于审计，没有根据留出市场重新校准。", "",
        "## 停止规则", "",
        gate["required_action"], "",
    ))
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = perf_counter()
    source = {
        market: {
            "train": _load_bundle(market, "train"),
            "val": _load_bundle(market, "val"),
        }
        for market in MARKETS
    }
    thresholds = {
        market: node_event_thresholds(source[market]["train"])
        for market in MARKETS
    }
    daily = {
        market: {
            "fit": make_daily_bundle(source[market]["train"], thresholds[market], years=FIT_YEARS),
            "select": make_daily_bundle(source[market]["train"], thresholds[market], years=(THRESHOLD_YEAR,)),
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
        model = fit_sparse_event_model(fit_x, fit_y, fit_markets)
        select_x, select_y, select_markets = _combine([
            (market, daily[market]["select"]) for market in training_markets
        ])
        select_probability = model.predict_proba(select_x)[:, 1]
        probability_threshold, threshold_rows = select_macro_f1_threshold(
            select_probability, select_y, select_markets
        )
        evaluation = daily[holdout]["evaluate"]
        probability = model.predict_proba(evaluation.x)[:, 1]
        metrics = binary_metrics(probability, evaluation.y, probability_threshold)
        bootstrap = bootstrap_pr_auc_difference(
            probability, evaluation.y, evaluation.dates,
            repeats=2000, seed=20261020 + fold_index,
        )
        clearly = bool(
            metrics["pr_auc_lift"] is not None
            and metrics["pr_auc_lift"] >= 1.25
            and bootstrap["ci95"][0] > 0.0
        )
        coefficients = coefficient_rows(model)
        folds[holdout] = {
            "holdout_market": holdout,
            "training_markets": training_markets,
            "fit_years": list(FIT_YEARS),
            "probability_threshold_selection_year": THRESHOLD_YEAR,
            "evaluation_year": EVALUATION_YEAR,
            "event_price_threshold_source": "each node 2020-2021 hourly DA q99",
            "probability_threshold": probability_threshold,
            "threshold_selection_best_macro_f1": max(row["macro_f1"] for row in threshold_rows),
            "metrics": metrics,
            "pr_auc_minus_event_rate_bootstrap": bootstrap,
            "clearly_above_baseline": clearly,
            "calibration": quantile_calibration(probability, evaluation.y),
            "coefficients": coefficients,
            "intercept": float(model.named_steps["event"].intercept_[0]),
            "nonzero_coefficients": int(sum(row["nonzero"] for row in coefficients)),
            "event_price_thresholds_by_node": {
                str(node): float(value)
                for node, value in zip(evaluation.nodes, thresholds[holdout], strict=True)
            },
        }
        print(json.dumps({
            "event": "positive_spike_day_fold_complete", "holdout": holdout,
            "threshold": probability_threshold, "metrics": metrics,
            "bootstrap_ci95": bootstrap["ci95"], "clearly_above_baseline": clearly,
        }, ensure_ascii=False), flush=True)

    passed = [market for market in MARKETS if folds[market]["clearly_above_baseline"]]
    failed = [market for market in MARKETS if not folds[market]["clearly_above_baseline"]]
    audit_passed = len(passed) >= 2
    gate = {
        "rule": "at least two of three holdout markets must have lift>=1.25 and bootstrap CI lower>0",
        "passed_markets": passed,
        "failed_markets": failed,
        "audit_passed": audit_passed,
        "conclusion": (
            "正尖峰日级概率包含可跨市场泛化信号"
            if audit_passed else "现有输入缺少跨市场尖峰先验信号"
        ),
        "allow_next_model_preregistration": audit_passed,
        "required_action": (
            "审计通过后只能另写预注册；本轮仍不得直接训练概率事件头或条件幅度分布模型。"
            if audit_passed else
            "至少两个留出市场未明显超过发生率基线；停止，不训练概率事件头或条件幅度分布模型。"
        ),
    }
    report = {
        "experiment": "positive_spike_day_lomo_audit_v1",
        "status": "complete",
        "markets": list(MARKETS),
        "uses_ercot": False,
        "ercot_files_read": [],
        "test_2025_read": False,
        "model": {
            "type": "StandardScaler + L1 LogisticRegression",
            "solver": "liblinear", "C": 0.1, "seed": 20261020,
            "class_reweighting": False, "market_equal_weighting": True,
            "features": list(DAY_FEATURE_NAMES),
        },
        "event_definition": "delivery day max DA > node 2020-2021 hourly DA q99",
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
    _write_calibration_plot(report)
    _write_markdown(report)
    manifest_paths = [OUTPUT / name for name in (
        "summary.json", "summary.md", "calibration.csv", "coefficients.csv",
        "calibration_curve.png",
    )]
    (OUTPUT / "completed_outputs.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "positive_spike_day_audit_complete", "gate": gate,
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "runtime_seconds": report["runtime_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
