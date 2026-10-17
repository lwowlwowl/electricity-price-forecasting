#!/usr/bin/env python
"""审计论文补实验的本地回收产物，并生成机器可读/可读汇总。

本脚本不下载文件、不训练模型，也不修改原始结果。它用于云端结果回收后立即检查：

* ERCOT 三任务 seed 1/2 是否齐全，并与冻结 seed 0 使用同一真值；
* pooled LOMO XGBoost 的四市场 × 三任务是否齐全且没有目标市场标签泄漏；
* 当前结构、全 F0、全 F1 的 ERCOT 消融是否齐全；
* 所有归档均未读取 2025，且预测输入没有 node_id/market_id；
* checkpoint 哈希（文件存在时）和 JSON/NPZ 元数据是否一致；
* 两份 ERCOT 结算报告是否包含预期同构系统。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MARKETS = ("caiso", "ercot", "nyiso", "pjm")
TASKS = ("da", "rt_at_da", "rt")
EXPECTED_TRAINING_MARKETS = {
    holdout: sorted(set(market.upper() for market in MARKETS) - {holdout.upper()})
    for holdout in MARKETS
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in archive.files:
        raise ValueError("NPZ 缺少 metadata")
    raw = archive["metadata"]
    if np.ndim(raw) != 0:
        raise ValueError("metadata 必须是标量 JSON")
    return json.loads(str(raw.item()))


def _year_strings(dates: np.ndarray) -> set[str]:
    return {str(item)[:4] for item in np.asarray(dates).reshape(-1)}


def _archive_record(
    path: Path,
    *,
    holdout: str,
    task: str,
    protocol: str,
    require_no_target_labels: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": path.as_posix(), "exists": False}, [f"缺少 {path}"]
    with np.load(path, allow_pickle=False) as archive:
        required = {"nodes", "dates", "prediction", "target", "metadata"}
        missing = sorted(required - set(archive.files))
        if missing:
            return {"path": path.as_posix(), "exists": True}, [
                f"{path} 缺少字段 {missing}"
            ]
        nodes = archive["nodes"]
        dates = archive["dates"]
        prediction = archive["prediction"]
        target = archive["target"]
        metadata = _metadata(archive)
        if prediction.shape != target.shape:
            errors.append(f"{path} prediction/target 形状不一致")
        if prediction.shape[0] != len(nodes):
            errors.append(f"{path} 第一维与节点数不一致")
        years = _year_strings(dates)
        if "2025" in years:
            errors.append(f"{path} 读取了锁定的 2025")
        if str(metadata.get("protocol")) != protocol:
            errors.append(f"{path} protocol 不符: {metadata.get('protocol')}")
        if str(metadata.get("holdout_market", "")).lower() != holdout:
            errors.append(f"{path} 留出市场不符")
        if str(metadata.get("task", "")) != task:
            errors.append(f"{path} 任务不符")
        if metadata.get("uses_node_id_for_prediction") is not False:
            errors.append(f"{path} 未声明禁用 node_id")
        if metadata.get("uses_market_id_for_prediction") is not False:
            errors.append(f"{path} 未声明禁用 market_id")
        training_markets = sorted(str(item).upper() for item in metadata.get("training_markets", []))
        if training_markets != EXPECTED_TRAINING_MARKETS[holdout]:
            errors.append(f"{path} 训练市场不等于另外三个源市场: {training_markets}")
        if require_no_target_labels and metadata.get("uses_target_market_training_labels") is not False:
            errors.append(f"{path} 未明确禁止目标市场训练标签")
        checkpoint = metadata.get("checkpoint")
        checkpoint_hash_ok: bool | None = None
        if checkpoint:
            checkpoint_path = ROOT / str(checkpoint)
            if checkpoint_path.is_file():
                checkpoint_hash_ok = _sha256(checkpoint_path) == metadata.get("checkpoint_sha256")
                if not checkpoint_hash_ok:
                    errors.append(f"{path} checkpoint SHA-256 不一致")
        mae = float(np.mean(np.mean(np.abs(prediction - target), axis=tuple(range(1, target.ndim)))))
        record = {
            "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix(),
            "exists": True,
            "sha256": _sha256(path),
            "holdout": holdout.upper(),
            "task": task,
            "node_count": int(len(nodes)),
            "dates": int(len(dates)),
            "years": sorted(years),
            "node_macro_mae_recomputed": mae,
            "node_macro_mae_declared": metadata.get("node_macro_mae"),
            "checkpoint_hash_ok": checkpoint_hash_ok,
        }
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in ("protocol", "holdout_market", "task", "checkpoint_sha256"):
            if sidecar_metadata.get(field) != metadata.get(field):
                errors.append(f"{path} 与 sidecar 的 {field} 不一致")
        record["sidecar_sha256"] = _sha256(sidecar)
    return record, errors


def _settlement_summary(path: Path, expected_families: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": path.as_posix(), "exists": False}, [f"缺少 {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = str(payload.get("split", ""))
    if "2025" in split:
        errors.append(f"{path} 错误使用 2025")
    systems = payload.get("systems", {})
    selected: dict[str, Any] = {}
    for family in expected_families:
        label = f"da_{family}__rt_at_da_{family}__rt_{family}"
        if label not in systems:
            errors.append(f"{path} 缺少同构系统 {label}")
            continue
        system = systems[label]
        selected[family] = {
            "label": label,
            "macro_mean_daily_revenue": system.get("macro_mean_daily_revenue"),
            "bootstrap_95ci": system.get("macro_mean_daily_revenue_7day_block_bootstrap_95ci"),
            "worst_node_mean_daily_revenue": system.get("worst_node_mean_daily_revenue"),
            "node_count": system.get("node_count"),
            "days": system.get("days"),
        }
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix(),
        "exists": True,
        "sha256": _sha256(path),
        "split": split,
        "contract_hash": payload.get("contract_hash"),
        "selected_systems": selected,
    }, errors


def audit() -> dict[str, Any]:
    errors: list[str] = []
    multiseed: list[dict[str, Any]] = []
    for task in TASKS:
        for seed in (1, 2):
            path = ROOT / "data/results/universal_lomo_multiseed_v1" / f"ercot_{task}_inner_seed{seed}.npz"
            record, found = _archive_record(
                path,
                holdout="ercot",
                task=task,
                protocol="leave_one_market_out_zero_shot",
            )
            record["seed"] = seed
            multiseed.append(record)
            errors.extend(found)

    xgboost: list[dict[str, Any]] = []
    for holdout in MARKETS:
        for task in TASKS:
            path = ROOT / "data/results/universal_lomo_xgboost_v1" / f"holdout_{holdout}_{task}_inner_seed0.npz"
            record, found = _archive_record(
                path,
                holdout=holdout,
                task=task,
                protocol="pooled_lomo_xgboost_equal_training_years",
                require_no_target_labels=True,
            )
            xgboost.append(record)
            errors.extend(found)

    ablation_specs = (
        ("all_f0", "da"),
        ("all_f0", "rt"),
        ("all_f1", "rt_at_da"),
    )
    ablation: list[dict[str, Any]] = []
    for family, task in ablation_specs:
        path = ROOT / "data/results/universal_lomo_ercot_ablation_predictions_v1" / f"{family}_{task}_inner_seed0.npz"
        record, found = _archive_record(
            path,
            holdout="ercot",
            task=task,
            protocol="leave_one_market_out_zero_shot",
        )
        record["family"] = family
        ablation.append(record)
        errors.extend(found)

    multiseed_settlement, found = _settlement_summary(
        ROOT / "data/results/universal_lomo_ercot_multiseed_continuous_mpc_v1.json",
        ("seed0", "seed1", "seed2", "ensemble"),
    )
    errors.extend(found)
    ablation_settlement, found = _settlement_summary(
        ROOT / "data/results/universal_lomo_ercot_ablation_continuous_mpc_v1.json",
        ("current", "all_f0", "all_f1"),
    )
    errors.extend(found)

    return {
        "audit": "paper_gap_cloud_artifacts_v1",
        "passed": not errors,
        "locked_2025_opened": False if not any("2025" in item for item in errors) else None,
        "expected": {
            "multiseed_archives": 6,
            "pooled_lomo_xgboost_archives": 12,
            "ablation_new_archives": 3,
        },
        "multiseed": multiseed,
        "pooled_lomo_xgboost": xgboost,
        "ablation": ablation,
        "settlement": {
            "multiseed": multiseed_settlement,
            "ablation": ablation_settlement,
        },
        "errors": errors,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 论文补实验回收审计",
        "",
        f"- 总状态：{'通过' if report['passed'] else '未通过'}",
        f"- 错误数：{len(report['errors'])}",
        "- 2025：保持锁定（仅当下表和错误列表没有相反证据）",
        "",
        "## 预测产物",
        "",
        "| 分组 | 市场/结构 | 任务 | seed | 节点 | MAE | SHA-256 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for group, records in (
        ("多 seed", report["multiseed"]),
        ("pooled XGBoost", report["pooled_lomo_xgboost"]),
        ("结构消融", report["ablation"]),
    ):
        for item in records:
            identity = item.get("family", item.get("holdout", "-"))
            lines.append(
                f"| {group} | {identity} | {item.get('task', '-')} | "
                f"{item.get('seed', 0)} | {item.get('node_count', '-')} | "
                f"{item.get('node_macro_mae_recomputed', float('nan')):.4f} | "
                f"{item.get('sha256', '缺失')} |"
            )
    lines.extend(["", "## ERCOT 同合同净收益", ""])
    for name, settlement in report["settlement"].items():
        lines.extend([f"### {name}", "", "| 系统 | 日均净收益 | 7日块95%区间 |", "|---|---:|---|"])
        for family, item in settlement.get("selected_systems", {}).items():
            lines.append(
                f"| {family} | {item.get('macro_mean_daily_revenue')} | {item.get('bootstrap_95ci')} |"
            )
        lines.append("")
    if report["errors"]:
        lines.extend(["## 未通过项", ""])
        lines.extend(f"- {item}" for item in report["errors"])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/results/paper_gap_artifact_audit_v1/summary.json",
    )
    parser.add_argument(
        "--markdown",
        default="data/results/paper_gap_artifact_audit_v1/summary.md",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit()
    output = ROOT / args.output
    markdown = ROOT / args.markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "errors": len(report["errors"]),
        "output": output.relative_to(ROOT).as_posix(),
        "markdown": markdown.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False))
    if report["errors"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
