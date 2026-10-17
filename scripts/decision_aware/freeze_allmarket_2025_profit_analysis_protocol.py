#!/usr/bin/env python
"""冻结/核验2025一次性测试的盈利主终点与论文解释规则。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/decision_aware/allmarket_locked_2025_profit_analysis_v1.yaml"
MAIN_PROTOCOL = "data/results/allmarket_final_2025_preregistration_v1/protocol.json"
BASELINE_PROTOCOL = "data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json"
OPEN_MARKER = "data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json"
EVALUATION_FILES = (
    "scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh",
    "scripts/decision_aware/summarize_allmarket_locked_2025.py",
    "scripts/decision_aware/audit_allmarket_locked_2025_outputs.py",
    "scripts/decision_aware/audit_allmarket_locked_2025_profit_claim.py",
    "scripts/documentation/generate_multinode_technical_report.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> dict:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    if config.get("selection_uses_2025") is not False:
        raise ValueError("盈利分析协议必须禁止使用2025选模")
    endpoint = config.get("primary_economic_endpoint", {})
    expected = {
        "market": "ERCOT",
        "system": "transformer_fixed_seed012_ensemble",
        "comparator": "pooled_allmarket_xgboost",
        "metric": "macro_mean_daily_net_revenue_usd_per_node_day",
        "uncertainty": "paired_7day_block_bootstrap_95ci",
        "stable_superiority_rule": "mean_difference_gt_0_and_ci_lower_gt_0",
    }
    mismatches = {
        key: (endpoint.get(key), value)
        for key, value in expected.items()
        if endpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"盈利主终点配置不匹配: {mismatches}")
    claim = config.get("claim_policy", {})
    if claim.get("mae_cannot_override_primary_economic_endpoint") is not True:
        raise ValueError("必须明确禁止MAE覆盖盈利主结论")
    if claim.get("no_model_or_threshold_change_after_opening") is not True:
        raise ValueError("必须禁止打开2025后修改模型或阈值")
    parents = (MAIN_PROTOCOL, BASELINE_PROTOCOL)
    for relative in (*parents, *EVALUATION_FILES):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    return {
        **config,
        "parent_protocol_sha256": {
            relative: sha256(ROOT / relative) for relative in parents
        },
        "evaluation_file_sha256": {
            relative: sha256(ROOT / relative) for relative in EVALUATION_FILES
        },
        "amendment_scope": (
            "pre_2025_reporting_and_claim_rule_only; no model, checkpoint, "
            "training data, feature, threshold, policy, or settlement change"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "data/results/allmarket_final_2025_profit_analysis_preregistration_v1/"
            "protocol.json"
        ),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    current = build()
    if args.verify:
        frozen = json.loads(output.read_text(encoding="utf-8"))
        if frozen != current:
            raise ValueError("2025盈利分析代码或主判据已偏离冻结协议")
        if (ROOT / OPEN_MARKER).exists():
            raise ValueError("2025已经打开，不能重新冻结盈利主判据")
        print(json.dumps({"event": "allmarket_2025_profit_analysis_protocol_verified"}))
        return
    if (ROOT / OPEN_MARKER).exists():
        raise ValueError("2025已经打开，拒绝新增或覆盖盈利分析预注册")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".sha256").write_text(
        f"{sha256(output)}  {output.relative_to(ROOT).as_posix()}\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "allmarket_2025_profit_analysis_protocol_frozen",
        "sha256": sha256(output),
    }))


if __name__ == "__main__":
    main()

