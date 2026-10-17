#!/usr/bin/env python
"""写入或核验最终四市场训练与2025一次性测试的机器可读预注册。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402


CONFIGS = {
    "da": "configs/decision_aware/allmarket_final_da_mae_v1.yaml",
    "rt_at_da": "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml",
    "rt": "configs/decision_aware/allmarket_final_rt_mae_v1.yaml",
}
PLAN_ROOTS = {
    "da": "data/checkpoints/universal_lomo_holdout_ercot_mae_v1",
    "rt_at_da": "data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1",
    "rt": "data/checkpoints/universal_lomo_holdout_ercot_mae_v1",
}
CODE_FILES = (
    "src/decision_aware/global_data.py",
    "scripts/decision_aware/train_global_forecaster.py",
    "scripts/decision_aware/evaluate_universal_holdout.py",
    "scripts/decision_aware/prepare_allmarket_locked_2025.py",
    "scripts/decision_aware/evaluate_global_continuous_mpc.py",
    "scripts/decision_aware/run_cloud_allmarket_final_train_v1.sh",
    "scripts/decision_aware/run_cloud_allmarket_locked_2025_v1.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan_path(task: str, seed: int) -> str:
    return f"{PLAN_ROOTS[task]}/{task}_inner_seed{seed}/best.pt"


def build_protocol() -> dict:
    files = [*CONFIGS.values(), *CODE_FILES]
    plans = []
    for task in CONFIGS:
        cfg = PilotConfig.from_yaml(ROOT / CONFIGS[task])
        if tuple(cfg.global_training_markets) != ("CAISO", "ERCOT", "NYISO", "PJM"):
            raise ValueError(f"{task}最终配置必须包含四市场且顺序冻结")
        if cfg.global_use_entity_embeddings or cfg.global_num_nodes or cfg.global_num_markets:
            raise ValueError(f"{task}最终配置禁止实体ID")
        if cfg.global_normalization_mode != "context_robust":
            raise ValueError(f"{task}最终配置必须使用context_robust")
        if cfg.global_use_critical_event_weighting:
            raise ValueError(f"{task}最终配置不得晋级seed-0尾部加权试验")
        expected_loss = "pseudo_huber" if task == "rt_at_da" else "mae"
        if cfg.global_training_loss != expected_loss:
            raise ValueError(f"{task}损失必须冻结为{expected_loss}")
        for seed in (0, 1, 2):
            relative = _plan_path(task, seed)
            path = ROOT / relative
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("task") != task or checkpoint.get("phase") != "inner":
                raise ValueError(f"{relative}不是同任务inner训练计划")
            if int(checkpoint.get("config", {}).get("seed", -1)) != seed:
                raise ValueError(f"{relative} seed错误")
            plans.append({
                "task": task,
                "seed": seed,
                "path": relative,
                "sha256": _sha256(path),
                "final_epoch": int(checkpoint["best_epoch"]),
                "learning_rate_trajectory_source": "checkpoint_history",
            })
            files.append(relative)
    return {
        "protocol_version": "allmarket_final_locked_2025_v1",
        "frozen_at_local_date": "2026-10-17",
        "model": {
            "tasks": ["da", "rt_at_da", "rt"],
            "fusion": {"da": "F1", "rt_at_da": "F0", "rt": "F1"},
            "loss": {"da": "mae", "rt_at_da": "pseudo_huber", "rt": "mae"},
            "tail_weighting_promoted": False,
            "tail_pilot_role": "reported single-factor ablation only",
            "uses_node_or_market_id": False,
            "normalization": "context_robust",
        },
        "training": {
            "markets": ["CAISO", "ERCOT", "NYISO", "PJM"],
            "years": [2020, 2021, 2022, 2023, 2024],
            "sampling": "market_node_balanced_16350_per_market_per_epoch",
            "seeds": [0, 1, 2],
            "ensemble": "arithmetic_mean_equal_weight_seed_0_1_2",
            "epoch_and_lr_selection": "frozen_from_same_seed_2023_inner_plan",
            "2025_read_during_training": False,
            "plans": plans,
        },
        "locked_test": {
            "year": 2025,
            "run_once": True,
            "selection_allowed": False,
            "no_post_test_model_or_threshold_changes": True,
            "markets": ["CAISO", "ERCOT", "NYISO", "PJM"],
            "primary_forecast_aggregation": "node_equal_then_market_macro",
            "ercot_profit_contract": "FORMAL_EXPERIMENT_CONTRACT",
            "tail_threshold_source": "2020-2024_training_only",
            "tail_slices": [
                "negative_price", "highest_1pct", "lowest_1pct",
                "abs_da_rt_spread_top_1pct", "abs_da_rt_spread_top_5pct",
                "ordinary",
            ],
            "report_regardless_of_outcome": True,
        },
        "baselines": {
            "timesfm3": "official_frozen_same_168h_and_cutoff",
            "timesfm25": "all_market_2020_2024_source_supervised_lora_seed0",
            "xgboost": "all_market_pooled_and_per_node_supervised_2020_2024",
            "lear": "day_level_da_and_rt_at_da_only",
        },
        "file_sha256": {
            relative: _sha256(ROOT / relative) for relative in sorted(set(files))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/results/allmarket_final_2025_preregistration_v1/protocol.json",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    current = build_protocol()
    if args.verify:
        frozen = json.loads(output.read_text(encoding="utf-8"))
        if frozen != current:
            raise ValueError("最终2025预注册与当前配置、训练计划或代码哈希不一致")
        print(json.dumps({"event": "allmarket_2025_protocol_verified"}))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".sha256").write_text(
        f"{_sha256(output)}  {output.relative_to(ROOT).as_posix()}\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "allmarket_2025_protocol_frozen",
        "path": output.relative_to(ROOT).as_posix(),
        "sha256": _sha256(output),
    }))


if __name__ == "__main__":
    main()
