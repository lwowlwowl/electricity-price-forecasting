"""在固定Transformer-Huber价格预测上选择偏差罚金场景的HardTopK合同。

该脚本不训练模型；它只使用验证集对固定策略参数进行网格比较，
以避免在策略合同未冻结时盲目运行耗时的decision-aware训练。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.joint_system import settle_joint_episode  # noqa: E402
from scripts.decision_aware.compare_joint_system_baselines import (  # noqa: E402
    _build_transformers,
    _load_source_checkpoint,
    _predict_transformer_system,
    _truth_and_contract,
)


def _metrics(predictions: dict, contract: dict, cfg: PilotConfig) -> dict:
    result = settle_joint_episode(
        torch.from_numpy(predictions["da"]),
        torch.from_numpy(predictions["rt_at_da"]),
        torch.from_numpy(predictions["rt"]),
        torch.from_numpy(contract["true_da"]),
        torch.from_numpy(contract["true_rt"]),
        torch.from_numpy(contract["starts_new_segment"]),
        cfg,
        coordination_mode=cfg.joint_coordination_mode,
    )
    daily = result["daily_revenue"].numpy().astype(np.float64)
    tail_count = max(1, math.ceil(0.10 * len(daily)))
    cumulative = np.cumsum(daily)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    return {
        "mean_daily_revenue": float(daily.mean()),
        "positive_day_rate": float(np.mean(daily > 0.0)),
        "lower_tail_10pct_mean": float(np.sort(daily)[:tail_count].mean()),
        "max_drawdown": float(np.max(peaks - cumulative, initial=0.0)),
        "mean_daily_deviation_penalty": float(
            result["deviation_penalty"].mean()
        ),
        "mean_daily_da_leg": float(result["da_leg"].mean()),
        "mean_daily_rt_deviation_leg": float(
            result["rt_deviation_leg"].mean()
        ),
        "mean_daily_degradation_cost": float(
            result["degradation_cost"].mean()
        ),
        "days": int(len(daily)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/decision_aware/joint_decision_aware_v3_penalty.yaml",
    )
    parser.add_argument(
        "--output",
        default="data/results/penalty_policy_selection_v1.json",
    )
    parser.add_argument("--max-validation-days", type=int, default=None)
    parser.add_argument(
        "--kappa", type=float, default=None,
        help="可选：覆盖配置中的单位实际吞吐成本，用于10/25/50敏感性。",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if args.kappa is not None:
        if args.kappa < 0:
            raise ValueError("kappa不能为负")
        cfg.bess_kappa = float(args.kappa)
    if not cfg.use_deviation_penalty:
        raise ValueError("策略选择必须使用use_deviation_penalty=true")
    source_paths = {
        "da": cfg.joint_da_checkpoint,
        "rt_at_da": cfg.joint_rt_at_da_checkpoint,
        "rt": cfg.joint_rt_checkpoint,
    }
    checkpoints = {
        task: _load_source_checkpoint(path) for task, path in source_paths.items()
    }
    _, val_dataset, _, _ = build_joint_datasets(
        cfg, norm_stats=checkpoints["da"]["norm_stats"]
    )
    contract = _truth_and_contract(val_dataset, args.max_validation_days)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _build_transformers(checkpoints, device)
    predictions = _predict_transformer_system(
        models, val_dataset, contract, device
    )
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    thresholds = [-1.0, 0.0, 10.0, 25.0, 50.0]
    for da_signal_mode in ("spread", "da_only"):
        for mode in ("follow_da", "plan_track_topk"):
            rt_values = [0] if mode == "follow_da" else [0, 1, 2]
            # K=0 是必需的“完全不交易”安全基线。高退化成本或强罚金下，
            # 如果所有交易候选都亏损，策略选择必须允许返回 0 收益，不能
            # 被迫从若干负收益策略中挑一个“最好”的。
            for da_k in (0, 1, 2, 3, 4):
                for rt_k in rt_values:
                    for threshold in thresholds:
                        candidate = copy.deepcopy(cfg)
                        candidate.joint_da_signal_mode = da_signal_mode
                        candidate.joint_coordination_mode = mode
                        candidate.topk_k_charge = da_k
                        candidate.topk_k_discharge = da_k
                        candidate.rt_topk_k_charge = rt_k
                        candidate.rt_topk_k_discharge = rt_k
                        candidate.topk_spread_threshold = threshold
                        row = {
                            "da_signal_mode": da_signal_mode,
                            "coordination_mode": mode,
                            "da_k_charge": da_k,
                            "da_k_discharge": da_k,
                            "rt_k_charge": rt_k,
                            "rt_k_discharge": rt_k,
                            "threshold_config": threshold,
                            "threshold_used": candidate.resolved_spread_threshold,
                            **_metrics(predictions, contract, candidate),
                        }
                        rows.append(row)
                        if not args.quiet:
                            print(json.dumps(row, ensure_ascii=False), flush=True)

    # 保留旧rt_only作为诊断，不纳入w10 §4.3的候选集。
    diagnostic = copy.deepcopy(cfg)
    diagnostic.joint_coordination_mode = "rt_only"
    diagnostic_row = {
        "da_signal_mode": diagnostic.joint_da_signal_mode,
        "coordination_mode": "rt_only_diagnostic_not_candidate",
        "da_k_charge": diagnostic.topk_k_charge,
        "da_k_discharge": diagnostic.topk_k_discharge,
        "rt_k_charge": diagnostic.rt_topk_k_charge,
        "rt_k_discharge": diagnostic.rt_topk_k_discharge,
        "threshold_config": diagnostic.topk_spread_threshold,
        "threshold_used": diagnostic.resolved_spread_threshold,
        **_metrics(predictions, contract, diagnostic),
    }
    ranking = sorted(
        rows, key=lambda row: row["mean_daily_revenue"], reverse=True
    )
    best_trading = next(row for row in ranking if row["da_k_charge"] > 0)
    best_rt_active = next(row for row in ranking if row["rt_k_charge"] > 0)
    best_pdf_main_shape = next(
        row for row in ranking
        if row["da_signal_mode"] == "spread"
        and row["coordination_mode"] == "plan_track_topk"
        and row["rt_k_charge"] > 0
    )
    report = {
        "contract": {
            "market": cfg.market,
            "node": cfg.node,
            "validation_days": len(contract["dates"]),
            "validation_first": contract["dates"][0],
            "validation_last": contract["dates"][-1],
            "deviation_penalty": "3% of |u_DA| tolerance; 2*|p_RT| on excess",
            "bess_kappa_usd_per_mwh": float(cfg.bess_kappa),
            "selection_metric": "validation mean daily net revenue",
            "reference_system": "frozen Transformer-Huber source triplet",
            "report_only_used": False,
        },
        "source_checkpoints": source_paths,
        "best": ranking[0],
        "best_trading": best_trading,
        "best_rt_active": best_rt_active,
        "best_pdf_main_shape": best_pdf_main_shape,
        "ranking": ranking,
        "rt_only_diagnostic": diagnostic_row,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "saved", "path": str(output), "best": ranking[0]
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
