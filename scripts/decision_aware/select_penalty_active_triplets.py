"""在固定的“偏差罚金 + RT TopK主动”口径下重选现有三路checkpoint。

这个脚本只重新评估已保存的纯Huber模型，不会训练、不会查看
report-only日期，也不会覆盖已有JSON。DA、RT-at-DA和滚动RT预测均
只生成一次并缓存在内存中，随后对三路候选的笛卡尔积统一结算。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from decision_aware.joint_system import settle_joint_episode  # noqa: E402
from scripts.decision_aware.compare_joint_system_baselines import (  # noqa: E402
    _load_source_checkpoint,
    _task_datasets,
    _truth_and_contract,
)
from scripts.decision_aware.select_penalty_source_checkpoints import (  # noqa: E402
    _candidate_paths,
    _predict_candidates,
)


TASKS = ("da", "rt_at_da", "rt")


def _validate_active_contract(cfg: PilotConfig) -> None:
    if not cfg.use_deviation_penalty:
        raise ValueError("必须启用use_deviation_penalty")
    if cfg.joint_coordination_mode not in {"plan_track_topk", "rt_only"}:
        raise ValueError(
            "RT TopK主动重选只支持plan_track_topk或rt_only协调口径"
        )
    if cfg.rt_topk_k_charge <= 0 or cfg.rt_topk_k_discharge <= 0:
        raise ValueError("RT TopK主动重选要求充、放电K都大于0")
    if cfg.joint_da_signal_mode not in {"spread", "da_only"}:
        raise ValueError("仅支持spread或da_only的DA决策信号")


def _as_settlement_tensors(
    candidates: dict[str, dict[str, dict]],
    contract: dict,
    cfg: PilotConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """把预测和真值只转换一次，避免在972次结算中重复复制。"""
    lower, upper = (float(value) for value in cfg.pred_clamp)
    cached = {}
    for task in TASKS:
        cached[task] = {}
        for name, item in candidates[task].items():
            prediction = np.clip(
                np.asarray(item["prediction"], dtype=np.float32), lower, upper
            )
            if task == "rt":
                prediction = prediction.reshape(
                    len(contract["dates"]), 24, cfg.horizon_rt
                )
            cached[task][name] = torch.from_numpy(prediction)
    truth = {
        "da": torch.from_numpy(contract["true_da"]),
        "rt": torch.from_numpy(contract["true_rt"]),
        "starts_new_segment": torch.from_numpy(
            contract["starts_new_segment"]
        ),
    }
    return cached, truth


def _triplet_metrics(result: dict) -> dict:
    daily = result["daily_revenue"].detach().cpu().numpy().astype(np.float64)
    if daily.ndim != 1 or daily.size == 0 or not np.all(np.isfinite(daily)):
        raise ValueError("三元组结算产生了空或非有限的逐日收益")
    tail_count = max(1, math.ceil(0.10 * len(daily)))
    cumulative = np.cumsum(daily)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    return {
        "mean_daily_revenue": float(daily.mean()),
        "total_net_revenue": float(daily.sum()),
        "positive_day_rate": float(np.mean(daily > 0.0)),
        "lower_tail_10pct_mean": float(np.sort(daily)[:tail_count].mean()),
        "max_drawdown": float(np.max(peaks - cumulative, initial=0.0)),
        "mean_daily_da_leg": float(result["da_leg"].mean()),
        "mean_daily_rt_deviation_leg": float(
            result["rt_deviation_leg"].mean()
        ),
        "mean_daily_degradation_cost": float(
            result["degradation_cost"].mean()
        ),
        "mean_daily_deviation_penalty": float(
            result["deviation_penalty"].mean()
        ),
        "final_plan_soc_mwh": float(result["plan_soc_end_mwh"][-1]),
        "final_actual_soc_mwh": float(result["actual_soc_end_mwh"][-1]),
    }


def evaluate_triplet_grid(
    candidates: dict[str, dict[str, dict]],
    contract: dict,
    cfg: PilotConfig,
    *,
    settle_fn: Callable = settle_joint_episode,
    progress_every: int = 50,
) -> list[dict]:
    """在固定策略下评估全部DA×RT-at-DA×RT三元组。"""
    _validate_active_contract(cfg)
    missing = [task for task in TASKS if not candidates.get(task)]
    if missing:
        raise ValueError(f"候选预测为空: {missing}")
    cached, truth = _as_settlement_tensors(candidates, contract, cfg)
    total = math.prod(len(cached[task]) for task in TASKS)
    rows: list[dict] = []
    completed = 0
    started = time.time()
    for da_name, p_da in cached["da"].items():
        for rt_da_name, p_rt_at_da in cached["rt_at_da"].items():
            for rt_name, p_rt in cached["rt"].items():
                result = settle_fn(
                    p_da,
                    p_rt_at_da,
                    p_rt,
                    truth["da"],
                    truth["rt"],
                    truth["starts_new_segment"],
                    cfg,
                    coordination_mode=cfg.joint_coordination_mode,
                )
                rows.append({
                    "da": da_name,
                    "rt_at_da": rt_da_name,
                    "rt": rt_name,
                    **_triplet_metrics(result),
                })
                completed += 1
                if progress_every > 0 and (
                    completed % progress_every == 0 or completed == total
                ):
                    print(json.dumps({
                        "event": "triplet_progress",
                        "completed": completed,
                        "total": total,
                        "elapsed_seconds": round(time.time() - started, 2),
                    }, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: (
        -row["mean_daily_revenue"],
        row["da"], row["rt_at_da"], row["rt"],
    ))
    return rows


def _candidate_manifest(candidates: dict[str, dict[str, dict]]) -> dict:
    return {
        task: [
            {
                "name": name,
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for name, item in values.items()
        ]
        for task, values in candidates.items()
    }


def _write_new_json(path: Path, value: dict) -> None:
    """用x模式写入：目标已存在时拒绝覆盖历史结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/decision_aware/"
            "joint_decision_aware_v3_penalty.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/results/penalty_active_triplet_selection_v1.json",
    )
    parser.add_argument("--max-validation-days", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    _validate_active_contract(cfg)
    if args.max_validation_days is not None and args.max_validation_days < 1:
        raise ValueError("max-validation-days必须大于等于1")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有结果: {output}")

    reference = _load_source_checkpoint(cfg.joint_da_checkpoint)
    _, val_dataset, _, _ = build_joint_datasets(
        cfg, norm_stats=reference["norm_stats"]
    )
    contract = _truth_and_contract(val_dataset, args.max_validation_days)
    datasets = _task_datasets(val_dataset)
    indices = {
        "da": contract["da_indices"],
        "rt_at_da": contract["rt_at_da_indices"],
        "rt": contract["rt_indices"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 复用零RT脚本的预测缓存逻辑：每个checkpoint只前向一次，
    # 随后所有三元组只读内存中的numpy/tensor。
    candidates = {
        "da": _predict_candidates(
            "da", _candidate_paths("data/checkpoints/da_fusion_ablation"),
            datasets["da"], indices["da"], device,
        ),
        "rt_at_da": _predict_candidates(
            "rt_at_da",
            _candidate_paths("data/checkpoints/rt_at_da_fusion_ablation"),
            datasets["rt_at_da"], indices["rt_at_da"], device,
        ),
        "rt": _predict_candidates(
            "rt", _candidate_paths("data/checkpoints/rt_fusion_ablation"),
            datasets["rt"], indices["rt"], device,
        ),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ranking = evaluate_triplet_grid(
        candidates, contract, cfg, progress_every=args.progress_every
    )
    selected = ranking[0]
    selected_paths = {
        task: candidates[task][selected[task]]["path"] for task in TASKS
    }
    selected_hashes = {
        task: candidates[task][selected[task]]["sha256"] for task in TASKS
    }
    report = {
        "contract": {
            "market": cfg.market,
            "node": cfg.node,
            "validation_days": len(contract["dates"]),
            "validation_first": contract["dates"][0],
            "validation_last": contract["dates"][-1],
            "selection_metric": "validation mean daily net revenue",
            "settlement": "complete DA/RT dual settlement",
            "deviation_penalty": "3% of |u_DA| tolerance; 2*|p_RT| on excess",
            "deviation_penalty_enabled": bool(cfg.use_deviation_penalty),
            "bess_kappa_usd_per_mwh": float(cfg.bess_kappa),
            "da_signal_mode": cfg.joint_da_signal_mode,
            "coordination_mode": cfg.joint_coordination_mode,
            "da_k": [cfg.topk_k_charge, cfg.topk_k_discharge],
            "rt_k": [cfg.rt_topk_k_charge, cfg.rt_topk_k_discharge],
            "spread_threshold_config": float(cfg.topk_spread_threshold),
            "spread_threshold_used": float(cfg.resolved_spread_threshold),
            "prediction_clamp": [float(value) for value in cfg.pred_clamp],
            "continuous_soc_within_validation_segments": True,
            "terminal_soc_monetized": False,
            "report_only_used": False,
            "checkpoint_training_or_epoch_selection_changed": False,
            "warning": (
                "这是现有checkpoint的罚金口径低成本重选；"
                "不等于已按该罚金口径重新训练或逐epoch选模。"
            ),
        },
        "candidate_counts": {
            task: len(values) for task, values in candidates.items()
        },
        "triplet_count": len(ranking),
        "candidate_manifest": _candidate_manifest(candidates),
        "selected": {
            **selected,
            "paths": selected_paths,
            "sha256": selected_hashes,
        },
        "ranking": ranking,
    }
    _write_new_json(output, report)
    print(json.dumps({
        "event": "saved",
        "path": str(output),
        "selected": report["selected"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
