"""用已有纯Huber checkpoint在偏差罚金验证口径下重选三路来源。

该脚本不重训模型。DA和RT-at-DA按固定罚金策略下的完整
双结算验证日均净收益成对选择；若策略选择为RT K=0，滚动RT
不影响收益，因此只按验证MAE选一个可复现的预测checkpoint。
"""
from __future__ import annotations

import argparse
import hashlib
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
    MODEL_TYPES,
    _config_from_dict,
    _load_source_checkpoint,
    _predict_model,
    _task_datasets,
    _truth_and_contract,
)


OUTPUT_KEYS = {"da": "p_da", "rt_at_da": "p_rt_at_da", "rt": "p_rt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_paths(root: str) -> list[Path]:
    paths = sorted(Path(root).glob("*/pilot_LZ_LCRA_best_full_dual.pt"))
    if not paths:
        raise FileNotFoundError(f"未找到候选checkpoint: {root}")
    return paths


@torch.no_grad()
def _predict_candidates(
    task: str,
    paths: list[Path],
    dataset,
    indices: list[int],
    device: torch.device,
) -> dict[str, dict]:
    values = {}
    batch_size = 128 if task == "rt" else 64
    for path in paths:
        checkpoint = _load_source_checkpoint(path)
        model = MODEL_TYPES[task](_config_from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        prediction = _predict_model(
            model, dataset, indices, OUTPUT_KEYS[task], batch_size, device,
            device.type == "cuda",
        )
        label = path.parent.name
        values[label] = {
            "path": str(path),
            "sha256": _sha256(path),
            "prediction": prediction.astype(np.float32),
        }
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({
            "event": "candidate_predicted", "task": task, "candidate": label
        }, ensure_ascii=False), flush=True)
    return values


def _daily_metrics(result: dict) -> dict:
    daily = result["daily_revenue"].numpy().astype(np.float64)
    tail_count = max(1, math.ceil(0.10 * len(daily)))
    cumulative = np.cumsum(daily)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    return {
        "mean_daily_revenue": float(daily.mean()),
        "positive_day_rate": float(np.mean(daily > 0.0)),
        "lower_tail_10pct_mean": float(np.sort(daily)[:tail_count].mean()),
        "max_drawdown": float(np.max(peaks - cumulative, initial=0.0)),
        "mean_daily_deviation_penalty": float(result["deviation_penalty"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/decision_aware/joint_decision_aware_v3_penalty.yaml",
    )
    parser.add_argument(
        "--output",
        default="data/results/penalty_source_checkpoint_selection_v1.json",
    )
    parser.add_argument("--max-validation-days", type=int, default=None)
    args = parser.parse_args()

    cfg = PilotConfig.from_yaml(args.config)
    if not cfg.use_deviation_penalty:
        raise ValueError("必须在偏差罚金口径下重选checkpoint")
    if cfg.rt_topk_k_charge != 0 or cfg.rt_topk_k_discharge != 0:
        raise ValueError("本脚本的RT按MAE打破收益平局规则只适用于RT K=0")

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

    zero_rt = np.zeros(
        (len(contract["dates"]), 24, cfg.horizon_rt), dtype=np.float32
    )
    pair_rows = []
    if cfg.joint_da_signal_mode == "da_only":
        # DA-only策略的动作与RT-at-DA预测无关；用一个零占位即可评估DA，
        # 避免制造108个其实完全相同的“组合”结果。
        rt_da_items = [(None, {"prediction": np.zeros_like(
            next(iter(candidates["rt_at_da"].values()))["prediction"]
        )})]
    else:
        rt_da_items = list(candidates["rt_at_da"].items())
    for da_name, da_item in candidates["da"].items():
        for rt_da_name, rt_da_item in rt_da_items:
            result = settle_joint_episode(
                torch.from_numpy(da_item["prediction"]),
                torch.from_numpy(rt_da_item["prediction"]),
                torch.from_numpy(zero_rt),
                torch.from_numpy(contract["true_da"]),
                torch.from_numpy(contract["true_rt"]),
                torch.from_numpy(contract["starts_new_segment"]),
                cfg,
                coordination_mode=cfg.joint_coordination_mode,
            )
            pair_rows.append({
                "da": da_name,
                "rt_at_da": rt_da_name,
                **_daily_metrics(result),
            })
    pair_rows.sort(key=lambda row: row["mean_daily_revenue"], reverse=True)

    true_rt_windows = contract["true_rt_windows"]
    rt_da_rows = []
    true_rt_day = contract["true_rt"]
    for name, item in candidates["rt_at_da"].items():
        error = (
            item["prediction"].astype(np.float64)
            - true_rt_day.astype(np.float64)
        )
        rt_da_rows.append({
            "rt_at_da": name,
            "validation_mae": float(np.mean(np.abs(error))),
            "validation_rmse": float(np.sqrt(np.mean(np.square(error)))),
        })
    rt_da_rows.sort(key=lambda row: row["validation_mae"])

    rt_rows = []
    for name, item in candidates["rt"].items():
        prediction = item["prediction"].reshape(
            len(contract["dates"]), 24, -1
        )
        error = prediction.astype(np.float64) - true_rt_windows.astype(np.float64)
        rt_rows.append({
            "rt": name,
            "validation_mae": float(np.mean(np.abs(error))),
            "validation_rmse": float(np.sqrt(np.mean(np.square(error)))),
        })
    rt_rows.sort(key=lambda row: row["validation_mae"])

    selected_pair = pair_rows[0]
    selected_rt_da = (
        rt_da_rows[0]
        if cfg.joint_da_signal_mode == "da_only"
        else next(
            row for row in rt_da_rows
            if row["rt_at_da"] == selected_pair["rt_at_da"]
        )
    )
    selected_rt = rt_rows[0]
    selected_paths = {
        "da": candidates["da"][selected_pair["da"]]["path"],
        "rt_at_da": candidates["rt_at_da"][selected_rt_da["rt_at_da"]]["path"],
        "rt": candidates["rt"][selected_rt["rt"]]["path"],
    }
    report = {
        "contract": {
            "validation_days": len(contract["dates"]),
            "selection_metric_da_pair": "validation mean daily net revenue",
            "selection_metric_rt_at_da_tiebreak": (
                "validation daily-horizon MAE when DA signal is DA-only"
            ),
            "selection_metric_rt_tiebreak": "validation all-horizon MAE",
            "coordination_mode": cfg.joint_coordination_mode,
            "da_signal_mode": cfg.joint_da_signal_mode,
            "da_k": [cfg.topk_k_charge, cfg.topk_k_discharge],
            "rt_k": [cfg.rt_topk_k_charge, cfg.rt_topk_k_discharge],
            "spread_threshold": cfg.resolved_spread_threshold,
            "deviation_penalty_enabled": True,
            "report_only_used": False,
            "warning": (
                "现有checkpoint各自仍是无罚金收益选出的epoch；"
                "本步只是对已保存checkpoint的低成本预筛选。"
            ),
        },
        "candidate_counts": {
            task: len(items) for task, items in candidates.items()
        },
        "selected": {
            "da_rt_at_da_pair": selected_pair,
            "rt_at_da_forecast_tiebreak": selected_rt_da,
            "rt_forecast_tiebreak": selected_rt,
            "paths": selected_paths,
        },
        "pair_ranking": pair_rows,
        "rt_at_da_forecast_ranking": rt_da_rows,
        "rt_forecast_ranking": rt_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "saved", "path": str(output), "selected": report["selected"]
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
