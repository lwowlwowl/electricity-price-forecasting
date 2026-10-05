#!/usr/bin/env python
"""只读复评一个已经锁定的联合模型 bundle。

默认只评估 validation。 ``report_only`` 必须同时由 ``--split`` 和
``--confirm-report-only`` 显式开启，避免候选尚未冻结时误看报告集。
模型、source checkpoint 和配置文件均不会被修改；唯一写操作是以独占
方式新建 ``--output`` 指定的 JSON 文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.dataset_joint import build_joint_datasets  # noqa: E402
from scripts.decision_aware.compare_joint_system_baselines import (  # noqa: E402
    _build_transformers,
    _check_norm_stats,
    _load_bundle_source_checkpoints,
    _sha256,
)
from scripts.decision_aware.train_joint_decision_aware import (  # noqa: E402
    _assert_independent,
    _check_contract,
    evaluate,
)


# 这些字段决定评估数据、HardTopK动作、SOC和现金结算。模型结构由bundle
# 自己记录的三个source checkpoint恢复，因此source路径不在此列表内。
EVALUATION_CONTRACT_FIELDS = (
    "market",
    "node",
    "data_version",
    "freq",
    "context_len",
    "horizon_da",
    "horizon_rt",
    "da_issue_hour_local",
    "train_stride",
    "val_stride",
    "eval_stride",
    "split_train_start",
    "split_train_end",
    "split_val_start",
    "split_val_end",
    "split_test_start",
    "split_test_end",
    "bess_power_mw",
    "bess_energy_mwh",
    "bess_eta",
    "bess_init_soc_frac",
    "bess_kappa",
    "bess_soc_min",
    "bess_soc_max",
    "bess_e_cyc",
    "use_deviation_penalty",
    "topk_k_charge",
    "topk_k_discharge",
    "rt_topk_k_charge",
    "rt_topk_k_discharge",
    "topk_spread_threshold",
    "joint_allow_source_kappa_mismatch",
    "joint_da_signal_mode",
    "joint_coordination_mode",
)


def _config_from_dict(values: dict[str, Any]) -> PilotConfig:
    fields = PilotConfig.__dataclass_fields__
    return PilotConfig(**{
        key: value for key, value in values.items() if key in fields
    })


def _same_contract_value(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _validate_evaluation_contract(
    evaluation_cfg: PilotConfig,
    bundle_config: dict[str, Any],
) -> PilotConfig:
    """拒绝用另一个策略/结算合同给已锁定bundle重新贴标签。"""
    if not bundle_config:
        raise ValueError("联合checkpoint缺少experiment_config，无法审计评估合同")
    training_cfg = _config_from_dict(bundle_config)
    mismatches = {}
    for field in EVALUATION_CONTRACT_FIELDS:
        current = getattr(evaluation_cfg, field)
        trained = getattr(training_cfg, field)
        if not _same_contract_value(current, trained):
            mismatches[field] = {"config": current, "checkpoint": trained}
    if mismatches:
        raise ValueError(
            "配置与联合checkpoint的评估合同不一致；请使用训练时锁定的配置: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return training_cfg


def _require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有评估结果: {path}")


def _validate_split_request(split: str, confirm_report_only: bool) -> None:
    if split == "report_only" and not confirm_report_only:
        raise ValueError(
            "report_only仅能在模型、策略和选模规则全部冻结后披露；"
            "若已冻结，请同时传入--confirm-report-only"
        )
    if split == "validation" and confirm_report_only:
        raise ValueError("validation评估不应传入--confirm-report-only")


def _validate_locked_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("kind") == "joint_training_state":
        raise ValueError(
            "latest训练状态不是已锁定的最佳模型；请改用"
            "best_validation_revenue.pt"
        )
    if set(bundle.get("model_states", {})) != {"da", "rt_at_da", "rt"}:
        raise ValueError("联合checkpoint必须包含da、rt_at_da、rt三组model_states")
    if not bundle.get("experiment_config"):
        raise ValueError("联合checkpoint缺少experiment_config，无法锁定策略合同")


def _check_source_physical_contract(
    training_cfg: PilotConfig,
    source_cfg: PilotConfig,
    label: str,
) -> dict[str, Any]:
    """只由bundle锁定的白名单决定是否允许source kappa不同。"""
    allow_kappa_mismatch = bool(
        training_cfg.joint_allow_source_kappa_mismatch
    )
    _check_contract(
        training_cfg,
        source_cfg,
        label,
        allow_kappa_mismatch=allow_kappa_mismatch,
    )
    source_kappa = float(source_cfg.bess_kappa)
    evaluation_kappa = float(training_cfg.bess_kappa)
    return {
        "source_bess_kappa": source_kappa,
        "evaluation_bess_kappa": evaluation_kappa,
        "kappa_mismatch": not math.isclose(
            source_kappa, evaluation_kappa, rel_tol=0.0, abs_tol=1e-12
        ),
        "kappa_mismatch_allowed_by_bundle": allow_kappa_mismatch,
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了CUDA评估，但当前PyTorch看不到CUDA")
    return torch.device(requested)


def _terminal_soc_report(full_dual: dict[str, Any]) -> dict[str, Any]:
    actual = full_dual["actual_state"]
    plan = full_dual["plan_feasibility"]
    return {
        "unit": "MWh",
        "monetized": False,
        "included_in_revenue": False,
        "actual": {
            "initial_soc_mwh": actual["initial_soc_mwh"],
            "segment_initial_soc_mwh": actual["segment_initial_soc_mwh"],
            "segment_final_soc_mwh": actual["segment_final_soc_mwh"],
            "segment_soc_change_mwh": actual["segment_soc_change_mwh"],
            "total_segment_soc_change_mwh": actual[
                "total_segment_soc_change_mwh"
            ],
            "final_soc_mwh": actual["final_soc_mwh"],
            "final_minus_initial_soc_mwh": actual[
                "final_minus_initial_soc_mwh"
            ],
        },
        "da_plan": {
            "initial_soc_mwh": plan["initial_soc_mwh"],
            "segment_initial_soc_mwh": plan["segment_initial_soc_mwh"],
            "segment_final_soc_mwh": plan["segment_final_soc_mwh"],
            "segment_soc_change_mwh": plan["segment_soc_change_mwh"],
            "total_segment_soc_change_mwh": plan[
                "total_segment_soc_change_mwh"
            ],
            "final_soc_mwh": plan["final_soc_mwh"],
            "final_minus_initial_soc_mwh": plan[
                "final_minus_initial_soc_mwh"
            ],
        },
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mode="x"在检查和写入之间仍能防止另一个进程覆盖同名文件。
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读评估锁定的三模型联合checkpoint"
    )
    parser.add_argument("--config", required=True, help="锁定的评估合同YAML")
    parser.add_argument("--checkpoint", required=True, help="联合模型bundle")
    parser.add_argument(
        "--split",
        choices=("validation", "report_only"),
        default="validation",
    )
    parser.add_argument(
        "--confirm-report-only",
        action="store_true",
        help="确认候选和规则均已冻结，允许一次性披露report-only",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True, help="必须是尚不存在的新JSON路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _arguments(argv)
    started = time.time()
    output_path = Path(args.output)
    _require_new_output(output_path)
    _validate_split_request(args.split, args.confirm_report_only)

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到评估配置: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到联合checkpoint: {checkpoint_path}")

    cfg = PilotConfig.from_yaml(str(config_path))
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_locked_bundle(bundle)
    training_cfg = _validate_evaluation_contract(
        cfg, bundle.get("experiment_config", {})
    )

    _, source_checkpoints, source_audit = _load_bundle_source_checkpoints(
        bundle, checkpoint_path
    )
    reference_stats = source_checkpoints["da"]["norm_stats"]
    for task, source_checkpoint in source_checkpoints.items():
        _check_norm_stats(
            reference_stats, source_checkpoint["norm_stats"],
            f"{checkpoint_path.name}:{task}",
        )
        kappa_audit = _check_source_physical_contract(
            training_cfg,
            _config_from_dict(source_checkpoint["config"]),
            task,
        )
        source_audit[task] = {**source_audit[task], **kappa_audit}

    _, validation_dataset, report_only_dataset, _ = build_joint_datasets(
        cfg, norm_stats=reference_stats
    )
    dataset = (
        validation_dataset if args.split == "validation" else report_only_dataset
    )
    if args.split == "validation" and cfg.frozen_expected_validation_days:
        if len(dataset) != cfg.frozen_expected_validation_days:
            raise ValueError(
                "验证天数不符合冻结合同: "
                f"expected={cfg.frozen_expected_validation_days}, actual={len(dataset)}"
            )
    saved_validation = bundle.get("validation", {}).get("full_dual", {})
    saved_days = saved_validation.get("days")
    if args.split == "validation" and saved_days is not None:
        if int(saved_days) != len(dataset):
            raise ValueError(
                "当前验证集与checkpoint保存的验证天数不同: "
                f"checkpoint={saved_days}, current={len(dataset)}"
            )

    device = _resolve_device(args.device)
    models = _build_transformers(
        source_checkpoints, device, bundle["model_states"]
    )
    _assert_independent(models)
    # 与训练脚本的正式验证保持一致：CUDA上使用历史AMP推理，CPU不用AMP。
    evaluation = evaluate(
        models, dataset, cfg, device, device.type == "cuda"
    )
    full_dual = evaluation["full_dual"]
    if int(full_dual["days"]) != len(dataset):
        raise ValueError(
            f"完整双结算只覆盖{full_dual['days']}/{len(dataset)}个联合交付日"
        )

    split_key = "val" if args.split == "validation" else "test"
    first_date = str(dataset.windows[0].delivery_date)
    last_date = str(dataset.windows[-1].delivery_date)
    payload = {
        "schema_version": 1,
        "tool": "evaluate_joint_checkpoint",
        "read_only_evaluation": True,
        "split": args.split,
        "split_role": (
            "model_selection_and_validation"
            if args.split == "validation"
            else "final_report_only_disclosure"
        ),
        "report_only_explicitly_confirmed": bool(args.confirm_report_only),
        "days": int(full_dual["days"]),
        "first_delivery_date": first_date,
        "last_delivery_date": last_date,
        "configured_split_bounds": list(cfg.split_bounds(split_key)),
        "device": str(device),
        "cuda_amp_inference": device.type == "cuda",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "kind": bundle.get("kind", "best_model_bundle"),
            "training_mode": bundle.get("training_mode"),
            "best_epoch": bundle.get("best_epoch"),
            "completed_epoch": bundle.get("completed_epoch"),
            "recorded_best_validation_revenue": bundle.get(
                "best_validation_revenue"
            ),
        },
        "source_checkpoints": source_audit,
        "contract": {
            field: getattr(training_cfg, field)
            for field in EVALUATION_CONTRACT_FIELDS
        } | {
            "resolved_spread_threshold": float(cfg.resolved_spread_threshold),
        },
        "metrics": {
            "mean_daily_revenue": full_dual["mean_daily_revenue"],
            "positive_day_rate": full_dual["positive_day_rate"],
            "lower_tail_10pct_mean": full_dual["lower_tail_10pct_mean"],
            "max_drawdown": full_dual["max_drawdown"],
            "weekly_block_bootstrap_95ci": full_dual[
                "weekly_block_bootstrap_95ci"
            ],
            "revenue_components": full_dual["revenue_components"],
            "forecast": evaluation["forecast"],
            "terminal_soc": _terminal_soc_report(full_dual),
        },
        "audit": {
            "segments": full_dual["segments"],
            "plan_feasibility": full_dual["plan_feasibility"],
            "actual_state": full_dual["actual_state"],
            "daily_net_revenue": full_dual["daily_net_revenue"],
        },
        "wall_time_seconds": time.time() - started,
    }
    _write_json_exclusive(output_path, payload)
    print(json.dumps({
        "event": "joint_checkpoint_evaluated",
        "split": args.split,
        "days": payload["days"],
        "mean_daily_revenue": payload["metrics"]["mean_daily_revenue"],
        "output": str(output_path),
    }, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    main()
