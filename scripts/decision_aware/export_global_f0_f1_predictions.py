#!/usr/bin/env python
"""导出ERCOT 15节点Global模型的完整年度验证预测。"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import (  # noqa: E402
    AnnualStage,
    FORMAL_EXPERIMENT_CONTRACT,
)
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    flatten_rt_delivery_day_batch,
    load_formal_entity_mappings,
)
from decision_aware.model_da import DecisionAwareDAForecaster  # noqa: E402
from decision_aware.model_rt import DecisionAwareRTForecaster  # noqa: E402
from decision_aware.model_rt_da import DecisionAwareRTAtDAForecaster  # noqa: E402


TASKS = {
    "da": (DecisionAwareDAForecaster, "p_da"),
    "rt_at_da": (DecisionAwareRTAtDAForecaster, "p_rt_at_da"),
    "rt": (DecisionAwareRTForecaster, "p_rt"),
}

FAMILIES = {
    "f0": "data/checkpoints/global_ercot_huber_f0_v1/{task}_inner_seed0/best.pt",
    "f1": "data/checkpoints/global_ercot_huber_v1/{task}_inner_seed0/best.pt",
}

PHASE_STAGES = {
    "inner": (AnnualStage.INNER_TRAIN, AnnualStage.INNER_VALIDATION),
    "outer": (AnnualStage.OUTER_TRAIN, AnnualStage.OUTER_VALIDATION),
    "final": (AnnualStage.FINAL_TRAIN, AnnualStage.LOCKED_RETROSPECTIVE_TEST),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_config(checkpoint: dict) -> PilotConfig:
    allowed = {field.name for field in fields(PilotConfig)}
    return PilotConfig(**{
        key: value for key, value in checkpoint["config"].items() if key in allowed
    })


def _nested_equal(left, right) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _dates_and_timestamps(node_dataset, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    if task == "rt":
        dates = []
        timestamps = []
        for delivery_date, hourly_indices in node_dataset.days:
            dates.append(str(delivery_date))
            timestamps.append([
                node_dataset.hourly_dataset.sample_time_contract(index)[
                    "executed_target"
                ].isoformat()
                for index in hourly_indices
            ])
        return np.asarray(dates), np.asarray(timestamps)
    dates = [
        str(node_dataset.sample_time_contract(index)["delivery_date"])
        for index in range(len(node_dataset))
    ]
    return np.asarray(dates), None


def _archive_schema_valid(existing: np.lib.npyio.NpzFile, task: str) -> bool:
    files = set(existing.files)
    if task == "da":
        return "target_da" in files and "executed_timestamps" not in files
    if task == "rt_at_da":
        return "target_da" not in files and "executed_timestamps" not in files
    return (
        {"executed_timestamps", "timezone"}.issubset(files)
        and existing["executed_timestamps"].dtype.kind != "O"
        and existing["timezone"].dtype.kind != "O"
    )


@torch.no_grad()
def _predict_family_task(
    family: str,
    task: str,
    checkpoint_path: Path,
    train_dataset,
    validation_dataset,
    mappings,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    phase: str = "inner",
    force: bool = False,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
    mappings.validate_checkpoint(checkpoint)
    if checkpoint.get("task") != task or checkpoint.get("phase") != phase:
        raise ValueError(f"{checkpoint_path}的task/phase不匹配")
    if not _nested_equal(
        checkpoint.get("norm_stats_by_node"), train_dataset.norm_stats_by_node
    ):
        raise ValueError(f"{checkpoint_path}的逐节点scaler与当前训练期数据不一致")
    cfg = _checkpoint_config(checkpoint)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    expected_upstream_sha256 = None
    if (
        task == "rt_at_da"
        and str(cfg.rt_at_da_parameterization).lower() in {
            "da_minus_spread", "economic_regime"
        }
    ):
        if not cfg.rt_at_da_upstream_da_checkpoint:
            raise ValueError("价差参数化checkpoint缺少冻结DA来源")
        expected_upstream_sha256 = _sha256(
            ROOT / cfg.rt_at_da_upstream_da_checkpoint
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{family}_{task}_{phase}_seed0.npz"
    if output_path.is_file() and not force:
        with np.load(output_path, allow_pickle=False) as existing:
            existing_metadata = json.loads(str(existing["metadata"].item()))
            if (
                _archive_schema_valid(existing, task)
                and
                existing_metadata.get("archive_version") == 2
                and existing_metadata.get("family") == family
                and existing_metadata.get("task") == task
                and existing_metadata.get("phase") == phase
                and existing_metadata.get("checkpoint_sha256") == _sha256(checkpoint_path)
                and existing_metadata.get("contract_hash") == checkpoint["contract_hash"]
                and (
                    existing_metadata.get("upstream_da") or {}
                ).get("checkpoint_sha256") == expected_upstream_sha256
            ):
                print(json.dumps({
                    "event": "prediction_archive_reused",
                    "path": output_path.relative_to(ROOT).as_posix(),
                }, ensure_ascii=False), flush=True)
                return output_path
    model_class, prediction_key = TASKS[task]
    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    upstream_da_model = None
    upstream_metadata = None
    if (
        task == "rt_at_da"
        and str(cfg.rt_at_da_parameterization).lower() in {
            "da_minus_spread", "economic_regime"
        }
    ):
        if not cfg.rt_at_da_upstream_da_checkpoint:
            raise ValueError("价差参数化checkpoint缺少冻结DA来源")
        upstream_path = ROOT / cfg.rt_at_da_upstream_da_checkpoint
        upstream = torch.load(
            upstream_path, map_location="cpu", weights_only=False
        )
        FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(upstream)
        mappings.validate_checkpoint(upstream)
        if upstream.get("task") != "da" or upstream.get("phase") != phase:
            raise ValueError("价差参数化的上游必须是同阶段DA checkpoint")
        if not _nested_equal(
            upstream.get("norm_stats_by_node"), train_dataset.norm_stats_by_node
        ):
            raise ValueError("上游DA checkpoint的逐节点scaler不匹配")
        upstream_cfg = _checkpoint_config(upstream)
        upstream_da_model = DecisionAwareDAForecaster(upstream_cfg).to(device)
        upstream_da_model.load_state_dict(upstream["model_state"], strict=True)
        upstream_da_model.eval()
        upstream_metadata = {
            "checkpoint": upstream_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": _sha256(upstream_path),
            "epoch": upstream.get("epoch"),
        }

    node_predictions = []
    node_target_da = []
    node_target_rt = []
    common_dates = None
    common_timestamps = None
    for entity_index, ((market, node), node_dataset) in enumerate(
        zip(validation_dataset.entities, validation_dataset.datasets)
    ):
        start = validation_dataset.offsets[entity_index]
        end = validation_dataset.offsets[entity_index + 1]
        loader = DataLoader(
            Subset(validation_dataset, range(start, end)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        predictions = []
        targets_da = []
        targets_rt = []
        for raw_batch in loader:
            if task == "rt":
                targets_rt.append(raw_batch["price_rt_tgt"].cpu().numpy())
                batch = flatten_rt_delivery_day_batch(raw_batch)
            else:
                if task == "da":
                    targets_da.append(raw_batch["price_da_tgt"].cpu().numpy())
                    targets_rt.append(raw_batch["price_rt_tgt"].cpu().numpy())
                else:
                    targets_rt.append(
                        raw_batch["price_rt_at_da_tgt"].cpu().numpy()
                    )
                batch = raw_batch
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if upstream_da_model is not None:
                conditioned_batch = dict(batch)
                conditioned_batch["p_da_upstream"] = upstream_da_model(batch)[
                    "p_da"
                ].detach()
            else:
                conditioned_batch = batch
            prediction = model(conditioned_batch)[prediction_key].detach().float().cpu().numpy()
            if task == "rt":
                prediction = prediction.reshape(-1, 24, prediction.shape[-1])
            predictions.append(prediction)

        dates, timestamps = _dates_and_timestamps(node_dataset, task)
        if common_dates is None:
            common_dates = dates
            common_timestamps = timestamps
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"{task}各节点交付日不一致，首次冲突节点={node}")
        elif task == "rt" and not np.array_equal(common_timestamps, timestamps):
            raise ValueError(f"{task}各节点执行时间不一致，首次冲突节点={node}")
        node_predictions.append(np.concatenate(predictions).astype(np.float32))
        if task == "da":
            node_target_da.append(np.concatenate(targets_da).astype(np.float32))
        node_target_rt.append(np.concatenate(targets_rt).astype(np.float32))
        print(json.dumps({
            "event": "node_prediction_complete",
            "family": family,
            "task": task,
            "market": market,
            "node": node,
            "days": len(dates),
        }, ensure_ascii=False), flush=True)

    arrays = {
        "nodes": np.asarray([node for _, node in validation_dataset.entities]),
        "dates": common_dates,
        "prediction": np.stack(node_predictions),
        "target_rt": np.stack(node_target_rt),
        "metadata": json.dumps({
            "archive_version": 2,
            "family": family,
            "task": task,
            "phase": phase,
            # 文件名中的seed0是档案schema约定；真实模型seed必须从
            # checkpoint读取，否则多seed导出会被误标。
            "seed": int(checkpoint.get("seed", cfg.seed)),
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "contract_hash": checkpoint["contract_hash"],
            "entity_mapping_hash": checkpoint["entity_mapping_hash"],
            "source_data_sha256": checkpoint["source_data_sha256"],
            "selection": (
                "checkpoint best 2023 node-equal macro MAE"
                if phase == "inner"
                else "checkpoint best 2024 node-equal macro MAE"
            ),
            "rt_at_da_parameterization": cfg.rt_at_da_parameterization,
            "upstream_da": upstream_metadata,
        }, ensure_ascii=False, sort_keys=True),
    }
    if task == "da":
        arrays["target_da"] = np.stack(node_target_da)
    elif task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(
            str(validation_dataset.datasets[0].hourly_dataset.timezone)
        )
    np.savez_compressed(output_path, **arrays)
    del model, checkpoint, upstream_da_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({
        "event": "prediction_archive_saved",
        "path": output_path.relative_to(ROOT).as_posix(),
        "shape": list(arrays["prediction"].shape),
    }, ensure_ascii=False), flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument("--families", default="f0,f1")
    parser.add_argument(
        "--candidate-manifest",
        help="可选JSON；families字段按family/task给出checkpoint相对路径",
    )
    parser.add_argument("--tasks", default="da,rt_at_da,rt")
    parser.add_argument("--phase", choices=tuple(PHASE_STAGES), default="inner")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir", default="data/results/global_f0_f1_predictions_v1"
    )
    args = parser.parse_args()

    family_paths = {
        family: {task: template.format(task=task) for task in TASKS}
        for family, template in FAMILIES.items()
    }
    if args.candidate_manifest:
        candidate_report = json.loads(
            (ROOT / args.candidate_manifest).read_text(encoding="utf-8")
        )
        family_paths = candidate_report["families"]
    families = tuple(value.strip().lower() for value in args.families.split(",") if value.strip())
    tasks = tuple(value.strip().lower() for value in args.tasks.split(",") if value.strip())
    unknown_families = sorted(set(families) - set(family_paths))
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_families or unknown_tasks:
        raise ValueError(
            f"未知family/task: families={unknown_families}, tasks={unknown_tasks}"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        ["ERCOT"]
    )
    entities = [("ERCOT", node) for node in mappings.node_to_id]
    output_dir = ROOT / args.output_dir
    completed = []
    train_stage, validation_stage = PHASE_STAGES[args.phase]
    for task in tasks:
        train_dataset, validation_dataset = build_global_task_datasets(
            cfg,
            task,
            entities,
            mappings,
            train_stage=train_stage,
            validation_stage=validation_stage,
        )
        for family in families:
            if task not in family_paths[family]:
                raise ValueError(f"candidate manifest缺少{family}/{task}")
            checkpoint_path = ROOT / family_paths[family][task]
            completed.append(_predict_family_task(
                family,
                task,
                checkpoint_path,
                train_dataset,
                validation_dataset,
                mappings,
                device,
                args.batch_size,
                output_dir,
                args.phase,
                args.force,
            ))
        del train_dataset, validation_dataset
    print(json.dumps({
        "event": "all_prediction_archives_complete",
        "device": str(device),
        "archives": [path.relative_to(ROOT).as_posix() for path in completed],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
