#!/usr/bin/env python
"""导出年度验证集的统一预测档案模板，不训练模型也不使用未来真值调参。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)
from scripts.decision_aware.export_global_f0_f1_predictions import (  # noqa: E402
    _dates_and_timestamps,
)


PHASE_STAGES = {
    "inner": (AnnualStage.INNER_TRAIN, AnnualStage.INNER_VALIDATION),
    "outer": (AnnualStage.OUTER_TRAIN, AnnualStage.OUTER_VALIDATION),
}


def _export_task(
    *, cfg: PilotConfig, task: str, phase: str, family: str,
    entities: list[tuple[str, str]], mappings, batch_size: int, output_dir: Path,
) -> Path:
    train_stage, validation_stage = PHASE_STAGES[phase]
    train_dataset, validation_dataset = build_global_task_datasets(
        cfg,
        task,
        entities,
        mappings,
        train_stage=train_stage,
        validation_stage=validation_stage,
    )
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
        )
        targets_da = []
        targets_rt = []
        for batch in loader:
            if task == "da":
                targets_da.append(batch["price_da_tgt"].numpy())
                targets_rt.append(batch["price_rt_tgt"].numpy())
            elif task == "rt_at_da":
                targets_rt.append(batch["price_rt_at_da_tgt"].numpy())
            else:
                targets_rt.append(batch["price_rt_tgt"].numpy())
        dates, timestamps = _dates_and_timestamps(node_dataset, task)
        if common_dates is None:
            common_dates = dates
            common_timestamps = timestamps
        elif not np.array_equal(common_dates, dates):
            raise ValueError(f"{task}各节点交付日不一致，首次冲突节点={node}")
        elif task == "rt" and not np.array_equal(common_timestamps, timestamps):
            raise ValueError(f"{task}各节点执行时间不一致，首次冲突节点={node}")
        if task == "da":
            node_target_da.append(np.concatenate(targets_da).astype(np.float32))
        node_target_rt.append(np.concatenate(targets_rt).astype(np.float32))
        print(json.dumps({
            "event": "template_node_complete",
            "phase": phase,
            "task": task,
            "market": market,
            "node": node,
            "days": len(dates),
        }, ensure_ascii=False), flush=True)

    target_rt = np.stack(node_target_rt)
    arrays = {
        "nodes": np.asarray([node for _, node in validation_dataset.entities]),
        "dates": common_dates,
        "prediction": np.zeros_like(target_rt, dtype=np.float32),
        "target_rt": target_rt,
        "metadata": json.dumps({
            "archive_version": 2,
            "family": family,
            "task": task,
            "phase": phase,
            "template_only": True,
            "selection_stage": (
                "2023_inner_validation_schema_template"
                if phase == "inner"
                else "2024_outer_validation_schema_template"
            ),
        }, ensure_ascii=False, sort_keys=True),
    }
    if task == "da":
        arrays["target_da"] = np.stack(node_target_da)
    elif task == "rt":
        arrays["executed_timestamps"] = common_timestamps
        arrays["timezone"] = np.asarray(
            str(validation_dataset.datasets[0].hourly_dataset.timezone)
        )
    output = output_dir / f"{family}_{task}_{phase}_seed0.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({
        "event": "prediction_template_saved",
        "path": output.relative_to(ROOT).as_posix(),
        "shape": list(arrays["prediction"].shape),
    }, ensure_ascii=False), flush=True)
    del train_dataset, validation_dataset
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/decision_aware/global_ercot_huber_v1.yaml"
    )
    parser.add_argument("--phase", choices=tuple(PHASE_STAGES), default="outer")
    parser.add_argument("--tasks", default="rt_at_da,rt")
    parser.add_argument("--family", default="outertemplate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir", default="data/results/global_outer_prediction_templates"
    )
    args = parser.parse_args()

    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    unknown = sorted(set(tasks) - {"da", "rt_at_da", "rt"})
    if not tasks or unknown:
        raise ValueError(f"未知或空tasks: {unknown}")
    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(
        ["ERCOT"]
    )
    entities = [("ERCOT", node) for node in mappings.node_to_id]
    completed = [
        _export_task(
            cfg=cfg,
            task=task,
            phase=args.phase,
            family=args.family,
            entities=entities,
            mappings=mappings,
            batch_size=args.batch_size,
            output_dir=ROOT / args.output_dir,
        )
        for task in tasks
    ]
    print(json.dumps({
        "event": "all_prediction_templates_complete",
        "archives": [path.relative_to(ROOT).as_posix() for path in completed],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
