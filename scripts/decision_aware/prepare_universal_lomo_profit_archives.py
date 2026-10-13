#!/usr/bin/env python
"""将ERCOT无ID留一市场预测转换为连续MPC的统一档案。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _require_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label}不一致")


def prepare(source_root: Path, template_root: Path, output_root: Path, *, phase: str = "inner") -> None:
    if phase not in ("inner", "outer"):
        raise ValueError("phase必须为inner或outer")
    suffix = "_outer_frozen" if phase == "outer" else ""
    template_family = "outertemplate" if phase == "outer" else "c0"
    source_paths = {
        task: source_root / f"universal_lomo_{task}{suffix}_v1" / f"ercot_{task}_v1.npz"
        for task in TASKS
    }
    source = {
        task: _load(source_paths[task])
        for task in TASKS
    }
    template = {
        task: _load(template_root / f"{template_family}_{task}_{phase}_seed0.npz") for task in TASKS
    }

    for task in TASKS:
        if phase == "outer":
            metadata = json.loads(str(source[task]["metadata"]))
            if metadata.get("evaluation_stage") != "outer_frozen" or metadata.get("retrained") is not False:
                raise ValueError(f"{task}不是冻结时间外检查产物")
        if not np.array_equal(source[task]["nodes"], template[task]["nodes"]):
            raise ValueError(f"{task}节点顺序与结算模板不一致")
        if not np.array_equal(source[task]["dates"], template[task]["dates"]):
            raise ValueError(f"{task}交付日与结算模板不一致")

    target_da = source["rt_at_da"]["target_da"]
    target_rt = source["rt_at_da"]["target"]
    _require_equal(source["da"]["target"], target_da, "DA真值")
    _require_equal(template["da"]["target_da"], target_da, "DA结算模板真值")
    _require_equal(template["da"]["target_rt"], target_rt, "RT-at-DA结算模板真值")
    _require_equal(template["rt_at_da"]["target_rt"], target_rt, "RT-at-DA真值")
    _require_equal(source["rt"]["target"], template["rt"]["target_rt"], "滚动RT真值")

    output_root.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        arrays: dict[str, np.ndarray | str] = {
            "nodes": source[task]["nodes"],
            "dates": source[task]["dates"],
            "prediction": source[task]["prediction"],
            "target_rt": target_rt if task != "rt" else source[task]["target"],
            "metadata": json.dumps(
                {
                    "family": "universal",
                    "task": task,
                    "phase": phase,
                    "source_metadata": json.loads(str(source[task]["metadata"])),
                    "protocol": "leave_one_market_out_zero_shot",
                    "holdout_market": "ERCOT",
                    "uses_node_id_for_prediction": False,
                    "uses_market_id_for_prediction": False,
                    "source": str(
                        source_paths[task]
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        if task == "da":
            arrays["target_da"] = target_da
        if task == "rt":
            arrays["executed_timestamps"] = template["rt"]["executed_timestamps"]
            arrays["timezone"] = template["rt"]["timezone"]
        np.savez_compressed(
            output_root / f"universal_{task}_{phase}_seed0.npz", **arrays
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/results")
    parser.add_argument("--phase", choices=("inner", "outer"), default="inner")
    parser.add_argument(
        "--template-root", default="data/results/global_f0_f1_stable_v2_predictions"
    )
    parser.add_argument(
        "--output-root", default="data/results/universal_lomo_ercot_profit_predictions_v1"
    )
    args = parser.parse_args()
    prepare(ROOT / args.source_root, ROOT / args.template_root, ROOT / args.output_root, phase=args.phase)
    print(json.dumps({"event": "universal_lomo_profit_archives_ready"}))


if __name__ == "__main__":
    main()
