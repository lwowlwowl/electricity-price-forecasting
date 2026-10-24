#!/usr/bin/env python
"""Build a pre-registered elementwise median ensemble of prediction archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_median(sources: list[Path], output: Path, *, label: str) -> dict:
    if len(sources) != 3:
        raise ValueError("固定中位数集成必须恰好包含seed 0、1、2三个归档")
    archives = [_load(path) for path in sources]
    reference = archives[0]
    required = {"nodes", "dates", "prediction", "target", "metadata"}
    if not required <= set(reference):
        raise ValueError(f"参考归档缺少字段: {sorted(required - set(reference))}")
    # 老的seed 0滚动RT导出不含executed_timestamps/timezone，而新导出含有。
    # 中位数构建只依赖预测任务共同合同；结算时间戳随后由统一模板补入。
    common_contract = {"nodes", "dates", "prediction", "target", "metadata"}
    common_keys = set.intersection(*(set(archive) for archive in archives))
    for index, archive in enumerate(archives[1:], start=1):
        if not common_contract <= set(archive):
            raise ValueError(
                f"seed {index}归档缺少共同字段: "
                f"{sorted(common_contract - set(archive))}"
            )
        for key in sorted(common_keys - {"prediction", "metadata"}):
            if key in {"prediction", "metadata"}:
                continue
            left = reference[key]
            right = archive[key]
            if left.shape != right.shape:
                raise ValueError(f"seed {index}字段{key}形状不一致")
            if np.issubdtype(left.dtype, np.number):
                if not np.allclose(left, right, rtol=0.0, atol=1e-5):
                    raise ValueError(f"seed {index}字段{key}数值不一致")
            elif not np.array_equal(left, right):
                raise ValueError(f"seed {index}字段{key}不一致")
    stacked = np.stack(
        [archive["prediction"].astype(np.float64) for archive in archives], axis=0
    )
    result = {
        key: reference[key]
        for key in sorted(common_keys - {"prediction", "metadata"})
    }
    result["prediction"] = np.median(stacked, axis=0).astype(np.float32)
    source_metadata = [json.loads(str(archive["metadata"])) for archive in archives]
    unanimous_metadata = {}
    for key in (
        "evaluation_stage", "retrained", "task", "holdout_market",
        "protocol", "uses_node_id_for_prediction", "uses_market_id_for_prediction",
        "normalization_mode",
    ):
        values = [metadata.get(key) for metadata in source_metadata]
        if values[0] is not None and all(value == values[0] for value in values[1:]):
            unanimous_metadata[key] = values[0]
    result["metadata"] = json.dumps(
        {
            "ensemble": "elementwise_hourly_median",
            "label": label,
            "seeds": [0, 1, 2],
            "selection_by_target_market": False,
            "test_2025_read": False,
            "sources": [_display_path(path) for path in sources],
            "source_sha256": [_sha256(path) for path in sources],
            "source_metadata": source_metadata,
            **unanimous_metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return {
        "output": _display_path(output),
        "sha256": _sha256(output),
        "shape": list(result["prediction"].shape),
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    report = build_median(
        [ROOT / item for item in args.source], ROOT / args.output, label=args.label
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
