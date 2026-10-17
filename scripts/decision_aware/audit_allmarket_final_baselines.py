#!/usr/bin/env python
"""审计最终基线训练产物；只检查2020—2024产物，不读取2025。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.global_data import MARKET_ORDER, load_formal_entity_mappings  # noqa: E402


TASKS = ("da", "rt_at_da", "rt")
WINDOWS_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
LOCAL_MODEL = {
    "da": "xgboost_absolute",
    "rt_at_da": "xgboost_absolute",
    "rt": "xgboost_absolute",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_node_key(node: str) -> str:
    unsafe = frozenset('<>:"/\\|?*%')
    parts: list[str] = []
    for character in node:
        if character in unsafe or ord(character) < 32:
            parts.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            parts.append(character)
    key = "".join(parts)
    stem = key.split(".", 1)[0].upper()
    if key in {".", ".."} or key.endswith((" ", ".")) or stem in WINDOWS_RESERVED_STEMS:
        digest = hashlib.sha256(node.encode("utf-8")).hexdigest()[:16]
        key = f"node_{digest}"
    return key


def _check_metadata(metadata: dict, expected: dict, label: str) -> None:
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label}冻结元数据不匹配: {mismatches}")


def build_report() -> dict:
    artifacts: list[dict] = []
    missing: list[str] = []

    for task in TASKS:
        pooled = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/pooled_xgboost/{task}_seed0.joblib"
        if not pooled.is_file():
            missing.append(pooled.relative_to(ROOT).as_posix())
        else:
            payload = joblib.load(pooled)
            _check_metadata(payload.get("metadata", {}), {
                "protocol": "all_market_pooled_xgboost_final_train",
                "task": task,
                "phase": "final",
                "seed": 0,
                "locked_2025_read": False,
                "selection_allowed": False,
            }, f"pooled XGBoost/{task}")
            artifacts.append({"kind": "pooled_xgboost", "task": task,
                              "path": pooled.relative_to(ROOT).as_posix(), "sha256": sha256(pooled)})

        dlinear = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/dlinear/{task}_seed0.pt"
        if not dlinear.is_file():
            missing.append(dlinear.relative_to(ROOT).as_posix())
        else:
            payload = torch.load(dlinear, map_location="cpu", weights_only=False)
            _check_metadata(payload.get("metadata", {}), {
                "protocol": "all_market_dlinear_final_train",
                "task": task,
                "phase": "final",
                "seed": 0,
                "locked_2025_read": False,
                "selection_allowed": False,
            }, f"DLinear/{task}")
            artifacts.append({"kind": "dlinear", "task": task,
                              "path": dlinear.relative_to(ROOT).as_posix(), "sha256": sha256(dlinear)})

        adapter = ROOT / f"data/checkpoints/allmarket_final_baselines_v1/timesfm25/{task}_seed0"
        adapter_file = adapter / "adapter_model.safetensors"
        summary_file = adapter / "final_training_summary.json"
        if not adapter_file.is_file() or not summary_file.is_file():
            missing.append(adapter.relative_to(ROOT).as_posix())
        else:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            _check_metadata(summary, {
                "protocol": "all_market_timesfm25_lora_final_train",
                "task": task,
                "seed": 0,
                "locked_2025_read": False,
                "selection_allowed": False,
            }, f"TimesFM-2.5/{task}")
            if summary.get("adapter_sha256") != sha256(adapter_file):
                raise ValueError(f"TimesFM-2.5/{task} adapter哈希不匹配")
            artifacts.append({"kind": "timesfm25_lora", "task": task,
                              "path": adapter_file.relative_to(ROOT).as_posix(), "sha256": sha256(adapter_file)})

    mappings = load_formal_entity_mappings(ROOT / "configs/nodes.yaml")
    for market in MARKET_ORDER:
        market_nodes = [node for node in mappings.node_to_id if mappings.node_to_market[node] == market]
        market_subdir = Path() if market == "ERCOT" else Path(market.lower())
        for task in TASKS:
            model = LOCAL_MODEL[task]
            for node in market_nodes:
                path = (ROOT / "data/checkpoints/global_baselines_v1" / market_subdir /
                        f"{task}_final" / f"{model}_{portable_node_key(node)}_seed0.joblib")
                if not path.is_file():
                    missing.append(path.relative_to(ROOT).as_posix())
                    continue
                payload = joblib.load(path)
                _check_metadata(payload.get("metadata", {}), {
                    "task": task, "market": market, "node": node,
                    "model": model, "seed": 0, "phase": "final",
                    "locked_2025_read": False,
                }, f"逐节点XGBoost/{market}/{task}/{node}")
                artifacts.append({"kind": "per_node_xgboost", "market": market,
                                  "task": task, "node": node,
                                  "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})

    return {
        "protocol": "allmarket_final_baseline_artifact_audit_v1",
        "training_years": [2020, 2021, 2022, 2023, 2024],
        "locked_2025_read": False,
        "complete": not missing,
        "artifact_count": len(artifacts),
        "missing_count": len(missing),
        "missing": missing,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/results/allmarket_final_baselines_v1/artifact_audit.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = build_report()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("complete", "artifact_count", "missing_count")}))
    if args.require_complete and not report["complete"]:
        raise SystemExit(f"最终基线产物未齐：缺少{report['missing_count']}项，见{output}")


if __name__ == "__main__":
    main()
