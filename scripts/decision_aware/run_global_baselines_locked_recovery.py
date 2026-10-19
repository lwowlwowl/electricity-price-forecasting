#!/usr/bin/env python
"""Recover locked-2025 local-baseline archives after the RT-at-DA DA-truth bug.

The frozen baseline runner correctly excludes realized delivery-day DA prices from
RT-at-DA model samples, but its market-level archive writer later expected that
evaluation-only target under a training-only key.  This recovery entrypoint leaves
the frozen runner untouched and exposes the DA target only to the archive writer.
It does not change features, predictions, checkpoints, model selection, or losses.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware import run_global_baselines as frozen_runner  # noqa: E402


_DA_ARCHIVE_KEY = "price_da_tgt_weighting_only"


class _RTAtDAArchiveView:
    """Add realized DA truth only while serializing a locked evaluation archive."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, index: int):
        sample = self._dataset[index]
        if _DA_ARCHIVE_KEY in sample:
            return sample
        if not hasattr(self._dataset, "windows") or not hasattr(
            self._dataset, "price_da"
        ):
            raise KeyError(
                "RT-at-DA locked archive recovery requires dataset.windows and "
                "dataset.price_da"
            )
        window = self._dataset.windows[index]
        target = np.asarray(
            self._dataset.price_da[window.target_positions], dtype=np.float32
        ).copy()
        recovered = dict(sample)
        recovered[_DA_ARCHIVE_KEY] = torch.from_numpy(target)
        return recovered


_FROZEN_SAVE_LOCKED_MARKET_ARCHIVE = frozen_runner._save_locked_market_archive


def _save_locked_market_archive_recovery(**kwargs):
    args = kwargs["args"]
    if args.task == "rt_at_da":
        kwargs = dict(kwargs)
        kwargs["validation_by_node"] = {
            node: _RTAtDAArchiveView(dataset)
            for node, dataset in kwargs["validation_by_node"].items()
        }
    return _FROZEN_SAVE_LOCKED_MARKET_ARCHIVE(**kwargs)


def main() -> None:
    frozen_runner._save_locked_market_archive = _save_locked_market_archive_recovery
    frozen_runner.main()


if __name__ == "__main__":
    main()
