from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.decision_aware.run_global_baselines_locked_recovery import (
    _DA_ARCHIVE_KEY,
    _RTAtDAArchiveView,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Window:
    target_positions: np.ndarray


class _Dataset:
    def __init__(self) -> None:
        self.price_da = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        self.windows = [_Window(np.asarray([1, 3]))]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return {"price_rt_at_da_tgt": torch.tensor([21.0, 41.0])}

    def sample_time_contract(self, index: int):
        return {"delivery_date": "2025-01-01"}


def test_archive_view_recovers_da_truth_without_mutating_source_sample() -> None:
    dataset = _Dataset()
    view = _RTAtDAArchiveView(dataset)

    recovered = view[0]

    np.testing.assert_array_equal(
        recovered[_DA_ARCHIVE_KEY].numpy(), np.asarray([20.0, 40.0], dtype=np.float32)
    )
    assert _DA_ARCHIVE_KEY not in dataset[0]
    assert view.sample_time_contract(0)["delivery_date"] == "2025-01-01"


def test_archive_view_preserves_existing_training_only_target() -> None:
    class _Existing(_Dataset):
        def __getitem__(self, index: int):
            return {_DA_ARCHIVE_KEY: torch.tensor([1.0, 2.0])}

    recovered = _RTAtDAArchiveView(_Existing())[0]

    np.testing.assert_array_equal(recovered[_DA_ARCHIVE_KEY].numpy(), [1.0, 2.0])


def test_archive_view_fails_closed_when_raw_da_target_is_unavailable() -> None:
    class _Invalid:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return {}

    with pytest.raises(KeyError, match="dataset.windows"):
        _RTAtDAArchiveView(_Invalid())[0]


def test_recovery_cloud_runner_is_resume_only_and_keeps_frozen_verification() -> None:
    text = (
        ROOT
        / "scripts"
        / "decision_aware"
        / "run_cloud_allmarket_locked_2025_v2_recovery.sh"
    ).read_text(encoding="utf-8")

    assert "freeze_allmarket_2025_protocol.py --verify" in text
    assert "freeze_allmarket_2025_baseline_protocol.py --verify" in text
    assert 'RESUME_LOCKED_RUN="${RESUME_LOCKED_RUN:-0}"' in text
    assert '"$PYTHON_BIN" "$RECOVERY_RUNNER"' in text
    assert 'scripts/decision_aware/run_global_baselines.py \\' not in text
    assert '"$PYTHON_BIN" "$PROFIT_RECOVERY_RUNNER"' in text
    assert 'scripts/decision_aware/prepare_locked_baseline_profit_archives.py \\' not in text
