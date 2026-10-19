from __future__ import annotations

import numpy as np
import pytest

from scripts.decision_aware.prepare_locked_baseline_profit_archives_recovery import (
    align_trailing_rt_boundary,
    canonicalize_rolling_first_step_truth,
    same,
)


def _item(dates: list[str], *, rt: bool = False) -> dict[str, np.ndarray]:
    day_count = len(dates)
    result = {
        "dates": np.asarray(dates),
        "prediction": np.zeros((2, day_count, 24), dtype=np.float32),
        "target": np.ones((2, day_count, 24), dtype=np.float32),
    }
    if rt:
        result["prediction"] = np.zeros((2, day_count, 24, 4), dtype=np.float32)
        result["target"] = np.ones((2, day_count, 24, 4), dtype=np.float32)
        result["executed_timestamps"] = np.zeros((day_count, 24), dtype="U1")
    return result


def test_aligns_only_one_missing_trailing_rt_day() -> None:
    full = ["2025-12-29", "2025-12-30", "2025-12-31"]
    rt = full[:-1]
    source = {
        "da": _item(full),
        "rt_at_da": {**_item(full), "target_da": _item(full)["target"]},
        "rt": _item(rt, rt=True),
    }

    aligned, dropped = align_trailing_rt_boundary(source)

    assert dropped == ["2025-12-31"]
    assert all(item["dates"].tolist() == rt for item in aligned.values())
    assert aligned["da"]["prediction"].shape[1] == 2
    assert aligned["rt_at_da"]["target_da"].shape[1] == 2


@pytest.mark.parametrize(
    "rt_dates",
    [
        ["2025-12-29"],
        ["2025-12-29", "2025-12-31"],
    ],
)
def test_rejects_more_than_one_or_nontrailing_missing_day(rt_dates: list[str]) -> None:
    full = ["2025-12-29", "2025-12-30", "2025-12-31"]
    source = {
        "da": _item(full),
        "rt_at_da": _item(full),
        "rt": _item(rt_dates, rt=True),
    }
    with pytest.raises(ValueError):
        align_trailing_rt_boundary(source)


def test_da_truth_comparison_accepts_only_float32_rounding_noise() -> None:
    reference = np.asarray([894.840087890625], dtype=np.float32)
    rounded = np.asarray([894.8400268554688], dtype=np.float32)
    same(reference, rounded, "DA真值")

    with pytest.raises(ValueError, match="DA真值不一致"):
        same(reference, reference + np.float32(1e-3), "DA真值")


def test_canonicalizes_float32_inverse_transform_roundoff() -> None:
    reference = np.asarray([[[1963.407470703125]]], dtype=np.float32)
    adjacent = np.nextafter(reference, np.float32(np.inf))
    windows = np.repeat(adjacent[..., None], 4, axis=-1)

    canonical, stats = canonicalize_rolling_first_step_truth(reference, windows)

    np.testing.assert_array_equal(canonical[..., 0], reference)
    np.testing.assert_array_equal(canonical[..., 1:], windows[..., 1:])
    assert stats["rt_truth_canonicalized_values"] == 1
    assert stats["rt_truth_max_abs_difference"] == pytest.approx(0.0001220703125)


def test_rejects_rolling_truth_difference_larger_than_recovery_tolerance() -> None:
    reference = np.asarray([[[1963.407470703125]]], dtype=np.float32)
    shifted = reference + np.float32(1e-3)
    windows = np.repeat(shifted[..., None], 4, axis=-1)

    with pytest.raises(ValueError, match="超过float32恢复容差"):
        canonicalize_rolling_first_step_truth(reference, windows)
