from __future__ import annotations

import numpy as np
import pytest

from scripts.decision_aware.summarize_allmarket_locked_2025_recovery import (
    AUDIT,
    canonicalize_numeric_truth,
)


def test_canonicalizes_small_float32_inverse_transform_difference() -> None:
    AUDIT.clear()
    reference = np.asarray([1812.30078125], dtype=np.float32)
    reconstructed = np.asarray([1812.300537109375], dtype=np.float32)

    result = canonicalize_numeric_truth(reference, reconstructed, "pooled/NYISO/rt")

    np.testing.assert_array_equal(result, reference)
    assert AUDIT == [
        {
            "label": "pooled/NYISO/rt",
            "changed_values": 1,
            "max_abs_difference": pytest.approx(0.000244140625),
        }
    ]


def test_rejects_difference_above_float32_recovery_tolerance() -> None:
    AUDIT.clear()
    reference = np.asarray([100.0], dtype=np.float32)
    shifted = np.asarray([100.001], dtype=np.float32)

    with pytest.raises(ValueError, match="超过float32恢复容差"):
        canonicalize_numeric_truth(reference, shifted, "bad")
