import numpy as np

from scripts.decision_aware.audit_2025_da_profit_rank_amplitude import (
    rank_value_recombine,
)


def test_rank_value_recombine_preserves_rank_source_and_value_multiset():
    rank_source = np.arange(24, dtype=float)[None, None, ::-1]
    value_source = (100.0 + 2.0 * np.arange(24))[None, None, :]
    result = rank_value_recombine(rank_source, value_source)
    assert np.array_equal(np.sort(result, axis=-1), np.sort(value_source, axis=-1))
    assert np.array_equal(np.argsort(result, axis=-1), np.argsort(rank_source, axis=-1))


def test_rank_value_recombine_rejects_non_daily_shape():
    with np.testing.assert_raises(ValueError):
        rank_value_recombine(np.zeros((2, 23)), np.zeros((2, 23)))
