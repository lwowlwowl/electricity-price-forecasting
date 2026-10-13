import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.prepare_global_economic_regime_multiseed import (  # noqa: E402
    _unanimous_or_base,
    _unweighted_majority,
)


def test_unweighted_majority_has_no_seed_priority():
    signs = np.asarray([
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ])
    expected = np.asarray([1.0, 1.0, -1.0])
    assert np.array_equal(_unweighted_majority(signs), expected)
    assert np.array_equal(_unweighted_majority(signs[[2, 0, 1]]), expected)


def test_unanimity_abstains_to_frozen_base_on_any_seed_disagreement():
    signs = np.asarray([
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, 1.0],
    ])
    base = np.asarray([-1.0, -1.0, 1.0, -1.0])
    selected, unanimous = _unanimous_or_base(signs, base)
    assert np.array_equal(unanimous, [True, False, True, False])
    assert np.array_equal(selected, [1.0, -1.0, -1.0, -1.0])
