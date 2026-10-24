import numpy as np

from scripts.decision_aware.audit_da_tail_distribution_v11 import (
    _beta_probability,
    _monotonicity_audit,
)


def test_strict_beta_transform_preserves_ranking_and_pr_auc():
    raw = np.linspace(0.01, 0.20, 100)
    event = np.zeros(100, dtype=bool)
    event[-10:] = True
    calibrated = _beta_probability(raw, [1.0, -1.0], 0.0)
    audit = _monotonicity_audit(raw, calibrated, event, [1.0, -1.0])
    assert audit["strictly_increasing_on_actual_range"]
    assert audit["ranking_and_pr_auc_preserved"]
    assert audit["absolute_pr_auc_difference"] <= 1.0e-12


def test_beta_transform_detects_nonmonotone_actual_range():
    raw = np.linspace(0.01, 0.99, 100)
    event = raw > 0.9
    calibrated = _beta_probability(raw, [0.4, 0.2], -1.0)
    audit = _monotonicity_audit(raw, calibrated, event, [0.4, 0.2])
    assert not audit["strictly_increasing_on_actual_range"]
    assert not audit["ranking_and_pr_auc_preserved"]
