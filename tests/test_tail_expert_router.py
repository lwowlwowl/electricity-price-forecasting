from __future__ import annotations

import numpy as np

from decision_aware.tail_expert_router import (
    blocked_oof_probability,
    build_feature_cube,
    fit_router,
    fit_regret_tree,
    model_explanation,
    predict_regret,
    regret_tree_explanation,
    router_probability,
)


def test_feature_cube_is_day_level_and_uses_tail_signals() -> None:
    nodes, days = 2, 10
    dates = np.asarray([f"2023-01-{day:02d}" for day in range(1, days + 1)])
    base = np.arange(24, dtype=np.float64)[None, None, :]
    transformer_da = np.broadcast_to(base, (nodes, days, 24)).copy()
    transformer_rt = transformer_da - 2.0
    dlinear_da = transformer_da + 3.0
    dlinear_rt = transformer_rt - 1.0
    dlinear_da[0, 0, 5] = -10.0
    cube, names = build_feature_cube(
        transformer_da, transformer_rt, dlinear_da, dlinear_rt, dates
    )
    assert cube.shape[:2] == (nodes, days)
    assert cube.shape[-1] == len(names)
    assert "dlinear_da_negative_hours" in names
    assert "transformer_spread_positive_6h_count" in names
    assert "spread_disagreement_mae" in names
    assert np.isfinite(cube).all()


def test_logistic_and_tree_routers_are_interpretable() -> None:
    rng = np.random.default_rng(20261020)
    features = rng.normal(size=(500, 6))
    advantage = 4.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(size=500)
    names = tuple(f"f{index}" for index in range(features.shape[1]))
    for kind in ("logistic", "tree"):
        model = fit_router(kind, features, advantage)
        probability = router_probability(model, features)
        explanation = model_explanation(model, names)
        assert probability.shape == (len(features),)
        assert np.all((probability >= 0.0) & (probability <= 1.0))
        assert explanation["feature_count"] <= len(names)


def test_blocked_oof_probability_covers_every_row() -> None:
    rng = np.random.default_rng(9)
    days, nodes = 20, 3
    features = rng.normal(size=(days * nodes, 4))
    advantage = features[:, 0] + rng.normal(scale=0.2, size=days * nodes)
    dates = np.tile(np.asarray([f"2023-01-{day:02d}" for day in range(1, days + 1)]), nodes)
    probability, folds = blocked_oof_probability(
        "logistic", features, advantage, dates, blocks=5
    )
    assert len(folds) == 5
    assert np.isfinite(probability).all()


def test_regret_tree_has_direct_economic_decision_meaning() -> None:
    rng = np.random.default_rng(20261021)
    features = rng.normal(size=(600, 5))
    advantage = np.where(features[:, 0] > 0.0, 8.0, -3.0)
    names = tuple(f"f{index}" for index in range(features.shape[1]))
    model = fit_regret_tree(features, advantage, min_samples_leaf=50)
    prediction = predict_regret(model, features)
    explanation = regret_tree_explanation(model, names)
    assert np.mean((prediction > 0.0) == (advantage > 0.0)) > 0.95
    assert explanation["feature_count"] <= 3
    assert explanation["leaves"]
