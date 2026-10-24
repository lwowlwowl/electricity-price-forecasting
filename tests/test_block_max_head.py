import numpy as np
import pytest
import torch

from decision_aware.block_max_head import (
    DailyBlockMaxQuantileHead,
    DailyExtremeEventHead,
    MonotoneLogitCalibrator,
    DailyBlockMaxGumbelHead,
    event_probability_from_scalar_quantiles,
    conditional_tail_mean_from_quantiles,
    expected_tail_localized_adjustment,
    extend_upper_quantiles,
    peak_localized_adjustment,
    select_delivery_patch_embeddings,
    scalar_quantile_score,
    select_oof_tail_shrinkage,
    online_conformalize_block_quantiles,
    gumbel_negative_log_likelihood,
    gumbel_quantiles,
)


def test_head_starts_at_non_crossing_baseline() -> None:
    head = DailyBlockMaxQuantileHead(
        model_dim=4, num_output_patches=3, num_quantiles=5, hidden_dim=8
    )
    embeddings = torch.randn(2, 3, 4)
    baseline = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 2.0, 4.0, 5.0, 6.0]])
    output = head(embeddings, baseline)
    assert torch.all(output[:, 1:] >= output[:, :-1])
    assert torch.allclose(output[0], baseline[0])
    assert torch.allclose(output[1], torch.tensor([3.0, 3.0, 4.0, 5.0, 6.0]))


def test_event_head_starts_at_training_prevalence() -> None:
    head = DailyExtremeEventHead(
        model_dim=4,
        num_output_patches=3,
        num_quantiles=5,
        hidden_dim=8,
        initial_probability=0.04,
    )
    logits = head(torch.randn(2, 3, 4), torch.randn(2, 5))
    assert torch.allclose(torch.sigmoid(logits), torch.full((2,), 0.04), atol=1e-6)


def test_logit_calibrator_starts_as_identity_and_stays_monotone() -> None:
    calibrator = MonotoneLogitCalibrator()
    raw = torch.tensor([0.01, 0.2, 0.8])
    calibrated = calibrator(raw)
    assert torch.allclose(calibrated, raw, atol=1e-6)
    calibrator.log_slope.data.fill_(-1.0)
    calibrator.intercept.data.fill_(0.5)
    calibrated = calibrator(raw)
    assert torch.all(calibrated[1:] > calibrated[:-1])


def test_gumbel_head_starts_from_baseline_and_has_finite_likelihood() -> None:
    levels = torch.tensor([0.1, 0.5, 0.9])
    baseline = torch.tensor([[1.0, 3.0, 8.0], [2.0, 4.0, 9.0]])
    head = DailyBlockMaxGumbelHead(
        model_dim=4, num_output_patches=3, num_quantiles=3, hidden_dim=8
    )
    location, scale = head(torch.randn(2, 3, 4), baseline, levels)
    predictions = gumbel_quantiles(location, scale, levels)
    assert torch.all(predictions[:, 1:] > predictions[:, :-1])
    assert torch.allclose(predictions[:, 1], baseline[:, 1], atol=1e-5)
    loss = gumbel_negative_log_likelihood(location, scale, torch.tensor([3.0, 4.0]))
    assert torch.isfinite(loss)


def test_extend_upper_quantiles_reuses_highest_available_baseline() -> None:
    hourly = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
    result = extend_upper_quantiles(hourly, [0.1, 0.5, 0.9], [0.1, 0.5, 0.9, 0.99])
    assert result.tolist() == [[2.0, 4.0, 6.0, 6.0]]


def test_scalar_quantile_score_rewards_exact_target() -> None:
    levels = [0.1, 0.5, 0.9]
    exact = scalar_quantile_score(torch.tensor([[2.0, 2.0, 2.0]]), torch.tensor([2.0]), levels)
    wrong = scalar_quantile_score(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([2.0]), levels)
    assert exact.item() == pytest.approx(0.0)
    assert wrong.item() > exact.item()


def test_oof_shrinkage_falls_back_when_proposal_is_worse() -> None:
    baseline = np.array([[1.0, 2.0], [2.0, 3.0]])
    proposed = baseline - 100.0
    alpha, audit = select_oof_tail_shrinkage(
        baseline, proposed, np.array([2.0, 3.0]), [0.9, 0.99]
    )
    assert alpha == 0.0
    assert len(audit) == 21


def test_online_conformal_calibration_never_reads_future_targets() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    predictions = np.zeros((8, 3), dtype=np.float32)
    target = np.arange(8, dtype=np.float32)
    first = online_conformalize_block_quantiles(
        predictions,
        target,
        levels,
        num_entities=1,
        window_days=4,
        minimum_history_days=2,
    )
    changed = target.copy()
    changed[6:] = 1000.0
    second = online_conformalize_block_quantiles(
        predictions,
        changed,
        levels,
        num_entities=1,
        window_days=4,
        minimum_history_days=2,
    )
    np.testing.assert_allclose(first[:7], second[:7])
    assert np.all(first[:, 1:] >= first[:, :-1])


def test_scalar_event_probability_increases_with_quantiles() -> None:
    low = event_probability_from_scalar_quantiles(
        torch.tensor([[0.0, 1.0, 2.0]]), [0.1, 0.5, 0.9], 5.0, temperature=1.0
    )
    high = event_probability_from_scalar_quantiles(
        torch.tensor([[6.0, 7.0, 8.0]]), [0.1, 0.5, 0.9], 5.0, temperature=1.0
    )
    assert high.item() > low.item()


def test_conditional_tail_mean_and_expected_adjustment_are_interpretable() -> None:
    quantiles = torch.tensor([[1.0, 5.0, 9.0]])
    mean = conditional_tail_mean_from_quantiles(quantiles, [0.1, 0.5, 0.9], 4.0)
    assert 5.0 < mean.item() < 9.0
    curves = np.array([[1.0, 3.0, 2.0, 0.0]])
    adjusted = expected_tail_localized_adjustment(
        curves, np.array([7.0]), np.array([0.5])
    )
    assert adjusted.sum() - curves.sum() == pytest.approx(2.0)
    assert int(np.argmax(adjusted)) == 1


def test_peak_adjustment_is_local_and_probability_gated() -> None:
    curves = np.array([[1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
    adjusted = peak_localized_adjustment(
        curves, np.array([5.0, 5.0]), np.array([1.0, 0.05]), probability_floor=0.1
    )
    np.testing.assert_allclose(adjusted[0], [1.7, 5.0, 2.7, 0.0])
    np.testing.assert_allclose(adjusted[1], curves[1])


def test_delivery_patch_selection_uses_offset_overlap() -> None:
    embeddings = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    selected = select_delivery_patch_embeddings(
        embeddings, torch.tensor([14, 14]), patch_size=16
    )
    assert selected.shape == (2, 3, 2)
    assert torch.equal(selected[0], embeddings[0, :3])
