import torch

from decision_aware.tail_quantile_adapter import (
    DEFAULT_QUANTILES,
    MeanResidualAdapter,
    NonCrossingQuantileAdapter,
    RankPreservingPeakScaleAdapter,
    log_peak_excursion_loss,
    multi_quantile_pinball_loss,
    quantile_integral_weights,
)


def test_integral_weights_are_positive_and_sum_to_one():
    weights = quantile_integral_weights()
    assert torch.all(weights > 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_quantile_adapter_is_non_crossing_and_starts_around_base():
    adapter = NonCrossingQuantileAdapter(8)
    representation = torch.zeros(3, 24, 8)
    base = torch.randn(3, 24)
    prediction = adapter(representation, base)
    assert prediction.shape == (3, 24, len(DEFAULT_QUANTILES))
    assert torch.all(prediction[..., 1:] > prediction[..., :-1])
    assert torch.allclose(prediction[..., 4], base)


def test_conditional_mean_is_between_outer_quantiles():
    adapter = NonCrossingQuantileAdapter(4)
    prediction = adapter(torch.randn(2, 24, 4), torch.randn(2, 24))
    mean = adapter.conditional_mean(prediction)
    assert torch.all(mean >= prediction[..., 0])
    assert torch.all(mean <= prediction[..., -1])


def test_pinball_loss_is_zero_at_exact_target_for_all_quantiles():
    target = torch.randn(2, 24)
    prediction = target.unsqueeze(-1).repeat(1, 1, len(DEFAULT_QUANTILES))
    assert multi_quantile_pinball_loss(prediction, target).item() == 0.0


def test_mean_adapter_initially_preserves_base_prediction():
    adapter = MeanResidualAdapter(7)
    representation = torch.randn(2, 24, 7)
    base = torch.randn(2, 24)
    assert torch.allclose(adapter(representation, base), base)


def test_peak_scale_adapter_starts_as_identity_and_preserves_nonpositive_side():
    adapter = RankPreservingPeakScaleAdapter(5)
    representation = torch.randn(2, 24, 5)
    base = torch.randn(2, 24)
    corrected, scale = adapter(representation, base)
    assert torch.allclose(scale, torch.ones_like(scale))
    assert torch.allclose(corrected, base)


def test_peak_excursion_loss_is_zero_for_identical_curves():
    values = torch.randn(4, 24)
    assert log_peak_excursion_loss(values, values).item() == 0.0

