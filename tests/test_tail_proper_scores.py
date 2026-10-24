import pytest
import torch

from decision_aware.tail_proper_scores import (
    TailChainingSpec,
    chained_price,
    chained_quantile_score,
    daily_hurdle_tail_loss,
    daily_extreme_shape_loss,
    event_probability_from_quantiles,
    expected_tail_amplitude,
    nested_tail_price,
    nested_tail_quantile_score,
    quantile_mass_weights,
    smooth_extrema,
)


def test_chained_price_is_continuous_monotone_and_stretches_both_tails() -> None:
    spec = TailChainingSpec(
        positive_threshold=100.0,
        negative_threshold=0.0,
        positive_slope=4.0,
        negative_slope=2.0,
    )
    values = torch.tensor([-20.0, 0.0, 50.0, 100.0, 130.0])
    transformed = chained_price(values, spec)
    assert torch.all(transformed[1:] > transformed[:-1])
    assert transformed.tolist() == pytest.approx([-60.0, 0.0, 50.0, 100.0, 250.0])


def test_quantile_mass_weights_cover_unit_probability() -> None:
    weights = quantile_mass_weights([0.01, 0.1, 0.5, 0.9, 0.99])
    assert float(weights.sum()) == pytest.approx(1.0)
    assert torch.all(weights > 0)


def test_event_probability_tracks_quantile_location() -> None:
    levels = [0.1, 0.5, 0.9]
    low = torch.tensor([[[-20.0], [-10.0], [5.0]]])
    high = low + 100.0
    p_low = event_probability_from_quantiles(
        low, levels, 0.0, side="below", temperature=1.0
    )
    p_high = event_probability_from_quantiles(
        high, levels, 0.0, side="below", temperature=1.0
    )
    assert float(p_low) > 0.7
    assert float(p_high) < 1e-6


def test_chained_score_penalizes_equal_errors_more_in_tail() -> None:
    spec = TailChainingSpec(positive_threshold=100.0, positive_slope=4.0)
    levels = [0.5]
    ordinary = chained_quantile_score(
        torch.tensor([[[40.0]]]), torch.tensor([[50.0]]), levels, spec
    )
    tail = chained_quantile_score(
        torch.tensor([[[120.0]]]), torch.tensor([[130.0]]), levels, spec
    )
    assert float(tail) == pytest.approx(5.0 * float(ordinary))


def test_nested_tail_transform_is_continuous_monotone_and_progressive() -> None:
    values = torch.tensor([-10.0, 0.0, 50.0, 100.0, 150.0, 200.0])
    transformed = nested_tail_price(
        values,
        [100.0, 150.0, 200.0],
        total_positive_slope=3.0,
        negative_slope=1.0,
    )
    assert torch.all(transformed[1:] > transformed[:-1])
    assert transformed.tolist() == pytest.approx([-20.0, 0.0, 50.0, 100.0, 200.0, 350.0])


def test_nested_tail_score_has_finite_gradient() -> None:
    prediction = torch.tensor([[[90.0, 160.0], [110.0, 180.0]]], requires_grad=True)
    target = torch.tensor([[120.0, 220.0]])
    loss = nested_tail_quantile_score(
        prediction,
        target,
        [0.25, 0.75],
        [100.0, 150.0, 200.0],
        total_positive_slope=3.0,
        negative_slope=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_expected_tail_amplitude_is_zero_without_crossing() -> None:
    predictions = torch.tensor([[[10.0], [20.0], [30.0]]])
    amplitude = expected_tail_amplitude(
        predictions, [0.1, 0.5, 0.9], 50.0, side="above"
    )
    assert float(amplitude) == 0.0


def test_smooth_extrema_are_close_to_hard_extrema() -> None:
    values = torch.tensor([[1.0, 5.0, -2.0]])
    maximum, minimum = smooth_extrema(values, temperature=0.01)
    assert float(maximum) == pytest.approx(5.0, abs=0.02)
    assert float(minimum) == pytest.approx(-2.0, abs=0.02)


def test_daily_extreme_shape_loss_has_finite_gradient() -> None:
    prediction = torch.tensor([[2.0, 4.0, 3.0]], requires_grad=True)
    target = torch.tensor([[2.0, 8.0, 3.0]])
    loss = daily_extreme_shape_loss(
        prediction,
        target,
        positive_threshold=5.0,
        temperature=1.0,
        amplitude_scale=10.0,
        event_weight=0.25,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


def test_daily_hurdle_tail_loss_separates_conditional_amplitude() -> None:
    levels = [0.1, 0.5, 0.9]
    prediction = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        ],
        requires_grad=True,
    )
    target = torch.tensor([[2.0, 3.0], [2.0, 12.0]])
    loss = daily_hurdle_tail_loss(
        prediction,
        target,
        levels,
        positive_threshold=10.0,
        probability_temperature=1.0,
        extrema_temperature=1.0,
        amplitude_scale=5.0,
        event_weight=0.25,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad[1].abs().sum()) > float(prediction.grad[0].abs().sum())
