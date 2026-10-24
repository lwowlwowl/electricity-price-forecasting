"""Proper distributional scores and diagnostics for electricity-price tails.

The central device is a monotone chaining transform.  Applying an ordinary
quantile score after a monotone transform remains a proper score for the
corresponding transformed quantiles, while stretching price ranges that matter
for negative prices and scarcity spikes.  Unlike outcome-dependent sample
weights, this does not change the target probability of a tail event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TailChainingSpec:
    """Fixed thresholds and slopes for the monotone price transform."""

    positive_threshold: float
    negative_threshold: float = 0.0
    positive_slope: float = 4.0
    negative_slope: float = 2.0

    def __post_init__(self) -> None:
        if self.positive_threshold <= self.negative_threshold:
            raise ValueError("positive_threshold must exceed negative_threshold")
        if self.positive_slope < 0 or self.negative_slope < 0:
            raise ValueError("tail slopes must be non-negative")


def chained_price(values: torch.Tensor, spec: TailChainingSpec) -> torch.Tensor:
    """Map prices through a continuous, strictly increasing tail stretch."""

    positive_excess = F.relu(values - spec.positive_threshold)
    negative_shortfall = F.relu(spec.negative_threshold - values)
    return (
        values
        + spec.positive_slope * positive_excess
        - spec.negative_slope * negative_shortfall
    )


def quantile_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean doubled pinball loss for ``[batch, quantile, horizon]``."""

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [batch, quantile, horizon]")
    if target.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("target must have shape [batch, horizon]")
    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    if levels.shape != (predictions.shape[1],):
        raise ValueError("quantile_levels length must match the quantile dimension")
    errors = target.unsqueeze(1) - predictions
    losses = 2.0 * torch.maximum(
        levels.view(1, -1, 1) * errors,
        (levels - 1.0).view(1, -1, 1) * errors,
    )
    if mask is None:
        return losses.mean()
    if mask.shape != target.shape:
        raise ValueError("mask must have the same shape as target")
    expanded = mask.to(losses.dtype).unsqueeze(1).expand_as(losses)
    denominator = expanded.sum().clamp_min(1.0)
    return (losses * expanded).sum() / denominator


def chained_quantile_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    spec: TailChainingSpec,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Proper quantile score in the fixed, monotone tail-stretched price space."""

    return quantile_score(
        chained_price(predictions, spec),
        chained_price(target, spec),
        quantile_levels,
        mask=mask,
    )


def nested_tail_price(
    values: torch.Tensor,
    positive_thresholds: Sequence[float],
    *,
    total_positive_slope: float,
    negative_threshold: float = 0.0,
    negative_slope: float = 0.0,
) -> torch.Tensor:
    """Continuously stretch progressively rarer positive-tail regions.

    Equal slope increments avoid tuning a separate weight for every threshold.
    The transform is strictly increasing and fixed from training-data quantiles.
    """

    thresholds = tuple(float(value) for value in positive_thresholds)
    if not thresholds or any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("positive_thresholds must be non-empty and increasing")
    if thresholds[0] <= negative_threshold:
        raise ValueError("positive thresholds must exceed the negative threshold")
    if total_positive_slope < 0 or negative_slope < 0:
        raise ValueError("tail slopes must be non-negative")
    increment = float(total_positive_slope) / len(thresholds)
    transformed = values - float(negative_slope) * F.relu(
        float(negative_threshold) - values
    )
    for threshold in thresholds:
        transformed = transformed + increment * F.relu(values - threshold)
    return transformed


def nested_tail_quantile_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    positive_thresholds: Sequence[float],
    *,
    total_positive_slope: float,
    negative_threshold: float = 0.0,
    negative_slope: float = 0.0,
) -> torch.Tensor:
    """Proper quantile score after a fixed nested monotone tail transform."""

    kwargs = {
        "positive_thresholds": positive_thresholds,
        "total_positive_slope": total_positive_slope,
        "negative_threshold": negative_threshold,
        "negative_slope": negative_slope,
    }
    return quantile_score(
        nested_tail_price(predictions, **kwargs),
        nested_tail_price(target, **kwargs),
        quantile_levels,
    )


def quantile_mass_weights(
    quantile_levels: torch.Tensor | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Voronoi masses for treating a non-uniform quantile grid as samples."""

    levels = torch.as_tensor(quantile_levels, dtype=dtype or torch.float32, device=device)
    if levels.ndim != 1 or levels.numel() < 2:
        raise ValueError("at least two one-dimensional quantile levels are required")
    if not bool(torch.all((levels > 0) & (levels < 1))):
        raise ValueError("quantile levels must lie strictly between zero and one")
    if not bool(torch.all(levels[1:] > levels[:-1])):
        raise ValueError("quantile levels must be strictly increasing")
    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=levels.dtype, device=levels.device),
            0.5 * (levels[:-1] + levels[1:]),
            torch.ones(1, dtype=levels.dtype, device=levels.device),
        ]
    )
    return boundaries[1:] - boundaries[:-1]


def event_probability_from_quantiles(
    predictions: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    side: str,
    temperature: float,
) -> torch.Tensor:
    """Estimate a differentiable event probability from forecast quantiles.

    ``side='below'`` returns ``P(Y < threshold)`` and ``side='above'`` returns
    ``P(Y > threshold)``.  The temperature is in the same units as price.
    """

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [batch, quantile, horizon]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if side not in {"below", "above"}:
        raise ValueError("side must be 'below' or 'above'")
    weights = quantile_mass_weights(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    ).view(1, -1, 1)
    below = (
        weights * torch.sigmoid((float(threshold) - predictions) / temperature)
    ).sum(dim=1)
    return below if side == "below" else 1.0 - below


def event_brier_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    side: str,
    temperature: float,
) -> torch.Tensor:
    """Unweighted Brier score, preserving the natural event base rate."""

    probabilities = event_probability_from_quantiles(
        predictions,
        quantile_levels,
        threshold,
        side=side,
        temperature=temperature,
    )
    labels = target < threshold if side == "below" else target > threshold
    return torch.mean((probabilities - labels.to(probabilities.dtype)).square())


def expected_tail_amplitude(
    predictions: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    threshold: float,
    *,
    side: str,
) -> torch.Tensor:
    """Return the quantile-integrated unconditional excess or shortfall."""

    weights = quantile_mass_weights(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    ).view(1, -1, 1)
    if side == "above":
        amplitude = F.relu(predictions - float(threshold))
    elif side == "below":
        amplitude = F.relu(float(threshold) - predictions)
    else:
        raise ValueError("side must be 'below' or 'above'")
    return (weights * amplitude).sum(dim=1)


def smooth_extrema(values: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable daily maximum and minimum for a price curve."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, horizon]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    smooth_max = float(temperature) * torch.logsumexp(values / temperature, dim=-1)
    smooth_min = -float(temperature) * torch.logsumexp(-values / temperature, dim=-1)
    return smooth_max, smooth_min


def daily_extreme_shape_loss(
    point_predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_threshold: float,
    temperature: float,
    amplitude_scale: float,
    event_weight: float,
) -> torch.Tensor:
    """Joint occurrence/amplitude score for daily positive and negative extremes."""

    if point_predictions.shape != target.shape or point_predictions.ndim != 2:
        raise ValueError("point_predictions and target must both have shape [batch, horizon]")
    if amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")
    predicted_max, predicted_min = smooth_extrema(point_predictions, temperature)
    target_max = target.max(dim=-1).values
    target_min = target.min(dim=-1).values
    max_amplitude = F.smooth_l1_loss(
        (predicted_max - positive_threshold) / amplitude_scale,
        (target_max - positive_threshold) / amplitude_scale,
    )
    min_amplitude = F.smooth_l1_loss(
        predicted_min / amplitude_scale,
        target_min / amplitude_scale,
    )
    positive_event = torch.sigmoid((predicted_max - positive_threshold) / temperature)
    negative_event = torch.sigmoid(-predicted_min / temperature)
    positive_label = (target_max > positive_threshold).to(point_predictions.dtype)
    negative_label = (target_min < 0.0).to(point_predictions.dtype)
    event_loss = F.binary_cross_entropy(positive_event, positive_label)
    event_loss = event_loss + F.binary_cross_entropy(negative_event, negative_label)
    return max_amplitude + min_amplitude + float(event_weight) * event_loss


def daily_hurdle_tail_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantile_levels: torch.Tensor | Sequence[float],
    *,
    positive_threshold: float,
    probability_temperature: float,
    extrema_temperature: float,
    amplitude_scale: float,
    event_weight: float,
) -> torch.Tensor:
    """Separate daily event occurrence from amplitude conditional on occurrence."""

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [batch, quantile, horizon]")
    if target.shape != (predictions.shape[0], predictions.shape[2]):
        raise ValueError("target must have shape [batch, horizon]")
    if amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")

    positive_hourly = event_probability_from_quantiles(
        predictions,
        quantile_levels,
        positive_threshold,
        side="above",
        temperature=probability_temperature,
    )
    negative_hourly = event_probability_from_quantiles(
        predictions,
        quantile_levels,
        0.0,
        side="below",
        temperature=probability_temperature,
    )

    def any_event_probability(hourly: torch.Tensor) -> torch.Tensor:
        no_event = torch.log1p(-hourly.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        return 1.0 - torch.exp(no_event)

    positive_probability = any_event_probability(positive_hourly)
    negative_probability = any_event_probability(negative_hourly)
    target_max = target.max(dim=-1).values
    target_min = target.min(dim=-1).values
    positive_label = target_max > positive_threshold
    negative_label = target_min < 0.0
    event_loss = F.mse_loss(
        positive_probability, positive_label.to(predictions.dtype)
    ) + F.mse_loss(negative_probability, negative_label.to(predictions.dtype))

    levels = torch.as_tensor(
        quantile_levels, dtype=predictions.dtype, device=predictions.device
    )
    median_index = int(torch.argmin(torch.abs(levels - 0.5)))
    point = predictions[:, median_index, :]
    predicted_max, predicted_min = smooth_extrema(point, extrema_temperature)
    zero = predictions.sum() * 0.0
    positive_amplitude = zero
    if bool(positive_label.any()):
        positive_amplitude = F.smooth_l1_loss(
            (predicted_max[positive_label] - positive_threshold) / amplitude_scale,
            (target_max[positive_label] - positive_threshold) / amplitude_scale,
        )
    negative_amplitude = zero
    if bool(negative_label.any()):
        negative_amplitude = F.smooth_l1_loss(
            predicted_min[negative_label] / amplitude_scale,
            target_min[negative_label] / amplitude_scale,
        )
    return (
        positive_amplitude
        + negative_amplitude
        + float(event_weight) * event_loss
    )
