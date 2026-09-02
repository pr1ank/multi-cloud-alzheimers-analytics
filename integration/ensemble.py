"""Prediction-combination helpers shared by the dashboard and pipeline.

The combiner deliberately has no model or cloud-provider assumptions. It accepts
Python numbers/lists as well as NumPy arrays, so it can be exercised with dummy
predictions before Person A's model is available.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Sequence


def _validate_weights(weights: Sequence[Real], count: int) -> tuple[float, ...]:
    if len(weights) != count:
        raise ValueError(
            f"predictions and weights must have the same length; "
            f"got {count} predictions and {len(weights)} weights"
        )

    parsed: list[float] = []
    for index, weight in enumerate(weights):
        if not isinstance(weight, Real) or isinstance(weight, bool):
            raise TypeError(f"weight at index {index} must be a real number")
        value = float(weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"weight at index {index} must be finite and non-negative"
            )
        parsed.append(value)

    total = sum(parsed)
    if total <= 0:
        raise ValueError("at least one weight must be greater than zero")
    return tuple(value / total for value in parsed)


def _combine_python(predictions: Sequence[Any], normalized_weights: Sequence[float]) -> Any:
    """Combine nested Python sequences without requiring NumPy."""
    first = predictions[0]

    if isinstance(first, Real) and not isinstance(first, bool):
        if any(
            not isinstance(prediction, Real) or isinstance(prediction, bool)
            for prediction in predictions
        ):
            raise TypeError("all scalar predictions must be real numbers")
        return sum(
            weight * float(prediction)
            for prediction, weight in zip(predictions, normalized_weights)
        )

    if isinstance(first, (list, tuple)):
        expected_length = len(first)
        if any(
            not isinstance(prediction, (list, tuple))
            or len(prediction) != expected_length
            for prediction in predictions
        ):
            raise ValueError("all prediction arrays must have the same shape")
        combined = [
            _combine_python(
                [prediction[index] for prediction in predictions],
                normalized_weights,
            )
            for index in range(expected_length)
        ]
        return tuple(combined) if isinstance(first, tuple) else combined

    raise TypeError(
        "predictions must be numeric scalars, Python sequences, or NumPy arrays"
    )


def combine(predictions: list, weights: list):
    """Return the normalized weighted average of prediction arrays.

    Parameters
    ----------
    predictions:
        A non-empty list of same-shaped numeric scalars, lists, tuples, or
        NumPy arrays. For classifier probabilities, use one array per model or
        provider with one probability per class.
    weights:
        One non-negative weight per prediction. Weights are normalized internally
        so both ``[1, 1]`` and ``[0.5, 0.5]`` produce the same result.

    Returns
    -------
    Any
        A NumPy array when NumPy inputs are supplied; otherwise a scalar, list,
        or tuple matching the input structure.

    Examples
    --------
    >>> combine([[0.8, 0.2], [0.4, 0.6]], [0.75, 0.25])
    [0.7, 0.3]
    """
    if not isinstance(predictions, (list, tuple)) or not predictions:
        raise ValueError("predictions must be a non-empty list")
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError("weights must be a non-empty list")

    normalized_weights = _validate_weights(weights, len(predictions))

    # Keep NumPy optional for day-one dummy arrays and lightweight unit tests.
    try:
        import numpy as np
    except ImportError:
        return _combine_python(predictions, normalized_weights)

    try:
        arrays = [np.asarray(prediction, dtype=float) for prediction in predictions]
    except (TypeError, ValueError) as error:
        raise TypeError("predictions must contain numeric values") from error

    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("all prediction arrays must have the same shape")

    combined = np.zeros_like(arrays[0], dtype=float)
    for array, weight in zip(arrays, normalized_weights):
        combined = combined + array * weight
    return combined


__all__ = ["combine"]
