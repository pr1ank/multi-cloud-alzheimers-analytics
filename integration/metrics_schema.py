"""Small, dependency-free checks for the shared JSON contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MODEL_METRICS_FIELDS = (
    "model_name",
    "accuracy",
    "training_time_min",
    "inference_latency_ms",
    "params_millions",
    "class_labels",
)


def validate_model_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a model metrics object without changing its shape."""
    missing = [field for field in MODEL_METRICS_FIELDS if field not in metrics]
    if missing:
        raise ValueError(f"model metrics missing required fields: {', '.join(missing)}")
    if not isinstance(metrics["class_labels"], list) or not metrics["class_labels"]:
        raise ValueError("class_labels must be a non-empty list")

    for field in MODEL_METRICS_FIELDS[1:-1]:
        value = metrics[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"model metric '{field}' must be numeric")
    return dict(metrics)


def validate_recommendation(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum recommendation contract consumed by the dashboard."""
    provider = recommendation.get("recommended_provider") or recommendation.get(
        "provider"
    )
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("recommendation must include recommended_provider")
    result = dict(recommendation)
    result.setdefault("recommended_provider", provider)
    return result


__all__ = ["MODEL_METRICS_FIELDS", "validate_model_metrics", "validate_recommendation"]
