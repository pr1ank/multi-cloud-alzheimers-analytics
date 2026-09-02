"""Mock-first Streamlit dashboard for the multi-cloud Alzheimer's MVP.

Run from the repository root with:

    streamlit run dashboard/app.py

The app renders immediately with demo data. Once Person A/B write the agreed
JSON files under ``outputs/``, those files are picked up automatically on the
next refresh.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from integration.ensemble import combine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
RECOMMENDATION_PATH = OUTPUTS_DIR / "cloud_recommendation.json"
PROFILES_PATH = PROJECT_ROOT / "cloud_engine" / "cloud_profiles.json"


MOCK_MODEL_METRICS: dict[str, Any] = {
    "model_name": "MobileNetV2_finetuned",
    "accuracy": 0.91,
    "training_time_min": 18,
    "inference_latency_ms": 42,
    "params_millions": 3.4,
    "class_labels": [
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    ],
}

# These are deliberately isolated so the demo can be replaced with exact
# paper values or Person A's measured baseline without changing UI code.
MOCK_TABLE_5: list[dict[str, Any]] = [
    {
        "scenario": "Single-cloud baseline",
        "accuracy": 0.89,
        "training_time_min": 22,
        "latency_ms": 32,
    },
    {
        "scenario": "Proposed multicloud ensemble",
        "accuracy": 0.91,
        "training_time_min": 18,
        "latency_ms": 31,
    },
]

MOCK_PROFILES: dict[str, dict[str, float]] = {
    "AWS": {"avg_latency_ms": 32, "accuracy_boost": 0.0},
    "Azure": {"avg_latency_ms": 42, "accuracy_boost": 0.02},
    "GCP": {"avg_latency_ms": 24, "accuracy_boost": 0.03},
}

MOCK_RECOMMENDATION: dict[str, Any] = {
    "recommended_provider": "GCP",
    "recommendation_reason": "Lowest estimated cost and latency for the demo workload.",
    "ensemble_weights": {"AWS": 0.31, "Azure": 0.29, "GCP": 0.40},
    "ranked_comparison": [
        {
            "provider": "GCP",
            "score": 0.91,
            "estimated_cost_usd": 22.70,
            "avg_latency_ms": 24,
            "accuracy_boost": 0.03,
        },
        {
            "provider": "Azure",
            "score": 0.78,
            "estimated_cost_usd": 32.66,
            "avg_latency_ms": 42,
            "accuracy_boost": 0.02,
        },
        {
            "provider": "AWS",
            "score": 0.70,
            "estimated_cost_usd": 36.50,
            "avg_latency_ms": 32,
            "accuracy_boost": 0.0,
        },
    ],
    "cost_breakdown": {
        "AWS": {"compute_usd": 36.25, "storage_usd": 0.25, "total_usd": 36.50},
        "Azure": {"compute_usd": 32.50, "storage_usd": 0.16, "total_usd": 32.66},
        "GCP": {"compute_usd": 22.50, "storage_usd": 0.20, "total_usd": 22.70},
    },
    "provider_predictions": {
        "AWS": [0.05, 0.65, 0.25, 0.05],
        "Azure": [0.04, 0.60, 0.30, 0.06],
        "GCP": [0.03, 0.55, 0.34, 0.08],
    },
}


def load_json(path: Path, fallback: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Load a JSON object and return ``(data, loaded_from_disk)``."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data, True
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return dict(fallback), False


def load_dashboard_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    """Load shared outputs, falling back to demo data when they are unavailable."""
    metrics, metrics_loaded = load_json(MODEL_METRICS_PATH, MOCK_MODEL_METRICS)
    recommendation, recommendation_loaded = load_json(
        RECOMMENDATION_PATH, MOCK_RECOMMENDATION
    )
    profiles, _ = load_json(PROFILES_PATH, MOCK_PROFILES)
    return metrics, recommendation, profiles, metrics_loaded and recommendation_loaded


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _provider_rows(recommendation: Mapping[str, Any], profiles: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize common Person B ranked-output shapes for the dashboard."""
    raw_rows = (
        recommendation.get("ranked_comparison")
        or recommendation.get("ranked_providers")
        or recommendation.get("comparison")
        or []
    )
    rows: list[dict[str, Any]] = []
    if isinstance(raw_rows, Mapping):
        raw_rows = [dict(value, provider=key) if isinstance(value, Mapping) else {"provider": key} for key, value in raw_rows.items()]
    for raw in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(raw, Mapping):
            continue
        provider = raw.get("provider") or raw.get("cloud") or raw.get("name")
        if not provider:
            continue
        profile = profiles.get(provider, {}) if isinstance(profiles, Mapping) else {}
        rows.append(
            {
                "provider": str(provider),
                "score": _number(raw.get("score")),
                "estimated_cost_usd": _number(
                    raw.get("estimated_cost_usd", raw.get("total_cost_usd", raw.get("cost")))
                ),
                "avg_latency_ms": _number(
                    raw.get("avg_latency_ms", raw.get("latency_ms", raw.get("latency", profile.get("avg_latency_ms"))))
                ),
                "accuracy_boost": _number(
                    raw.get("accuracy_boost", profile.get("accuracy_boost"))
                ),
            }
        )
    if not rows:
        rows = []
        for provider, profile in profiles.items() if isinstance(profiles, Mapping) else []:
            if isinstance(profile, Mapping):
                rows.append(
                    {
                        "provider": provider,
                        "score": 0.0,
                        "estimated_cost_usd": 0.0,
                        "avg_latency_ms": _number(profile.get("avg_latency_ms")),
                        "accuracy_boost": _number(profile.get("accuracy_boost")),
                    }
                )
    return rows


def build_comparison_rows(metrics: Mapping[str, Any], recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the three Table 5-style metrics shown by the dashboard."""
    rows = [dict(row) for row in MOCK_TABLE_5]
    proposed = rows[1]
    proposed["accuracy"] = _number(metrics.get("accuracy"), proposed["accuracy"])
    proposed["training_time_min"] = _number(
        metrics.get("training_time_min"), proposed["training_time_min"]
    )
    proposed["latency_ms"] = _number(
        recommendation.get(
            "ensemble_latency_ms",
            recommendation.get("weighted_latency_ms", recommendation.get("latency_ms")),
        ),
        proposed["latency_ms"],
    )
    return rows


def _cost_rows(recommendation: Mapping[str, Any], providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_costs = recommendation.get("cost_breakdown")
    output: list[dict[str, Any]] = []
    if isinstance(raw_costs, Mapping):
        for provider, values in raw_costs.items():
            if isinstance(values, Mapping):
                output.append(
                    {
                        "provider": provider,
                        "compute_usd": _number(values.get("compute_usd", values.get("compute"))),
                        "storage_usd": _number(values.get("storage_usd", values.get("storage"))),
                        "total_usd": _number(values.get("total_usd", values.get("total"))),
                    }
                )
    if output:
        return output
    return [
        {"provider": row["provider"], "compute_usd": 0.0, "storage_usd": 0.0, "total_usd": row["estimated_cost_usd"]}
        for row in providers
    ]


def build_final_prediction(metrics: Mapping[str, Any], recommendation: Mapping[str, Any]) -> dict[str, Any]:
    """Combine provider probabilities and return the class-level panel data."""
    labels = list(metrics.get("class_labels") or MOCK_MODEL_METRICS["class_labels"])
    raw_predictions = recommendation.get("provider_predictions") or recommendation.get(
        "predictions"
    )
    weights = recommendation.get("ensemble_weights") or recommendation.get("weights")
    if not isinstance(raw_predictions, Mapping):
        raw_predictions = MOCK_RECOMMENDATION["provider_predictions"]
    if not isinstance(weights, Mapping):
        weights = MOCK_RECOMMENDATION["ensemble_weights"]

    providers = [provider for provider in weights if provider in raw_predictions]
    if not providers:
        providers = list(MOCK_RECOMMENDATION["provider_predictions"])
        raw_predictions = MOCK_RECOMMENDATION["provider_predictions"]
        weights = MOCK_RECOMMENDATION["ensemble_weights"]
    combined = combine(
        [raw_predictions[provider] for provider in providers],
        [_number(weights[provider]) for provider in providers],
    )
    probabilities = combined.tolist() if hasattr(combined, "tolist") else list(combined)
    if len(probabilities) != len(labels):
        labels = list(MOCK_MODEL_METRICS["class_labels"])
    index = max(range(len(probabilities)), key=lambda item: probabilities[item])
    return {
        "label": labels[index],
        "confidence": _number(probabilities[index]),
        "probabilities": dict(zip(labels, probabilities)),
        "providers": providers,
    }


def render_dashboard() -> None:
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import streamlit as st
    except ImportError as error:  # pragma: no cover - depends on local UI environment
        raise SystemExit(
            "Dashboard dependencies are missing. Install requirements.txt and run "
            "`streamlit run dashboard/app.py`."
        ) from error

    st.set_page_config(page_title="Alzheimer's Multicloud MVP", page_icon="🧠", layout="wide")
    metrics, recommendation, profiles, live_data = load_dashboard_data()
    providers = _provider_rows(recommendation, profiles)
    comparison = build_comparison_rows(metrics, recommendation)
    costs = _cost_rows(recommendation, providers)
    prediction = build_final_prediction(metrics, recommendation)
    recommended = recommendation.get("recommended_provider") or recommendation.get("provider") or "GCP"

    st.title("Alzheimer’s Multicloud Analytics")
    st.caption(
        "Person C integration view · "
        + ("Live A/B JSON outputs loaded" if live_data else "Mock data mode — ready before A/B outputs")
    )

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.subheader("Single-cloud baseline vs proposed ensemble")
        st.caption("Mock-first Table 5-style comparison; replace seeded values with the paper's exact figures when available.")
        frame = pd.DataFrame(comparison)
        figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Accuracy", "Training time (min)", "Latency (ms)"),
            horizontal_spacing=0.10,
        )
        metrics_to_plot = [("accuracy", "Accuracy", ".0%"), ("training_time_min", "Minutes", ".1f"), ("latency_ms", "Milliseconds", ".0f")]
        colors = ["#94a3b8", "#2563eb"]
        for col, (field, axis_title, fmt) in enumerate(metrics_to_plot, start=1):
            figure.add_trace(
                go.Bar(
                    x=frame["scenario"],
                    y=frame[field],
                    marker_color=colors,
                    text=[format(value, fmt) for value in frame[field]],
                    textposition="outside",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            figure.update_yaxes(title_text=axis_title, row=1, col=col, rangemode="tozero")
        figure.update_layout(height=380, margin=dict(t=60, b=30, l=20, r=20), showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
    with top_right:
        st.subheader("Recommended provider")
        st.metric("Best fit", str(recommended))
        st.write(recommendation.get("recommendation_reason", "Provider selected by the weighted cloud decision engine."))
        st.metric("Model accuracy", f"{_number(metrics.get('accuracy')):.1%}")
        st.metric("Inference latency", f"{_number(metrics.get('inference_latency_ms')):.0f} ms")

    st.divider()
    cost_left, prediction_right = st.columns(2)
    with cost_left:
        st.subheader("Cost breakdown")
        if costs:
            cost_frame = pd.DataFrame(costs).set_index("provider")
            cost_figure = go.Figure()
            cost_figure.add_bar(name="Compute", x=cost_frame.index, y=cost_frame["compute_usd"], marker_color="#2563eb")
            cost_figure.add_bar(name="Storage", x=cost_frame.index, y=cost_frame["storage_usd"], marker_color="#93c5fd")
            cost_figure.update_layout(barmode="stack", yaxis_title="USD", height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(cost_figure, use_container_width=True)
            st.dataframe(cost_frame["total_usd"].rename("Total USD").round(2), use_container_width=True)
    with prediction_right:
        st.subheader("Final prediction")
        st.metric("Predicted class", prediction["label"], f"{prediction['confidence']:.1%} confidence")
        st.caption("Weighted provider probabilities combined with integration/ensemble.py")
        probability_frame = pd.DataFrame(
            {"Class": list(prediction["probabilities"]), "Probability": list(prediction["probabilities"].values())}
        ).set_index("Class")
        st.bar_chart(probability_frame, y="Probability", color="#2563eb")
        st.caption("Contributing providers: " + ", ".join(prediction["providers"]))

    with st.expander("Pipeline payloads"):
        st.json({"model_metrics": metrics, "cloud_recommendation": recommendation})


if __name__ == "__main__":
    render_dashboard()
