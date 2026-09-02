"""End-to-end Person C orchestrator.

The normal path consumes Person A's metrics, calls Person B's decision engine,
persists its recommendation, and optionally launches the Streamlit dashboard.
Use ``--mock`` to exercise the flow before either upstream track is finished.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from integration.metrics_schema import validate_model_metrics, validate_recommendation


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
DEFAULT_RECOMMENDATION_PATH = ROOT / "outputs" / "cloud_recommendation.json"
DEFAULT_PROFILES_PATH = ROOT / "cloud_engine" / "cloud_profiles.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def build_workload_description(
    metrics: Mapping[str, Any],
    data_size_gb: float,
    latency_requirement_ms: float,
    budget_usd: float,
) -> dict[str, Any]:
    """Build the stable workload payload passed to Person B's engine."""
    return {
        "data_size_gb": data_size_gb,
        "latency_requirement_ms": latency_requirement_ms,
        "budget_usd": budget_usd,
        "model_name": metrics.get("model_name"),
        "model_accuracy": metrics.get("accuracy"),
        "model_latency_ms": metrics.get("inference_latency_ms"),
    }


def _invoke_decision_engine(
    workload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    profiles_path: Path,
    recommendation_path: Path,
) -> Mapping[str, Any]:
    """Call common Person B function names while keeping B's module independent.

    Person B can expose any one of the conventional names below. The adapter
    maps only recognized parameter names and does not require C to duplicate B's
    scoring implementation.
    """
    try:
        module = importlib.import_module("cloud_engine.decision_engine")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Person B's cloud_engine/decision_engine.py is not available yet. "
            "Run `python main.py --mock` for the C demo path."
        ) from error

    profiles = read_json(profiles_path)
    function_names = (
        "recommend_cloud",
        "recommend_provider",
        "decide",
        "recommend",
        "run_decision_engine",
        "get_recommendation",
    )
    parameter_values: dict[str, Any] = {
        "workload": workload,
        "workload_description": workload,
        "request": workload,
        "metrics": metrics,
        "model_metrics": metrics,
        "profiles": profiles,
        "cloud_profiles": profiles,
        "profiles_path": profiles_path,
        "profile_path": profiles_path,
        "recommendation_path": recommendation_path,
        "output_path": recommendation_path,
        "output_file": recommendation_path,
    }

    for function_name in function_names:
        function = getattr(module, function_name, None)
        if not callable(function):
            continue
        signature = inspect.signature(function)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            kwargs = parameter_values
        else:
            kwargs = {
                name: parameter_values[name]
                for name in signature.parameters
                if name in parameter_values
            }
        result = function(**kwargs)
        if result is None and recommendation_path.exists():
            result = read_json(recommendation_path)
        if isinstance(result, Mapping):
            return result
        raise TypeError(
            f"cloud decision function {function_name} must return a JSON-like mapping"
        )

    raise AttributeError(
        "cloud_engine.decision_engine.py must expose one of: "
        + ", ".join(function_names)
    )


def run_pipeline(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    recommendation_path: Path = DEFAULT_RECOMMENDATION_PATH,
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    *,
    data_size_gb: float = 1.0,
    latency_requirement_ms: float = 50.0,
    budget_usd: float = 40.0,
) -> dict[str, Any]:
    """Run A → B → output wiring and return the recommendation."""
    metrics = validate_model_metrics(read_json(metrics_path))
    workload = build_workload_description(
        metrics,
        data_size_gb=data_size_gb,
        latency_requirement_ms=latency_requirement_ms,
        budget_usd=budget_usd,
    )
    recommendation = validate_recommendation(
        _invoke_decision_engine(workload, metrics, profiles_path, recommendation_path)
    )
    recommendation.setdefault("workload", workload)
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(recommendation_path, recommendation)
    return recommendation


def run_mock_pipeline() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the same two payloads used by the dashboard's day-one demo."""
    from dashboard.app import MOCK_MODEL_METRICS, MOCK_RECOMMENDATION

    return dict(MOCK_MODEL_METRICS), dict(MOCK_RECOMMENDATION)


def launch_dashboard() -> None:
    dashboard_path = ROOT / "dashboard" / "app.py"
    subprocess.Popen(  # noqa: S603 - command is fixed to this repository's app
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        cwd=ROOT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use C's day-one mock payloads")
    parser.add_argument("--launch-dashboard", action="store_true", help="start Streamlit after orchestration")
    parser.add_argument("--data-size-gb", type=float, default=1.0)
    parser.add_argument("--latency-requirement-ms", type=float, default=50.0)
    parser.add_argument("--budget-usd", type=float, default=40.0)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--recommendation-path", type=Path, default=DEFAULT_RECOMMENDATION_PATH)
    parser.add_argument("--profiles-path", type=Path, default=DEFAULT_PROFILES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        metrics, recommendation = run_mock_pipeline()
        print(f"Mock model metrics ready: {metrics['model_name']} ({metrics['accuracy']:.1%})")
        print(f"Mock recommended provider: {recommendation['recommended_provider']}")
    else:
        recommendation = run_pipeline(
            metrics_path=args.metrics_path,
            recommendation_path=args.recommendation_path,
            profiles_path=args.profiles_path,
            data_size_gb=args.data_size_gb,
            latency_requirement_ms=args.latency_requirement_ms,
            budget_usd=args.budget_usd,
        )
        print(f"Wrote cloud recommendation: {args.recommendation_path}")
        print(f"Recommended provider: {recommendation['recommended_provider']}")

    if args.launch_dashboard:
        launch_dashboard()
        print("Streamlit dashboard launched.")
    else:
        print("Run `streamlit run dashboard/app.py` to open the dashboard.")


if __name__ == "__main__":
    main()
