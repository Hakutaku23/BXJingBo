from __future__ import annotations

import json
from pathlib import Path

try:
    from interface import recommend_t90_controls
    from core import load_stage_aware_example_bundle
except ModuleNotFoundError:
    from .interface import recommend_t90_controls
    from .core import load_stage_aware_example_bundle


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"


def build_online_request() -> dict[str, object]:
    online_request = load_stage_aware_example_bundle(CONFIG_PATH, project_dir=PROJECT_DIR)
    online_request["skip_feature_ranking"] = True
    return online_request


def _build_result_summary(name: str, result: dict[str, object]) -> dict[str, object]:
    stage_decision = result.get("stage_decision", {})
    recommendation = result.get("recommendation", {})
    best_point = recommendation.get("best_point", {})
    return {
        "case": name,
        "stage": stage_decision.get("current_stage_name"),
        "ph_optional_input": stage_decision.get("ph_optional_input"),
        "ph_history_provided": stage_decision.get("ph_history_provided"),
        "ph_policy_enabled": stage_decision.get("ph_policy_enabled"),
        "ph_used": stage_decision.get("ph_used"),
        "ph_fallback_to_dcs_only": stage_decision.get("ph_fallback_to_dcs_only"),
        "selected_ph_lag_minutes": stage_decision.get("selected_ph_lag_minutes"),
        "recommended_calcium_range": recommendation.get("recommended_calcium_range"),
        "recommended_bromine_range": recommendation.get("recommended_bromine_range"),
        "best_point": {
            "calcium": best_point.get("calcium"),
            "bromine": best_point.get("bromine"),
            "in_spec_probability": best_point.get("in_spec_probability"),
        },
        "warnings": result.get("warnings", []),
    }


def main() -> None:
    online_request_with_ph = build_online_request()
    online_request_without_ph = {
        "dcs_window": online_request_with_ph["dcs_window"],
        "runtime_time": online_request_with_ph["runtime_time"],
        "skip_feature_ranking": True,
    }

    result_with_ph = recommend_t90_controls(online_request_with_ph)
    result_without_ph = recommend_t90_controls(online_request_without_ph)

    request_summary = {
        "runtime_time": online_request_with_ph.get("runtime_time"),
        "dcs_window_rows": len(online_request_with_ph["dcs_window"]),
        "ph_history_rows": len(online_request_with_ph["ph_history"]) if "ph_history" in online_request_with_ph else 0,
        "ph_is_optional": True,
    }
    comparison = {
        "request_summary": request_summary,
        "with_ph": _build_result_summary("with_ph", result_with_ph),
        "without_ph": _build_result_summary("without_ph", result_without_ph),
    }

    print("example_comparison")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
