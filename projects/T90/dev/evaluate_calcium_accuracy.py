from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_DIR / "dev" / "artifacts" / "test_recommendation_results.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "dev" / "artifacts" / "calcium_accuracy_summary.json"


def _safe_stat(series: pd.Series, fn: str) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    if fn == "mean":
        return float(clean.mean())
    if fn == "median":
        return float(clean.median())
    if fn == "max":
        return float(clean.max())
    if fn == "p90":
        return float(clean.quantile(0.90))
    raise ValueError(f"Unsupported stat function: {fn}")


def _safe_ratio(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())


def summarize_calcium_accuracy(results: pd.DataFrame) -> dict[str, object]:
    successful = results.loc[results["error_message"].isna()].copy()
    in_spec = successful.loc[successful["is_in_spec"]]
    out_of_spec = successful.loc[~successful["is_in_spec"]]

    summary = {
        "evaluated_samples": int(len(results)),
        "successful_samples": int(len(successful)),
        "failed_samples": int(results["error_message"].notna().sum()),
        "in_spec_samples": int(len(in_spec)),
        "out_of_spec_samples": int(len(out_of_spec)),
        "overall": {
            "mean_abs_error": _safe_stat(successful["calcium_abs_error_to_best"], "mean"),
            "median_abs_error": _safe_stat(successful["calcium_abs_error_to_best"], "median"),
            "p90_abs_error": _safe_stat(successful["calcium_abs_error_to_best"], "p90"),
            "max_abs_error": _safe_stat(successful["calcium_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": _safe_ratio(successful, "actual_calcium_inside_range"),
        },
        "in_spec_only": {
            "mean_abs_error": _safe_stat(in_spec["calcium_abs_error_to_best"], "mean"),
            "median_abs_error": _safe_stat(in_spec["calcium_abs_error_to_best"], "median"),
            "p90_abs_error": _safe_stat(in_spec["calcium_abs_error_to_best"], "p90"),
            "max_abs_error": _safe_stat(in_spec["calcium_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": _safe_ratio(in_spec, "actual_calcium_inside_range"),
        },
        "out_of_spec_only": {
            "mean_abs_error": _safe_stat(out_of_spec["calcium_abs_error_to_best"], "mean"),
            "median_abs_error": _safe_stat(out_of_spec["calcium_abs_error_to_best"], "median"),
            "p90_abs_error": _safe_stat(out_of_spec["calcium_abs_error_to_best"], "p90"),
            "max_abs_error": _safe_stat(out_of_spec["calcium_abs_error_to_best"], "max"),
            "inside_recommended_range_ratio": _safe_ratio(out_of_spec, "actual_calcium_inside_range"),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calcium recommendation accuracy from offline replay results.")
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to projects/T90/test.py output csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Optional JSON output path for the calcium accuracy summary.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. Please run `python projects/T90/test.py` first."
        )

    results = pd.read_csv(results_path)
    summary = summarize_calcium_accuracy(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_saved_to={output_path}")


if __name__ == "__main__":
    main()
