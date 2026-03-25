from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from test import load_lims_grouped


DEFAULT_RESULTS_DIR = THIS_DIR / "artifacts"
DEFAULT_PH_PATH = PROJECT_DIR / "data" / "B4-AI-C53001A.PV.F_CV.xlsx"
DEFAULT_LIMS_PATH = PROJECT_DIR / "data" / "t90-溴丁橡胶.xlsx"


def load_ph_data(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    if raw.shape[1] < 3:
        raise ValueError(f"PH source {path} does not have the expected columns.")

    frame = raw.iloc[:, :4].copy()
    frame.columns = ["tag", "time", "value", "source"][: frame.shape[1]]
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"PH source {path} did not yield any valid rows.")
    return frame


def build_ph_features(ph: pd.DataFrame, feature_window_minutes: int) -> pd.DataFrame:
    if feature_window_minutes <= 1:
        raise ValueError("feature_window_minutes must be greater than 1.")

    frame = ph[["time", "value"]].copy()
    rolling = frame["value"].rolling(window=feature_window_minutes, min_periods=max(5, feature_window_minutes // 3))
    frame["ph_point"] = frame["value"]
    frame["ph_mean"] = rolling.mean()
    frame["ph_std"] = rolling.std()
    frame["ph_min"] = rolling.min()
    frame["ph_max"] = rolling.max()
    frame["ph_delta"] = frame["value"] - frame["value"].shift(feature_window_minutes - 1)
    return frame.drop(columns=["value"])


def build_lagged_dataset(
    *,
    lims: pd.DataFrame,
    ph_features: pd.DataFrame,
    lag_minutes: int,
    tolerance_minutes: int,
) -> pd.DataFrame:
    valid_lims = lims.loc[lims["t90"].notna(), ["sample_time", "t90"]].copy()
    valid_lims["lag_target_time"] = valid_lims["sample_time"] - pd.to_timedelta(lag_minutes, unit="m")

    aligned = pd.merge_asof(
        valid_lims.sort_values("lag_target_time"),
        ph_features.sort_values("time"),
        left_on="lag_target_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    aligned = aligned.dropna(subset=["ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta"]).copy()
    aligned["lag_minutes"] = lag_minutes
    aligned["alignment_error_minutes"] = (
        (aligned["time"] - aligned["lag_target_time"]).abs() / pd.Timedelta(minutes=1)
    ).astype(float)
    return aligned.sort_values("sample_time").reset_index(drop=True)


def evaluate_lag_dataset(aligned: pd.DataFrame, n_splits: int) -> dict[str, object]:
    feature_columns = ["ph_point", "ph_mean", "ph_std", "ph_min", "ph_max", "ph_delta"]
    clean = aligned.dropna(subset=["t90", *feature_columns]).copy()
    if len(clean) < max(40, n_splits + 5):
        return {
            "aligned_samples": int(len(clean)),
            "point_t90_corr": None,
            "mean_t90_corr": None,
            "delta_t90_corr": None,
            "max_abs_corr": None,
            "ridge_cv_r2_mean": None,
            "ridge_cv_mae_mean": None,
            "alignment_error_minutes_mean": None,
        }

    corr_point = float(clean["ph_point"].corr(clean["t90"]))
    corr_mean = float(clean["ph_mean"].corr(clean["t90"]))
    corr_delta = float(clean["ph_delta"].corr(clean["t90"]))

    X = clean[feature_columns]
    y = clean["t90"]
    splits = min(n_splits, len(clean) - 1)
    if splits < 2:
        return {
            "aligned_samples": int(len(clean)),
            "point_t90_corr": corr_point,
            "mean_t90_corr": corr_mean,
            "delta_t90_corr": corr_delta,
            "max_abs_corr": max(abs(corr_point), abs(corr_mean), abs(corr_delta)),
            "ridge_cv_r2_mean": None,
            "ridge_cv_mae_mean": None,
            "alignment_error_minutes_mean": float(clean["alignment_error_minutes"].mean()),
        }

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    splitter = TimeSeriesSplit(n_splits=splits)
    r2_scores: list[float] = []
    mae_scores: list[float] = []
    for train_index, test_index in splitter.split(X):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        prediction = model.predict(X.iloc[test_index])
        r2_scores.append(float(r2_score(y.iloc[test_index], prediction)))
        mae_scores.append(float(mean_absolute_error(y.iloc[test_index], prediction)))

    return {
        "aligned_samples": int(len(clean)),
        "point_t90_corr": corr_point,
        "mean_t90_corr": corr_mean,
        "delta_t90_corr": corr_delta,
        "max_abs_corr": max(abs(corr_point), abs(corr_mean), abs(corr_delta)),
        "ridge_cv_r2_mean": float(np.mean(r2_scores)),
        "ridge_cv_mae_mean": float(np.mean(mae_scores)),
        "alignment_error_minutes_mean": float(clean["alignment_error_minutes"].mean()),
    }


def create_plot(summary: pd.DataFrame, output_path: Path, focus_low: int, focus_high: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(summary["lag_minutes"], summary["aligned_samples"], marker="o", color="#1f77b4")
    axes[0, 0].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[0, 0].set_title("Aligned samples by PH lag")
    axes[0, 0].set_xlabel("Lag minutes")
    axes[0, 0].set_ylabel("Samples")

    axes[0, 1].plot(summary["lag_minutes"], summary["point_t90_corr"], marker="o", label="point corr", color="#d62728")
    axes[0, 1].plot(summary["lag_minutes"], summary["mean_t90_corr"], marker="o", label="mean corr", color="#2ca02c")
    axes[0, 1].plot(summary["lag_minutes"], summary["delta_t90_corr"], marker="o", label="delta corr", color="#9467bd")
    axes[0, 1].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[0, 1].set_title("PH to T90 correlation by lag")
    axes[0, 1].set_xlabel("Lag minutes")
    axes[0, 1].set_ylabel("Correlation")
    axes[0, 1].legend()

    axes[1, 0].plot(summary["lag_minutes"], summary["ridge_cv_r2_mean"], marker="o", color="#ff7f0e", label="ridge CV R2")
    axes[1, 0].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[1, 0].set_title("PH feature explanatory power by lag")
    axes[1, 0].set_xlabel("Lag minutes")
    axes[1, 0].set_ylabel("Cross-validated R2")
    axes[1, 0].legend()

    axes[1, 1].plot(summary["lag_minutes"], summary["ridge_cv_mae_mean"], marker="o", color="#8c564b", label="ridge CV MAE")
    axes[1, 1].axvspan(focus_low, focus_high, color="#ffdd99", alpha=0.25)
    axes[1, 1].set_title("PH feature error by lag")
    axes[1, 1].set_xlabel("Lag minutes")
    axes[1, 1].set_ylabel("Cross-validated MAE")
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(summary: pd.DataFrame, *, focus_low: int, focus_high: int, plot_path: Path) -> dict[str, object]:
    if summary.empty:
        return {
            "message": "No PH lag rows were generated.",
            "plot_path": str(plot_path),
        }

    valid_r2 = summary.dropna(subset=["ridge_cv_r2_mean"]).copy()
    valid_corr = summary.dropna(subset=["max_abs_corr"]).copy()
    focus = summary.loc[summary["lag_minutes"].between(focus_low, focus_high)].copy()
    focus_r2 = focus.dropna(subset=["ridge_cv_r2_mean"]).copy()
    focus_corr = focus.dropna(subset=["max_abs_corr"]).copy()

    best_r2 = valid_r2.sort_values("ridge_cv_r2_mean", ascending=False).iloc[0] if not valid_r2.empty else None
    best_corr = valid_corr.sort_values("max_abs_corr", ascending=False).iloc[0] if not valid_corr.empty else None
    best_focus_r2 = focus_r2.sort_values("ridge_cv_r2_mean", ascending=False).iloc[0] if not focus_r2.empty else None
    best_focus_corr = focus_corr.sort_values("max_abs_corr", ascending=False).iloc[0] if not focus_corr.empty else None

    return {
        "plot_path": str(plot_path),
        "focus_window_minutes": [focus_low, focus_high],
        "supports_process_claim": bool(
            best_r2 is not None and focus_low <= int(best_r2["lag_minutes"]) <= focus_high
        ),
        "best_lag_by_ridge_cv_r2": None if best_r2 is None else int(best_r2["lag_minutes"]),
        "best_lag_by_abs_correlation": None if best_corr is None else int(best_corr["lag_minutes"]),
        "best_focus_lag_by_ridge_cv_r2": None if best_focus_r2 is None else int(best_focus_r2["lag_minutes"]),
        "best_focus_lag_by_abs_correlation": None if best_focus_corr is None else int(best_focus_corr["lag_minutes"]),
        "top_r2_lags": [] if valid_r2.empty else valid_r2.sort_values("ridge_cv_r2_mean", ascending=False).head(10).to_dict(orient="records"),
        "top_abs_corr_lags": [] if valid_corr.empty else valid_corr.sort_values("max_abs_corr", ascending=False).head(10).to_dict(orient="records"),
        "rows": summary.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PH to T90 lag on offline data in dev only.")
    parser.add_argument("--ph-path", default=str(DEFAULT_PH_PATH), help="PH Excel path.")
    parser.add_argument("--lims-path", default=str(DEFAULT_LIMS_PATH), help="LIMS Excel path.")
    parser.add_argument("--min-lag", type=int, default=0, help="Minimum lag in minutes.")
    parser.add_argument("--max-lag", type=int, default=360, help="Maximum lag in minutes.")
    parser.add_argument("--step", type=int, default=5, help="Lag step in minutes.")
    parser.add_argument("--feature-window", type=int, default=30, help="Rolling PH feature window in minutes.")
    parser.add_argument("--tolerance", type=int, default=2, help="Merge tolerance in minutes.")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit count.")
    parser.add_argument("--focus-low", type=int, default=180, help="Lower lag bound of the process-claimed focus window.")
    parser.add_argument("--focus-high", type=int, default=270, help="Upper lag bound of the process-claimed focus window.")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix.")
    args = parser.parse_args()

    if args.min_lag < 0 or args.max_lag < args.min_lag or args.step <= 0:
        raise ValueError("Lag arguments must satisfy 0 <= min-lag <= max-lag and step > 0.")

    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    summary_path = DEFAULT_RESULTS_DIR / f"{prefix}ph_lag_experiment_summary.csv"
    report_path = DEFAULT_RESULTS_DIR / f"{prefix}ph_lag_experiment_summary.json"
    plot_path = DEFAULT_RESULTS_DIR / f"{prefix}ph_lag_experiment_metrics.png"

    lims = load_lims_grouped(args.lims_path)
    ph = load_ph_data(args.ph_path)
    ph_features = build_ph_features(ph, feature_window_minutes=args.feature_window)

    rows: list[dict[str, object]] = []
    for lag_minutes in range(args.min_lag, args.max_lag + 1, args.step):
        print(f"evaluating lag {lag_minutes} minutes")
        aligned = build_lagged_dataset(
            lims=lims,
            ph_features=ph_features,
            lag_minutes=lag_minutes,
            tolerance_minutes=args.tolerance,
        )
        metrics = evaluate_lag_dataset(aligned, n_splits=args.cv_splits)
        rows.append({"lag_minutes": lag_minutes, **metrics})

    summary = pd.DataFrame(rows).sort_values("lag_minutes").reset_index(drop=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    create_plot(summary, plot_path, focus_low=args.focus_low, focus_high=args.focus_high)

    report = build_report(summary, focus_low=args.focus_low, focus_high=args.focus_high, plot_path=plot_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    best_r2 = report.get("best_lag_by_ridge_cv_r2")
    best_corr = report.get("best_lag_by_abs_correlation")
    print(f"best lag by ridge CV R2: {best_r2}")
    print(f"best lag by abs correlation: {best_corr}")
    print(f"supports process claim (focus window {args.focus_low}-{args.focus_high} min): {report['supports_process_claim']}")
    print(f"summary written to {summary_path}")
    print(f"report written to {report_path}")
    print(f"plot written to {plot_path}")


if __name__ == "__main__":
    main()
