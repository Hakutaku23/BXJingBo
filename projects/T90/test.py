from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from core import build_casebase_from_windows, get_context_sensors, get_target_range, load_runtime_config, save_casebase_dataset
    from core.online_recommender import _build_control_recommendation, _fit_local_control_model
except ModuleNotFoundError:
    from .core import build_casebase_from_windows, get_context_sensors, get_target_range, load_runtime_config, save_casebase_dataset
    from .core.online_recommender import _build_control_recommendation, _fit_local_control_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "t90_runtime.yaml"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "dev" / "artifacts"
SENSOR_SUFFIXES = (
    "_pv_f_cv",
    "_s_pv_cv",
    "_pv_cv",
    "_pv_f",
    "_s_pv",
    "_pv",
    "_cv",
    "_s",
)


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _normalize_sensor_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().replace(".", "_")
    text = re.sub(r"[^0-9a-z_]+", "", text)
    for suffix in SENSOR_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip("_")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("≤", "", regex=False)
        .str.replace("≥", "", regex=False)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )
    text = text.str.replace(r"[^0-9eE+\-\.]+", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _first_valid(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _infer_lims_columns(columns: list[str]) -> dict[str, str]:
    normalized = {_normalize_name(column): column for column in columns}

    def match(*patterns: str, excludes: tuple[str, ...] = ()) -> str | None:
        for pattern in patterns:
            for normalized_name, original_name in normalized.items():
                if pattern in normalized_name and all(excluded not in normalized_name for excluded in excludes):
                    return original_name
        return None

    inferred = {
        "sample_time": match("采样时间"),
        "t90": match("tc90", "t90min"),
        "bromine": match("溴含量"),
        "calcium_stearate": match("硬脂酸钙含量"),
        "calcium": match("钙含量", excludes=("硬脂酸钙含量",)),
        "volatile": match("挥发分", excludes=("在线监测挥发分",)),
        "online_volatile": match("在线监测挥发分"),
        "stabilizer": match("稳定剂"),
        "mooney": match("门尼粘度"),
        "antioxidant": match("防老剂"),
    }
    if inferred["sample_time"] is None or inferred["t90"] is None:
        raise ValueError("Unable to identify required LIMS columns for sample_time and t90.")
    return inferred


def load_lims_grouped(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    columns = _infer_lims_columns(raw.columns.tolist())

    prepared = pd.DataFrame()
    prepared["sample_time"] = pd.to_datetime(raw[columns["sample_time"]], errors="coerce")
    for output_name, column_name in columns.items():
        if output_name == "sample_time" or column_name is None:
            continue
        if output_name == "online_volatile":
            prepared[output_name] = pd.to_numeric(raw[column_name], errors="coerce")
        else:
            prepared[output_name] = _coerce_numeric(raw[column_name])

    grouped = prepared.groupby("sample_time", as_index=False).agg(_first_valid)
    grouped["calcium"] = grouped["calcium"].combine_first(grouped["calcium_stearate"])
    grouped = grouped.sort_values("sample_time").reset_index(drop=True)
    return grouped


def _load_dcs_source(path: str | Path, sensors: list[str], time_column: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    requested = {_normalize_sensor_name(sensor): sensor for sensor in sensors}
    selected_columns: dict[str, str] = {}

    for actual_column in header:
        if actual_column == time_column:
            continue
        sensor_name = requested.get(_normalize_sensor_name(actual_column))
        if sensor_name is None:
            continue
        if sensor_name not in selected_columns or actual_column == sensor_name:
            selected_columns[sensor_name] = actual_column

    usecols = [time_column, *selected_columns.values()]
    frame = pd.read_csv(path, usecols=usecols)
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=[time_column]).copy()
    frame = frame.rename(columns={actual: canonical for canonical, actual in selected_columns.items()})
    for sensor in sensors:
        if sensor not in frame.columns:
            frame[sensor] = np.nan
        frame[sensor] = pd.to_numeric(frame[sensor], errors="coerce")
    return frame[[time_column, *sensors]]


def load_dcs_data(paths: list[str], sensors: list[str], time_column: str = "time") -> pd.DataFrame:
    frames = [_load_dcs_source(PROJECT_DIR / path, sensors=sensors, time_column=time_column) for path in paths]
    dcs = pd.concat(frames, ignore_index=True)
    dcs = dcs.sort_values(time_column).drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)
    return dcs


def build_windows_and_outcomes(
    dcs: pd.DataFrame,
    lims: pd.DataFrame,
    window_minutes: int,
    *,
    time_column: str = "time",
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    dcs = dcs.sort_values(time_column).reset_index(drop=True)
    time_index = pd.to_datetime(dcs[time_column]).to_numpy(dtype="datetime64[ns]")

    windows: list[pd.DataFrame] = []
    outcomes: list[dict[str, object]] = []

    required_columns = ["sample_time", "t90", "calcium", "bromine"]
    available_columns = [column for column in lims.columns if column not in required_columns]

    for row in lims.itertuples(index=False):
        sample_time = pd.Timestamp(row.sample_time)
        if pd.isna(sample_time) or pd.isna(row.t90) or pd.isna(row.calcium) or pd.isna(row.bromine):
            continue

        end_index = np.searchsorted(time_index, sample_time.to_datetime64(), side="right") - 1
        if end_index < window_minutes - 1:
            continue

        window = dcs.iloc[end_index - window_minutes + 1 : end_index + 1].copy()
        if len(window) != window_minutes:
            continue

        window_times = pd.to_datetime(window[time_column])
        if sample_time - window_times.iloc[-1] > pd.Timedelta(minutes=2):
            continue
        if window_times.iloc[-1] - window_times.iloc[0] > pd.Timedelta(minutes=window_minutes + 2):
            continue

        windows.append(window.reset_index(drop=True))
        outcome = {
            "sample_time": sample_time,
            "t90": float(row.t90),
            "calcium": float(row.calcium),
            "bromine": float(row.bromine),
        }
        row_dict = row._asdict()
        for column in available_columns:
            outcome[column] = row_dict.get(column)
        outcomes.append(outcome)

    return windows, pd.DataFrame(outcomes)


def build_casebase_from_private_data(config_path: str | Path) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = load_runtime_config(config_path)
    sensors = get_context_sensors(config)
    target_low, target_high = get_target_range(config)
    window_minutes = int(config.get("window", {}).get("minutes", 15))
    time_column = str(config.get("window", {}).get("time_column", "time"))
    data_sources = config.get("data_sources", {})

    dcs = load_dcs_data(data_sources.get("dcs_paths", []), sensors=sensors, time_column=time_column)
    lims = load_lims_grouped(PROJECT_DIR / str(data_sources["lims_path"]))
    windows, outcomes = build_windows_and_outcomes(dcs, lims, window_minutes, time_column=time_column)
    casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=target_low,
        target_high=target_high,
        include_sensors=sensors,
    )
    return casebase, windows, outcomes, dcs, config


def _extract_minmax(range_value: dict[str, object] | None) -> tuple[float | None, float | None]:
    if not range_value:
        return None, None
    minimum = range_value.get("min", range_value.get("calcium_min", range_value.get("bromine_min")))
    maximum = range_value.get("max", range_value.get("calcium_max", range_value.get("bromine_max")))
    return (
        None if minimum is None else float(minimum),
        None if maximum is None else float(maximum),
    )


def evaluate_recommendations(
    casebase: pd.DataFrame,
    *,
    limit: int = 0,
    neighbor_count: int = 150,
    local_neighbor_count: int = 80,
    probability_threshold: float = 0.60,
    grid_points: int = 31,
) -> pd.DataFrame:
    total = len(casebase) if limit <= 0 else min(limit, len(casebase))
    rows: list[dict[str, object]] = []
    blocked = {"sample_time", "t90", "is_in_spec", "calcium", "bromine"}
    context_columns = [
        column
        for column in casebase.select_dtypes(include=["number"]).columns
        if column not in blocked
    ]
    if not context_columns:
        raise ValueError("No encoded context columns were found in the rebuilt casebase.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    context_matrix = scaler.fit_transform(imputer.fit_transform(casebase[context_columns]))
    k = min(neighbor_count + 1, len(casebase))
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(context_matrix)
    distances, indices = neighbors.kneighbors(context_matrix)

    for index in range(total):
        if index and index % 100 == 0:
            print(f"evaluated {index}/{total} samples")

        current_row = casebase.iloc[index]
        neighbor_pairs = [
            (neighbor_index, neighbor_distance)
            for neighbor_index, neighbor_distance in zip(indices[index], distances[index])
            if neighbor_index != index
        ][:neighbor_count]
        if not neighbor_pairs:
            continue

        try:
            neighborhood = casebase.iloc[[neighbor_index for neighbor_index, _ in neighbor_pairs]].copy().reset_index(drop=True)
            neighborhood["context_distance"] = [neighbor_distance for _, neighbor_distance in neighbor_pairs]
            local, model = _fit_local_control_model(neighborhood, min(local_neighbor_count, len(neighborhood)))
            recommendation = _build_control_recommendation(
                local,
                model,
                reference_calcium=None,
                reference_bromine=None,
                probability_threshold=probability_threshold,
                grid_points=grid_points,
            )
            best_point = recommendation.get("best_point", {})
            calcium_min, calcium_max = _extract_minmax(recommendation.get("recommended_calcium_range"))
            bromine_min, bromine_max = _extract_minmax(recommendation.get("recommended_bromine_range"))
            best_calcium = best_point.get("calcium")
            best_bromine = best_point.get("bromine")
            calcium_error = None if best_calcium is None else abs(float(best_calcium) - float(current_row["calcium"]))
            bromine_error = None if best_bromine is None else abs(float(best_bromine) - float(current_row["bromine"]))
            calcium_inside = calcium_min is not None and calcium_max is not None and calcium_min <= float(current_row["calcium"]) <= calcium_max
            bromine_inside = bromine_min is not None and bromine_max is not None and bromine_min <= float(current_row["bromine"]) <= bromine_max

            rows.append(
                {
                    "sample_time": current_row["sample_time"],
                    "t90": float(current_row["t90"]),
                    "is_in_spec": bool(current_row["is_in_spec"]),
                    "actual_calcium": float(current_row["calcium"]),
                    "actual_bromine": float(current_row["bromine"]),
                    "recommended_calcium_min": calcium_min,
                    "recommended_calcium_max": calcium_max,
                    "recommended_bromine_min": bromine_min,
                    "recommended_bromine_max": bromine_max,
                    "recommended_best_calcium": None if best_calcium is None else float(best_calcium),
                    "recommended_best_bromine": None if best_bromine is None else float(best_bromine),
                    "recommended_probability": best_point.get("in_spec_probability"),
                    "actual_calcium_inside_range": calcium_inside,
                    "actual_bromine_inside_range": bromine_inside,
                    "calcium_abs_error_to_best": calcium_error,
                    "bromine_abs_error_to_best": bromine_error,
                    "recommendation_status": recommendation.get("status"),
                    "error_message": None,
                }
            )
        except Exception as exc:  # pragma: no cover - kept for robust offline validation
            rows.append(
                {
                    "sample_time": current_row["sample_time"],
                    "t90": float(current_row["t90"]),
                    "is_in_spec": bool(current_row["is_in_spec"]),
                    "actual_calcium": float(current_row["calcium"]),
                    "actual_bromine": float(current_row["bromine"]),
                    "recommended_calcium_min": None,
                    "recommended_calcium_max": None,
                    "recommended_bromine_min": None,
                    "recommended_bromine_max": None,
                    "recommended_best_calcium": None,
                    "recommended_best_bromine": None,
                    "recommended_probability": None,
                    "actual_calcium_inside_range": False,
                    "actual_bromine_inside_range": False,
                    "calcium_abs_error_to_best": None,
                    "bromine_abs_error_to_best": None,
                    "recommendation_status": "error",
                    "error_message": str(exc),
                }
            )

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> dict[str, object]:
    successful = results.loc[results["error_message"].isna()].copy()
    in_spec = successful.loc[successful["is_in_spec"]]

    def ratio(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty:
            return None
        return float(frame[column].mean())

    summary = {
        "aligned_samples": int(len(results)),
        "successful_recommendations": int(len(successful)),
        "failed_recommendations": int(results["error_message"].notna().sum()),
        "in_spec_samples": int(in_spec.shape[0]),
        "actual_calcium_inside_recommended_range_ratio": ratio(successful, "actual_calcium_inside_range"),
        "actual_bromine_inside_recommended_range_ratio": ratio(successful, "actual_bromine_inside_range"),
        "in_spec_actual_calcium_inside_recommended_range_ratio": ratio(in_spec, "actual_calcium_inside_range"),
        "in_spec_actual_bromine_inside_recommended_range_ratio": ratio(in_spec, "actual_bromine_inside_range"),
        "mean_calcium_abs_error_to_best": None if successful["calcium_abs_error_to_best"].dropna().empty else float(successful["calcium_abs_error_to_best"].dropna().mean()),
        "mean_bromine_abs_error_to_best": None if successful["bromine_abs_error_to_best"].dropna().empty else float(successful["bromine_abs_error_to_best"].dropna().mean()),
        "median_calcium_abs_error_to_best": None if successful["calcium_abs_error_to_best"].dropna().empty else float(successful["calcium_abs_error_to_best"].dropna().median()),
        "median_bromine_abs_error_to_best": None if successful["bromine_abs_error_to_best"].dropna().empty else float(successful["bromine_abs_error_to_best"].dropna().median()),
    }
    return summary


def create_visualizations(results: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    successful = results.loc[results["error_message"].isna()].copy()
    successful["spec_label"] = np.where(successful["is_in_spec"], "in_spec", "out_of_spec")

    comparison_figure = output_path / "recommendation_vs_actual.png"
    error_figure = output_path / "recommendation_error_distribution.png"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_specs = [
        ("actual_calcium", "recommended_best_calcium", "Calcium"),
        ("actual_bromine", "recommended_best_bromine", "Bromine"),
    ]
    colors = {"in_spec": "#1f77b4", "out_of_spec": "#d62728"}

    for axis, (actual_column, recommended_column, label) in zip(axes[0], plot_specs):
        data = successful.dropna(subset=[actual_column, recommended_column])
        for spec_label, color in colors.items():
            subset = data.loc[data["spec_label"] == spec_label]
            axis.scatter(
                subset[actual_column],
                subset[recommended_column],
                s=16,
                alpha=0.7,
                color=color,
                label=spec_label,
            )
        if not data.empty:
            minimum = float(min(data[actual_column].min(), data[recommended_column].min()))
            maximum = float(max(data[actual_column].max(), data[recommended_column].max()))
            axis.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="#444444", linewidth=1.0)
        axis.set_title(f"{label}: actual vs recommended")
        axis.set_xlabel(f"Actual {label.lower()}")
        axis.set_ylabel(f"Recommended {label.lower()}")
        axis.legend()

    for axis, (error_column, label) in zip(
        axes[1],
        [
            ("calcium_abs_error_to_best", "Calcium absolute error"),
            ("bromine_abs_error_to_best", "Bromine absolute error"),
        ],
    ):
        data = successful[error_column].dropna()
        axis.hist(data, bins=30, color="#4c78a8", alpha=0.85, edgecolor="white")
        axis.set_title(label)
        axis.set_xlabel("Absolute error")
        axis.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(comparison_figure, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, (actual_column, range_min_column, range_max_column, label) in zip(
        axes,
        [
            ("actual_calcium", "recommended_calcium_min", "recommended_calcium_max", "Calcium"),
            ("actual_bromine", "recommended_bromine_min", "recommended_bromine_max", "Bromine"),
        ],
    ):
        data = successful.dropna(subset=[actual_column, range_min_column, range_max_column]).copy()
        if not data.empty:
            x_index = np.arange(len(data))
            axis.plot(x_index, data[actual_column].to_numpy(), color="#222222", linewidth=1.0, label="actual")
            axis.fill_between(
                x_index,
                data[range_min_column].to_numpy(),
                data[range_max_column].to_numpy(),
                color="#9ecae1",
                alpha=0.5,
                label="recommended range",
            )
        axis.set_title(f"{label}: actual value inside recommended band")
        axis.set_xlabel("Sample index")
        axis.set_ylabel(label.lower())
        axis.legend()

    fig.tight_layout()
    fig.savefig(error_figure, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "comparison_plot": str(comparison_figure),
        "range_band_plot": str(error_figure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay all private T90 data and compare recommendations against LIMS truth.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Runtime YAML config path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluated aligned samples. Default 0 means all samples.")
    parser.add_argument("--casebase-output", default=None, help="Optional csv/parquet output path for the rebuilt casebase.")
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    casebase, windows, outcomes, _dcs, config = build_casebase_from_private_data(config_path)
    casebase_output = Path(args.casebase_output) if args.casebase_output else PROJECT_DIR / config["artifacts"]["casebase_path"]
    save_casebase_dataset(casebase, casebase_output)

    results = evaluate_recommendations(
        casebase,
        limit=args.limit,
    )
    summary = summarize_results(results)
    plots = create_visualizations(results, results_dir)
    summary["plots"] = plots
    summary["casebase_path"] = str(casebase_output)

    results_path = results_dir / "test_recommendation_results.csv"
    summary_path = results_dir / "test_recommendation_summary.json"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"casebase_saved_to={casebase_output}")
    print(f"results_saved_to={results_path}")
    print(f"summary_saved_to={summary_path}")


if __name__ == "__main__":
    main()
