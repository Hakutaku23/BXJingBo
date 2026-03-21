from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "dev" / "artifacts"
DEFAULT_FEATURE_TABLE = ARTIFACTS_DIR / "t90_feature_table.csv"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "cabr_range_recommendation.json"

CONTROL_COLUMNS = ["calcium", "bromine"]
BLOCKED_COLUMNS = {
    "sample_time",
    "sheet_name",
    "t90",
    "t90_in_spec",
    "target_class",
    "calcium_stearate",
    *CONTROL_COLUMNS,
}


def load_feature_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["sample_time"])
    frame["target_class"] = frame["t90"].between(7.5, 8.5).astype(float)
    return frame


def choose_context_columns(frame: pd.DataFrame, include_lab_context: bool = False) -> list[str]:
    columns = [col for col in frame.columns if col not in BLOCKED_COLUMNS]
    if not include_lab_context:
        columns = [col for col in columns if col != "volatile_lab"]
    return columns


def rank_context_features(
    frame: pd.DataFrame,
    context_columns: list[str],
    top_n: int,
) -> tuple[list[str], dict[str, float]]:
    usable = frame.dropna(subset=["t90", *CONTROL_COLUMNS]).copy()
    usable = usable.dropna(axis=1, how="all")
    context_columns = [col for col in context_columns if col in usable.columns]
    X = usable[context_columns]
    y = usable["target_class"].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    importances = pd.Series(
        model.named_steps["rf"].feature_importances_,
        index=context_columns,
    ).sort_values(ascending=False)
    selected = importances.head(top_n).index.tolist()
    return selected, importances.head(top_n).round(6).to_dict()


def pick_target_sample(frame: pd.DataFrame, sample_time: str | None) -> pd.Series:
    usable = frame.dropna(subset=["t90", *CONTROL_COLUMNS]).copy()
    if sample_time:
        target_time = pd.to_datetime(sample_time)
        matched = usable.loc[usable["sample_time"] == target_time]
        if matched.empty:
            raise ValueError(f"Sample time not found: {sample_time}")
        return matched.sort_values("sample_time").iloc[-1]

    bad = usable.loc[~usable["t90"].between(7.5, 8.5)].sort_values("sample_time")
    if bad.empty:
        raise ValueError("No out-of-spec sample with calcium and bromine available.")
    return bad.iloc[-1]


def find_context_neighborhood(
    frame: pd.DataFrame,
    target_row: pd.Series,
    context_features: list[str],
    neighbor_count: int,
) -> pd.DataFrame:
    usable = frame.dropna(subset=["t90", *CONTROL_COLUMNS]).copy()
    usable = usable.dropna(axis=1, how="all")
    context_features = [col for col in context_features if col in usable.columns]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    context_matrix = imputer.fit_transform(usable[context_features])
    context_matrix = scaler.fit_transform(context_matrix)
    target_matrix = pd.DataFrame([target_row[context_features]], columns=context_features)
    target_vector = scaler.transform(imputer.transform(target_matrix))

    k = min(neighbor_count, len(usable))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(context_matrix)
    distances, indices = model.kneighbors(target_vector)
    neighborhood = usable.iloc[indices[0]].copy()
    neighborhood["context_distance"] = distances[0]
    neighborhood = neighborhood.sort_values("context_distance").reset_index(drop=True)
    return neighborhood


def summarize_empirical_ranges(good_neighbors: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for column in CONTROL_COLUMNS:
        series = good_neighbors[column].dropna()
        if series.empty:
            result[column] = None
            continue
        result[column] = {
            "p10": float(series.quantile(0.10)),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return result


def build_feasible_region(
    neighborhood: pd.DataFrame,
    target_row: pd.Series,
    probability_threshold: float,
    grid_points: int,
) -> dict[str, object]:
    usable = neighborhood.dropna(subset=["t90", *CONTROL_COLUMNS]).copy()
    usable["target_class"] = usable["t90"].between(7.5, 8.5).astype(int)
    good = usable.loc[usable["target_class"] == 1].copy()
    bad = usable.loc[usable["target_class"] == 0].copy()
    if len(good) < 20 or usable["target_class"].nunique() < 2:
        return {
            "status": "not_enough_local_support",
            "good_neighbors": int(len(good)),
            "bad_neighbors": int(len(bad)),
        }

    X = usable[CONTROL_COLUMNS]
    y = usable["target_class"]
    local_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    local_model.fit(X, y)

    calcium_pool = usable["calcium"].dropna()
    bromine_pool = usable["bromine"].dropna()
    calcium_grid = np.linspace(
        calcium_pool.quantile(0.05),
        calcium_pool.quantile(0.95),
        grid_points,
    )
    bromine_grid = np.linspace(
        bromine_pool.quantile(0.05),
        bromine_pool.quantile(0.95),
        grid_points,
    )
    mesh = pd.DataFrame(
        [(ca, br) for ca in calcium_grid for br in bromine_grid],
        columns=CONTROL_COLUMNS,
    )
    probabilities = local_model.predict_proba(mesh)[:, 1]
    mesh["in_spec_probability"] = probabilities
    feasible = mesh.loc[mesh["in_spec_probability"] >= probability_threshold].copy()
    if feasible.empty:
        feasible = mesh.nlargest(max(10, grid_points), "in_spec_probability").copy()

    best_point = mesh.loc[mesh["in_spec_probability"].idxmax()]

    calcium_at_current_bromine = pd.DataFrame(
        {
            "calcium": calcium_grid,
            "bromine": float(target_row["bromine"]),
        }
    )
    calcium_at_current_bromine["in_spec_probability"] = local_model.predict_proba(calcium_at_current_bromine)[:, 1]
    feasible_calcium_at_current_bromine = calcium_at_current_bromine.loc[
        calcium_at_current_bromine["in_spec_probability"] >= probability_threshold
    ]

    bromine_at_current_calcium = pd.DataFrame(
        {
            "calcium": float(target_row["calcium"]),
            "bromine": bromine_grid,
        }
    )
    bromine_at_current_calcium["in_spec_probability"] = local_model.predict_proba(bromine_at_current_calcium)[:, 1]
    feasible_bromine_at_current_calcium = bromine_at_current_calcium.loc[
        bromine_at_current_calcium["in_spec_probability"] >= probability_threshold
    ]

    return {
        "status": "ok",
        "good_neighbors": int(len(good)),
        "bad_neighbors": int(len(bad)),
        "empirical_good_ranges": summarize_empirical_ranges(good),
        "model_based_ranges": {
            "calcium": {
                "min": float(feasible["calcium"].min()),
                "max": float(feasible["calcium"].max()),
            },
            "bromine": {
                "min": float(feasible["bromine"].min()),
                "max": float(feasible["bromine"].max()),
            },
        },
        "best_point": {
            "calcium": float(best_point["calcium"]),
            "bromine": float(best_point["bromine"]),
            "in_spec_probability": float(best_point["in_spec_probability"]),
        },
        "calcium_range_given_current_bromine": None
        if feasible_calcium_at_current_bromine.empty
        else {
            "bromine_fixed_at": float(target_row["bromine"]),
            "calcium_min": float(feasible_calcium_at_current_bromine["calcium"].min()),
            "calcium_max": float(feasible_calcium_at_current_bromine["calcium"].max()),
            "max_probability": float(feasible_calcium_at_current_bromine["in_spec_probability"].max()),
        },
        "bromine_range_given_current_calcium": None
        if feasible_bromine_at_current_calcium.empty
        else {
            "calcium_fixed_at": float(target_row["calcium"]),
            "bromine_min": float(feasible_bromine_at_current_calcium["bromine"].min()),
            "bromine_max": float(feasible_bromine_at_current_calcium["bromine"].max()),
            "max_probability": float(feasible_bromine_at_current_calcium["in_spec_probability"].max()),
        },
    }


def build_result(
    frame: pd.DataFrame,
    target_row: pd.Series,
    context_features: list[str],
    feature_importance: dict[str, float],
    neighborhood: pd.DataFrame,
    feasible_region: dict[str, object],
) -> dict[str, object]:
    context_snapshot = {}
    for feature in context_features[:10]:
        value = target_row.get(feature, np.nan)
        if pd.notna(value):
            context_snapshot[feature] = float(value)

    return {
        "target_sample": {
            "sample_time": str(target_row["sample_time"]),
            "t90": float(target_row["t90"]),
            "current_calcium": float(target_row["calcium"]),
            "current_bromine": float(target_row["bromine"]),
            "is_in_spec": bool(7.5 <= target_row["t90"] <= 8.5),
        },
        "method": {
            "goal": "Recommend feasible calcium and bromine ranges under current observed process context.",
            "primary_control_priority": "calcium",
            "context_feature_count": len(context_features),
            "top_context_features": feature_importance,
            "context_snapshot": context_snapshot,
        },
        "neighborhood": {
            "samples": int(len(neighborhood)),
            "good_samples": int(neighborhood["t90"].between(7.5, 8.5).sum()),
            "bad_samples": int((~neighborhood["t90"].between(7.5, 8.5)).sum()),
            "median_context_distance": float(neighborhood["context_distance"].median()),
        },
        "recommendation": feasible_region,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend calcium/bromine feasible ranges under the current process context.")
    parser.add_argument("--feature-table", type=str, default=str(DEFAULT_FEATURE_TABLE))
    parser.add_argument("--sample-time", type=str, default="", help="Target sample time in YYYY-MM-DD HH:MM:SS format. Defaults to latest out-of-spec sample.")
    parser.add_argument("--top-context-features", type=int, default=20)
    parser.add_argument("--neighbor-count", type=int, default=400)
    parser.add_argument("--probability-threshold", type=float, default=0.60)
    parser.add_argument("--grid-points", type=int, default=41)
    parser.add_argument("--include-lab-context", action="store_true")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    frame = load_feature_table(Path(args.feature_table))
    context_columns = choose_context_columns(frame, include_lab_context=args.include_lab_context)
    selected_context, feature_importance = rank_context_features(
        frame,
        context_columns,
        top_n=args.top_context_features,
    )
    target_row = pick_target_sample(frame, sample_time=args.sample_time or None)
    neighborhood = find_context_neighborhood(
        frame,
        target_row,
        selected_context,
        neighbor_count=args.neighbor_count,
    )
    feasible_region = build_feasible_region(
        neighborhood,
        target_row,
        probability_threshold=args.probability_threshold,
        grid_points=args.grid_points,
    )
    result = build_result(
        frame,
        target_row,
        selected_context,
        feature_importance,
        neighborhood,
        feasible_region,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
