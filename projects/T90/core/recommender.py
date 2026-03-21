from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_LOW = 7.5
TARGET_HIGH = 8.5
DEFAULT_CONTROL_PRIORITY = ("calcium", "bromine")
DEFAULT_CONTROL_COLUMNS = list(DEFAULT_CONTROL_PRIORITY)
DEFAULT_EXAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "example_data"


def load_example_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
    base_dir = Path(data_dir) if data_dir else DEFAULT_EXAMPLE_DATA_DIR
    lims_path = base_dir / "LIMS_data_example.parquet"
    frame = pd.read_parquet(lims_path).copy()
    frame = frame.rename(
        columns={
            "sampling_time": "sample_time",
            "time": "sample_time",
        }
    )
    if "sample_time" not in frame.columns:
        raise ValueError("Example dataset must contain `sample_time` or `time` column.")
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    return frame.sort_values("sample_time").reset_index(drop=True)


def _resolve_target_range(target_range: Iterable[float] | None) -> tuple[float, float]:
    if target_range is None:
        return TARGET_LOW, TARGET_HIGH
    low, high = target_range
    if low >= high:
        raise ValueError("target_range must satisfy low < high.")
    return float(low), float(high)


def _detect_control_columns(frame: pd.DataFrame) -> list[str]:
    aliases = {
        "calcium": ["calcium", "钙含量", "ca"],
        "bromine": ["bromine", "溴含量", "br"],
    }
    resolved: list[str] = []
    rename_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        hit = next((col for col in frame.columns if col in candidates), None)
        if hit is None:
            raise ValueError(f"Missing required control column for `{canonical}`.")
        rename_map[hit] = canonical
        resolved.append(canonical)
    if rename_map:
        frame.rename(columns=rename_map, inplace=True)
    return resolved


def _detect_target_column(frame: pd.DataFrame) -> str:
    for candidate in ("t90", "T90"):
        if candidate in frame.columns:
            if candidate != "t90":
                frame.rename(columns={candidate: "t90"}, inplace=True)
            return "t90"
    raise ValueError("Input data must contain `t90` column.")


def _choose_context_columns(
    frame: pd.DataFrame,
    control_columns: list[str],
    include_columns: tuple[str, ...],
) -> list[str]:
    blocked = {"sample_time", "t90", "is_in_spec", *control_columns}
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    candidates = [col for col in numeric_columns if col not in blocked]
    if include_columns:
        candidates = [col for col in candidates if col in include_columns]
    if not candidates:
        raise ValueError("No usable numeric context features were found.")
    return candidates


def _rank_context_features(
    frame: pd.DataFrame,
    context_columns: list[str],
    top_n: int,
) -> tuple[list[str], dict[str, float]]:
    usable = frame.dropna(subset=["t90", "calcium", "bromine"]).copy()
    usable = usable.dropna(axis=1, how="all")
    columns = [col for col in context_columns if col in usable.columns]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(usable[columns], usable["is_in_spec"].astype(int))
    importance = pd.Series(
        model.named_steps["rf"].feature_importances_,
        index=columns,
    ).sort_values(ascending=False)
    selected = importance.head(min(top_n, len(importance))).index.tolist()
    return selected, importance.head(top_n).round(6).to_dict()


def _pick_target_row(frame: pd.DataFrame, sample_time: str | None) -> pd.Series:
    usable = frame.dropna(subset=["t90", "calcium", "bromine"]).copy()
    if sample_time:
        matched = usable.loc[usable["sample_time"] == pd.to_datetime(sample_time)]
        if matched.empty:
            raise ValueError(f"Sample time not found: {sample_time}")
        return matched.sort_values("sample_time").iloc[-1]
    bad = usable.loc[usable["is_in_spec"] == 0].sort_values("sample_time")
    if bad.empty:
        raise ValueError("No out-of-spec sample was found in the dataset.")
    return bad.iloc[-1]


def _find_context_neighborhood(
    frame: pd.DataFrame,
    target_row: pd.Series,
    context_columns: list[str],
    neighbor_count: int,
) -> pd.DataFrame:
    usable = frame.dropna(subset=["t90", "calcium", "bromine"]).copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    context_matrix = scaler.fit_transform(imputer.fit_transform(usable[context_columns]))
    target_vector = scaler.transform(
        imputer.transform(pd.DataFrame([target_row[context_columns]], columns=context_columns))
    )
    k = max(1, min(neighbor_count, len(usable)))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(context_matrix)
    distances, indices = model.kneighbors(target_vector)
    neighborhood = usable.iloc[indices[0]].copy()
    neighborhood["context_distance"] = distances[0]
    return neighborhood.sort_values("context_distance").reset_index(drop=True)


def _fit_local_control_model(
    neighborhood: pd.DataFrame,
    local_neighbor_count: int,
) -> tuple[pd.DataFrame, Pipeline]:
    local = neighborhood.head(min(local_neighbor_count, len(neighborhood))).copy()
    local = local.dropna(subset=["calcium", "bromine", "is_in_spec"])
    if local["is_in_spec"].nunique() < 2:
        raise ValueError("Local neighborhood does not contain both in-spec and out-of-spec samples.")
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(local[["calcium", "bromine"]], local["is_in_spec"].astype(int))
    return local, model


def _build_control_grid(
    local: pd.DataFrame,
    target_row: pd.Series,
    model: Pipeline,
    probability_threshold: float,
    grid_points: int,
) -> dict[str, object]:
    calcium_grid = np.linspace(local["calcium"].quantile(0.05), local["calcium"].quantile(0.95), grid_points)
    bromine_grid = np.linspace(local["bromine"].quantile(0.05), local["bromine"].quantile(0.95), grid_points)

    mesh = pd.DataFrame(
        [(ca, br) for ca in calcium_grid for br in bromine_grid],
        columns=["calcium", "bromine"],
    )
    mesh["in_spec_probability"] = model.predict_proba(mesh)[:, 1]
    feasible = mesh.loc[mesh["in_spec_probability"] >= probability_threshold].copy()
    if feasible.empty:
        feasible = mesh.nlargest(min(20, len(mesh)), "in_spec_probability").copy()
    feasible["distance_from_current"] = np.sqrt(
        ((feasible["calcium"] - float(target_row["calcium"])) ** 2)
        + ((feasible["bromine"] - float(target_row["bromine"])) ** 2)
    )
    best = feasible.sort_values(
        by=["distance_from_current", "in_spec_probability"],
        ascending=[True, False],
    ).iloc[0]

    calcium_scan = pd.DataFrame({"calcium": calcium_grid, "bromine": float(target_row["bromine"])})
    calcium_scan["in_spec_probability"] = model.predict_proba(calcium_scan)[:, 1]
    bromine_scan = pd.DataFrame({"calcium": float(target_row["calcium"]), "bromine": bromine_grid})
    bromine_scan["in_spec_probability"] = model.predict_proba(bromine_scan)[:, 1]

    calcium_feasible = calcium_scan.loc[calcium_scan["in_spec_probability"] >= probability_threshold]
    bromine_feasible = bromine_scan.loc[bromine_scan["in_spec_probability"] >= probability_threshold]

    return {
        "status": "ok",
        "feasible_points": int(len(feasible)),
        "best_point": {
            "calcium": float(best["calcium"]),
            "bromine": float(best["bromine"]),
            "in_spec_probability": float(best["in_spec_probability"]),
        },
        "recommended_adjustment": {
            "priority": "calcium",
            "calcium_delta": float(best["calcium"] - float(target_row["calcium"])),
            "bromine_delta": float(best["bromine"] - float(target_row["bromine"])),
        },
        "calcium_range_given_current_bromine": None
        if calcium_feasible.empty
        else {
            "bromine_fixed_at": float(target_row["bromine"]),
            "calcium_min": float(calcium_feasible["calcium"].min()),
            "calcium_max": float(calcium_feasible["calcium"].max()),
            "max_probability": float(calcium_feasible["in_spec_probability"].max()),
        },
        "bromine_range_given_current_calcium": None
        if bromine_feasible.empty
        else {
            "calcium_fixed_at": float(target_row["calcium"]),
            "bromine_min": float(bromine_feasible["bromine"].min()),
            "bromine_max": float(bromine_feasible["bromine"].max()),
            "max_probability": float(bromine_feasible["in_spec_probability"].max()),
        },
    }


def build_t90_recommendation(
    data: pd.DataFrame,
    sample_time: str | None = None,
    target_range: Iterable[float] | None = None,
    *,
    top_context_features: int = 12,
    neighbor_count: int = 120,
    local_neighbor_count: int = 80,
    probability_threshold: float = 0.60,
    grid_points: int = 31,
    include_columns: Iterable[str] = (),
) -> dict[str, object]:
    frame = data.copy()
    low, high = _resolve_target_range(target_range)
    _detect_target_column(frame)
    control_columns = _detect_control_columns(frame)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], errors="coerce")
    frame["t90"] = pd.to_numeric(frame["t90"], errors="coerce")
    for column in control_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_in_spec"] = frame["t90"].between(low, high).astype(int)

    context_columns = _choose_context_columns(frame, control_columns, tuple(include_columns))
    selected_context, feature_importance = _rank_context_features(
        frame,
        context_columns,
        top_n=top_context_features,
    )
    target_row = _pick_target_row(frame, sample_time=sample_time)
    neighborhood = _find_context_neighborhood(frame, target_row, selected_context, neighbor_count)
    local, model = _fit_local_control_model(neighborhood, local_neighbor_count)
    control_recommendation = _build_control_grid(
        local,
        target_row,
        model,
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )

    return {
        "target_range": {"low": low, "high": high},
        "target_sample": {
            "sample_time": str(target_row["sample_time"]),
            "t90": float(target_row["t90"]),
            "current_calcium": float(target_row["calcium"]),
            "current_bromine": float(target_row["bromine"]),
            "is_in_spec": bool(target_row["is_in_spec"]),
        },
        "method": {
            "type": "context-conditioned case-based recommendation",
            "goal": "Use current process context to recommend calcium-first adjustments and bromine backup ranges.",
            "primary_control_priority": list(DEFAULT_CONTROL_PRIORITY),
            "selected_context_features": selected_context,
            "top_context_feature_importance": feature_importance,
        },
        "neighborhood": {
            "context_samples": int(len(neighborhood)),
            "local_model_samples": int(len(local)),
            "local_good_samples": int(local["is_in_spec"].sum()),
            "local_bad_samples": int((1 - local["is_in_spec"]).sum()),
            "median_context_distance": float(neighborhood["context_distance"].median()),
        },
        "recommendation": control_recommendation,
    }
