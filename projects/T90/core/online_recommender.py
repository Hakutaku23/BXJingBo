from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .casebase import build_casebase_from_windows, load_casebase_dataset, normalize_casebase_frame
from .window_encoder import encode_dcs_window

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


TARGET_LOW = 7.5
TARGET_HIGH = 8.5
DEFAULT_CONTROL_PRIORITY = ("calcium", "bromine")
DEFAULT_EXAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "example_data"


def _resolve_target_range(target_range: Iterable[float] | None) -> tuple[float, float]:
    if target_range is None:
        return TARGET_LOW, TARGET_HIGH
    low, high = target_range
    if low >= high:
        raise ValueError("target_range must satisfy low < high.")
    return float(low), float(high)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _feature_is_selected(name: str, include_columns: tuple[str, ...]) -> bool:
    if not include_columns:
        return True
    root = name.split("__", 1)[0]
    return name in include_columns or root in include_columns


def _choose_context_columns(
    casebase: pd.DataFrame,
    current_context: dict[str, float],
    include_columns: tuple[str, ...],
) -> list[str]:
    blocked = {"sample_time", "t90", "is_in_spec", *DEFAULT_CONTROL_PRIORITY}
    numeric_columns = casebase.select_dtypes(include=["number"]).columns.tolist()
    candidates = [
        column
        for column in numeric_columns
        if column not in blocked and column in current_context and _feature_is_selected(column, include_columns)
    ]
    if not candidates:
        raise ValueError("No overlapping context features were found between the DCS window and the casebase.")
    return candidates


def _rank_context_features(
    casebase: pd.DataFrame,
    context_columns: list[str],
    top_n: int,
) -> tuple[list[str], dict[str, float]]:
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
    model.fit(casebase[context_columns], casebase["is_in_spec"].astype(int))
    importance = pd.Series(
        model.named_steps["rf"].feature_importances_,
        index=context_columns,
    ).sort_values(ascending=False)
    selected = importance.head(min(top_n, len(importance))).index.tolist()
    return selected, importance.head(top_n).round(6).to_dict()


def _find_context_neighborhood(
    casebase: pd.DataFrame,
    current_context: dict[str, float],
    context_columns: list[str],
    neighbor_count: int,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    casebase_matrix = imputer.fit_transform(casebase[context_columns])
    casebase_matrix = scaler.fit_transform(casebase_matrix)
    current_vector = pd.DataFrame([{column: current_context.get(column, np.nan) for column in context_columns}])
    current_matrix = scaler.transform(imputer.transform(current_vector))

    k = max(1, min(neighbor_count, len(casebase)))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(casebase_matrix)
    distances, indices = model.kneighbors(current_matrix)
    neighborhood = casebase.iloc[indices[0]].copy()
    neighborhood["context_distance"] = distances[0]
    return neighborhood.sort_values("context_distance").reset_index(drop=True)


def _fit_local_control_model(
    neighborhood: pd.DataFrame,
    local_neighbor_count: int,
) -> tuple[pd.DataFrame, Pipeline | None]:
    local = neighborhood.head(min(local_neighbor_count, len(neighborhood))).copy()
    local = local.dropna(subset=["calcium", "bromine", "is_in_spec"])
    if local.empty:
        raise ValueError("Local neighborhood does not contain usable calcium/bromine records.")
    if local["is_in_spec"].nunique() < 2:
        return local, None

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(local[["calcium", "bromine"]], local["is_in_spec"].astype(int))
    return local, model


def _summarize_range(series: pd.Series) -> dict[str, float] | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return {
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _build_control_recommendation(
    local: pd.DataFrame,
    model: Pipeline | None,
    *,
    reference_calcium: float | None,
    reference_bromine: float | None,
    probability_threshold: float,
    grid_points: int,
) -> dict[str, object]:
    good = local.loc[local["is_in_spec"] == 1].copy()
    bad = local.loc[local["is_in_spec"] == 0].copy()
    if good.empty:
        raise ValueError("Local neighborhood does not contain in-spec historical samples.")

    empirical = {
        "calcium": _summarize_range(good["calcium"]),
        "bromine": _summarize_range(good["bromine"]),
    }

    calcium_low = float(local["calcium"].quantile(0.05))
    calcium_high = float(local["calcium"].quantile(0.95))
    bromine_low = float(local["bromine"].quantile(0.05))
    bromine_high = float(local["bromine"].quantile(0.95))

    if model is None:
        anchor_source = "reference_values" if reference_calcium is not None and reference_bromine is not None else "empirical_median"
        best_point = {
            "calcium": empirical["calcium"]["median"],
            "bromine": empirical["bromine"]["median"],
            "in_spec_probability": None,
        }
        result = {
            "status": "empirical_only",
            "good_samples": int(len(good)),
            "bad_samples": int(len(bad)),
            "recommended_calcium_range": {
                "min": empirical["calcium"]["p25"],
                "max": empirical["calcium"]["p75"],
            },
            "recommended_bromine_range": {
                "min": empirical["bromine"]["p25"],
                "max": empirical["bromine"]["p75"],
            },
            "empirical_good_ranges": empirical,
            "best_point": best_point,
            "best_point_selection_anchor": {
                "source": anchor_source,
                "reference_calcium": reference_calcium,
                "reference_bromine": reference_bromine,
            },
            "recommended_adjustment": None,
            "calcium_range_given_reference_bromine": None,
            "bromine_range_given_reference_calcium": None,
        }
        if reference_calcium is not None and reference_bromine is not None:
            result["recommended_adjustment"] = {
                "priority": "calcium",
                "calcium_delta": float(best_point["calcium"] - reference_calcium),
                "bromine_delta": float(best_point["bromine"] - reference_bromine),
            }
        return result

    calcium_grid = np.linspace(calcium_low, calcium_high, grid_points)
    bromine_grid = np.linspace(bromine_low, bromine_high, grid_points)
    mesh = pd.DataFrame(
        [(calcium, bromine) for calcium in calcium_grid for bromine in bromine_grid],
        columns=["calcium", "bromine"],
    )
    mesh["in_spec_probability"] = model.predict_proba(mesh)[:, 1]
    feasible = mesh.loc[mesh["in_spec_probability"] >= probability_threshold].copy()
    if feasible.empty:
        feasible = mesh.nlargest(min(25, len(mesh)), "in_spec_probability").copy()

    anchor_calcium = reference_calcium if reference_calcium is not None else empirical["calcium"]["median"]
    anchor_bromine = reference_bromine if reference_bromine is not None else empirical["bromine"]["median"]
    feasible["distance_to_anchor"] = np.sqrt(
        (feasible["calcium"] - float(anchor_calcium)) ** 2
        + (feasible["bromine"] - float(anchor_bromine)) ** 2
    )
    best = feasible.sort_values(
        by=["distance_to_anchor", "in_spec_probability"],
        ascending=[True, False],
    ).iloc[0]

    calcium_scan = None
    if reference_bromine is not None:
        calcium_scan = pd.DataFrame({"calcium": calcium_grid, "bromine": float(reference_bromine)})
        calcium_scan["in_spec_probability"] = model.predict_proba(calcium_scan)[:, 1]
        calcium_scan = calcium_scan.loc[calcium_scan["in_spec_probability"] >= probability_threshold]

    bromine_scan = None
    if reference_calcium is not None:
        bromine_scan = pd.DataFrame({"calcium": float(reference_calcium), "bromine": bromine_grid})
        bromine_scan["in_spec_probability"] = model.predict_proba(bromine_scan)[:, 1]
        bromine_scan = bromine_scan.loc[bromine_scan["in_spec_probability"] >= probability_threshold]

    result = {
        "status": "ok",
        "good_samples": int(len(good)),
        "bad_samples": int(len(bad)),
        "recommended_calcium_range": {
            "min": float(feasible["calcium"].min()),
            "max": float(feasible["calcium"].max()),
        },
        "recommended_bromine_range": {
            "min": float(feasible["bromine"].min()),
            "max": float(feasible["bromine"].max()),
        },
        "empirical_good_ranges": empirical,
        "best_point": {
            "calcium": float(best["calcium"]),
            "bromine": float(best["bromine"]),
            "in_spec_probability": float(best["in_spec_probability"]),
        },
        "best_point_selection_anchor": {
            "source": "reference_values" if reference_calcium is not None and reference_bromine is not None else "empirical_median",
            "reference_calcium": reference_calcium,
            "reference_bromine": reference_bromine,
        },
        "recommended_adjustment": None,
        "calcium_range_given_reference_bromine": None,
        "bromine_range_given_reference_calcium": None,
    }
    if reference_calcium is not None and reference_bromine is not None:
        result["recommended_adjustment"] = {
            "priority": "calcium",
            "calcium_delta": float(best["calcium"] - reference_calcium),
            "bromine_delta": float(best["bromine"] - reference_bromine),
        }
    if calcium_scan is not None and not calcium_scan.empty:
        result["calcium_range_given_reference_bromine"] = {
            "bromine_fixed_at": float(reference_bromine),
            "calcium_min": float(calcium_scan["calcium"].min()),
            "calcium_max": float(calcium_scan["calcium"].max()),
            "max_probability": float(calcium_scan["in_spec_probability"].max()),
        }
    if bromine_scan is not None and not bromine_scan.empty:
        result["bromine_range_given_reference_calcium"] = {
            "calcium_fixed_at": float(reference_calcium),
            "bromine_min": float(bromine_scan["bromine"].min()),
            "bromine_max": float(bromine_scan["bromine"].max()),
            "max_probability": float(bromine_scan["in_spec_probability"].max()),
        }
    return result


def load_example_bundle(data_dir: str | Path | None = None) -> dict[str, object]:
    base_dir = Path(data_dir) if data_dir else DEFAULT_EXAMPLE_DATA_DIR
    dcs = pd.read_parquet(base_dir / "DCS_data_example.parquet").copy()
    ph = pd.read_parquet(base_dir / "PH_data_example.parquet").copy()
    lims = pd.read_parquet(base_dir / "LIMS_data_example.parquet").copy()

    if len(ph) == len(dcs):
        dcs["PH"] = pd.to_numeric(ph.iloc[:, 0], errors="coerce")

    window_size = 15
    lims["sample_time"] = pd.date_range("2025-01-01 00:00:00", periods=len(lims), freq="h")
    lims = normalize_casebase_frame(lims, target_low=TARGET_LOW, target_high=TARGET_HIGH)
    history_count = min(len(lims), max(10, (len(dcs) - window_size) // window_size - 1))
    endpoints = np.linspace(window_size, len(dcs) - window_size - 1, history_count, dtype=int)
    windows = []
    for end in endpoints:
        start = max(0, end - window_size + 1)
        windows.append(dcs.iloc[start : end + 1].reset_index(drop=True))

    outcomes = lims.head(history_count).copy()
    casebase = build_casebase_from_windows(
        windows,
        outcomes,
        target_low=TARGET_LOW,
        target_high=TARGET_HIGH,
    )
    current_window = dcs.iloc[-window_size:].reset_index(drop=True)

    return {
        "casebase": casebase,
        "dcs_window": current_window,
        "runtime_time": "2025-01-03 12:00:00",
    }


def load_example_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
    return load_example_bundle(data_dir=data_dir)["casebase"]


def load_casebase(
    path: str | Path,
    *,
    target_range: Iterable[float] | None = None,
) -> pd.DataFrame:
    low, high = _resolve_target_range(target_range)
    return load_casebase_dataset(path, target_low=low, target_high=high)


def build_t90_recommendation(
    *,
    dcs_window: pd.DataFrame,
    casebase: pd.DataFrame,
    runtime_time: str | None = None,
    reference_calcium: float | None = None,
    reference_bromine: float | None = None,
    target_range: Iterable[float] | None = None,
    top_context_features: int = 12,
    neighbor_count: int = 150,
    local_neighbor_count: int = 80,
    probability_threshold: float = 0.60,
    grid_points: int = 31,
    include_sensors: Iterable[str] = (),
    include_columns: Iterable[str] = (),
    skip_feature_ranking: bool = False,
) -> dict[str, object]:
    low, high = _resolve_target_range(target_range)
    include_columns = tuple(include_columns)
    include_sensors = tuple(include_sensors)

    prepared_casebase = normalize_casebase_frame(casebase, target_low=low, target_high=high)
    current_context = encode_dcs_window(dcs_window, include_sensors=include_sensors)
    context_columns = _choose_context_columns(prepared_casebase, current_context, include_columns)
    if skip_feature_ranking or top_context_features <= 0 or len(context_columns) <= top_context_features:
        selected_context = context_columns
        feature_importance = {}
    else:
        selected_context, feature_importance = _rank_context_features(
            prepared_casebase,
            context_columns,
            top_n=top_context_features,
        )
    neighborhood = _find_context_neighborhood(
        prepared_casebase,
        current_context,
        selected_context,
        neighbor_count,
    )
    local, model = _fit_local_control_model(neighborhood, local_neighbor_count)
    recommendation = _build_control_recommendation(
        local,
        model,
        reference_calcium=_safe_float(reference_calcium),
        reference_bromine=_safe_float(reference_bromine),
        probability_threshold=probability_threshold,
        grid_points=grid_points,
    )

    return {
        "target_range": {"low": low, "high": high},
        "runtime_context": {
            "runtime_time": runtime_time,
            "window_rows": int(len(dcs_window)),
            "reference_calcium": _safe_float(reference_calcium),
            "reference_bromine": _safe_float(reference_bromine),
        },
        "method": {
            "type": "context-conditioned case-based recommendation",
            "goal": "Use the latest DCS window to retrieve similar historical states and recommend target calcium and bromine ranges.",
            "primary_control_priority": list(DEFAULT_CONTROL_PRIORITY),
            "selected_context_features": selected_context,
            "top_context_feature_importance": feature_importance,
            "feature_ranking_skipped": bool(skip_feature_ranking),
        },
        "neighborhood": {
            "context_samples": int(len(neighborhood)),
            "local_model_samples": int(len(local)),
            "local_good_samples": int(local["is_in_spec"].sum()),
            "local_bad_samples": int((1 - local["is_in_spec"]).sum()),
            "median_context_distance": float(neighborhood["context_distance"].median()),
        },
        "recommendation": recommendation,
    }
