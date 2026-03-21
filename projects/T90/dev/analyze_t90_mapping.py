from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "dev" / "artifacts"

LIMS_PATH = DATA_DIR / "t90-溴丁橡胶.xlsx"
DCS_PATH = DATA_DIR / "merge_data.csv"
DCS_OTR_PATH = DATA_DIR / "merge_data_otr.csv"


@dataclass
class AnalysisConfig:
    dcs_path: Path
    lookback_minutes: int = 120
    min_train_samples: int = 80
    target_low: float = 7.5
    target_high: float = 8.5
    top_k_features: int = 15
    recommendation_neighbors: int = 5


def normalize_name(name: object) -> str:
    text = str(name).strip()
    text = text.replace("\xa0", " ")
    return text


def infer_columns(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        key = normalize_name(column)
        if "采样时间" in key:
            mapping["sample_time"] = column
        elif "样品名称" in key:
            mapping["sample_name"] = column
        elif "挥发分" in key and "在线" not in key:
            mapping["volatile_lab"] = column
        elif "在线监测挥发分" in key:
            mapping["volatile_online"] = column
        elif "90" in key.lower():
            mapping["t90"] = column
        elif "溴含量" in key:
            mapping["bromine"] = column
        elif "硬脂酸钙含量" in key:
            mapping["calcium_stearate"] = column
        elif key.startswith("钙含量"):
            mapping["calcium"] = column
        elif "稳定剂" in key:
            mapping["stabilizer"] = column
        elif "门尼粘度" in key:
            mapping["mooney"] = column
        elif "防老剂" in key:
            mapping["antioxidant"] = column
    return mapping


def load_lims_data(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    sheets = pd.read_excel(path, sheet_name=None)
    raw = pd.concat(
        [part.assign(sheet_name=sheet_name) for sheet_name, part in sheets.items()],
        ignore_index=True,
    )
    raw.columns = [normalize_name(col) for col in raw.columns]
    column_map = infer_columns(list(raw.columns))

    rename_map = {source: target for target, source in column_map.items()}
    lims = raw.rename(columns=rename_map)
    lims["sample_time"] = pd.to_datetime(lims["sample_time"], errors="coerce")

    numeric_cols = [
        "volatile_lab",
        "volatile_online",
        "t90",
        "bromine",
        "calcium_stearate",
        "calcium",
        "stabilizer",
        "mooney",
        "antioxidant",
    ]
    for col in numeric_cols:
        if col in lims.columns:
            lims[col] = pd.to_numeric(lims[col], errors="coerce")

    aggregation = {}
    for col in ["sample_name", "sheet_name", *numeric_cols]:
        if col in lims.columns:
            aggregation[col] = "first"

    grouped = (
        lims.sort_values("sample_time")
        .groupby("sample_time", as_index=False)
        .agg(aggregation)
        .sort_values("sample_time")
        .reset_index(drop=True)
    )
    grouped["sheet_name"] = grouped["sheet_name"].fillna("unknown")
    return grouped, column_map


def load_dcs_data(path: Path) -> pd.DataFrame:
    dcs = pd.read_csv(path)
    dcs["time"] = pd.to_datetime(dcs["time"], errors="coerce")
    dcs = dcs.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    numeric_columns = [col for col in dcs.columns if col != "time"]
    for col in numeric_columns:
        dcs[col] = pd.to_numeric(dcs[col], errors="coerce")
    return dcs


def summarize_window(window: pd.DataFrame, sample_time: pd.Timestamp) -> dict[str, float]:
    result: dict[str, float] = {}
    numeric_window = window.select_dtypes(include=["number"])
    if numeric_window.empty:
        return result

    for col in numeric_window.columns:
        series = numeric_window[col].dropna()
        if series.empty:
            continue
        result[f"{col}__last"] = float(series.iloc[-1])
        result[f"{col}__mean"] = float(series.mean())
        result[f"{col}__std"] = float(series.std(ddof=0)) if len(series) > 1 else 0.0
        result[f"{col}__min"] = float(series.min())
        result[f"{col}__max"] = float(series.max())

        if len(series) > 1:
            elapsed = (
                window.loc[series.index, "time"].astype("int64").to_numpy() - sample_time.value
            ) / 60_000_000_000
            if np.nanstd(elapsed) > 0:
                slope = np.polyfit(elapsed, series.to_numpy(dtype=float), 1)[0]
                result[f"{col}__slope"] = float(slope)
    return result


def build_feature_table(lims: pd.DataFrame, dcs: pd.DataFrame, lookback_minutes: int) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    dcs = dcs.sort_values("time").reset_index(drop=True)
    times = dcs["time"].to_numpy()
    lookback = pd.Timedelta(minutes=lookback_minutes)

    for record in lims.itertuples(index=False):
        if pd.isna(record.sample_time):
            continue

        sample_time = pd.Timestamp(record.sample_time)
        start_time = sample_time - lookback
        left = np.searchsorted(times, start_time.to_datetime64(), side="left")
        right = np.searchsorted(times, sample_time.to_datetime64(), side="right")
        window = dcs.iloc[left:right]
        if window.empty:
            continue

        row = {
            "sample_time": sample_time,
            "sheet_name": getattr(record, "sheet_name", "unknown"),
            "t90": getattr(record, "t90", np.nan),
            "t90_in_spec": float(
                getattr(record, "t90", np.nan) >= 0
                and not pd.isna(getattr(record, "t90", np.nan))
                and 7.5 <= getattr(record, "t90", np.nan) <= 8.5
            ),
        }
        for col in [
            "volatile_lab",
            "volatile_online",
            "bromine",
            "calcium_stearate",
            "calcium",
            "stabilizer",
            "mooney",
            "antioxidant",
        ]:
            value = getattr(record, col, np.nan)
            row[col] = value

        row.update(summarize_window(window, sample_time))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {"sample_time", "sheet_name", "t90", "t90_in_spec", "target_class"}
    return [col for col in frame.columns if col not in blocked]


def classify_feature_origin(name: str) -> str:
    lims_features = {
        "volatile_lab",
        "volatile_online",
        "bromine",
        "calcium_stearate",
        "calcium",
        "stabilizer",
        "mooney",
        "antioxidant",
    }
    return "lims_context" if name in lims_features else "dcs_candidate"


def split_feature_importance(importances: pd.Series, top_k: int) -> dict[str, dict[str, float]]:
    dcs = importances[[classify_feature_origin(name) == "dcs_candidate" for name in importances.index]]
    lims = importances[[classify_feature_origin(name) == "lims_context" for name in importances.index]]
    return {
        "overall": importances.head(top_k).round(6).to_dict(),
        "dcs_candidate": dcs.head(top_k).round(6).to_dict(),
        "lims_context": lims.head(top_k).round(6).to_dict(),
    }


def evaluate_regression(feature_table: pd.DataFrame, top_k: int) -> dict[str, object]:
    usable = feature_table.dropna(subset=["t90"]).copy()
    features = select_feature_columns(usable)
    usable = usable.dropna(axis=1, how="all")
    features = [col for col in features if col in usable.columns]
    usable = usable.dropna(subset=features, how="all")

    if len(usable) < 20:
        return {"status": "not_enough_samples", "samples": int(len(usable))}

    X = usable[features]
    y = usable["t90"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=3,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rf = model.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    return {
        "status": "ok",
        "samples": int(len(usable)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "top_features": split_feature_importance(importances, top_k),
    }


def evaluate_classification(feature_table: pd.DataFrame, target_low: float, target_high: float, top_k: int) -> dict[str, object]:
    usable = feature_table.dropna(subset=["t90"]).copy()
    usable["target_class"] = usable["t90"].between(target_low, target_high).astype(int)
    features = select_feature_columns(usable)
    usable = usable.dropna(axis=1, how="all")
    features = [col for col in features if col in usable.columns]
    usable = usable.dropna(subset=features, how="all")

    if usable["target_class"].nunique() < 2 or len(usable) < 20:
        return {"status": "not_enough_samples", "samples": int(len(usable))}

    X = usable[features]
    y = usable["target_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rf = model.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    return {
        "status": "ok",
        "samples": int(len(usable)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "report": classification_report(y_test, predictions, output_dict=True),
        "top_features": split_feature_importance(importances, top_k),
    }


def build_recommendations(
    feature_table: pd.DataFrame,
    target_low: float,
    target_high: float,
    neighbors: int,
    top_k: int,
) -> dict[str, object]:
    usable = feature_table.dropna(subset=["t90"]).copy()
    usable["target_class"] = usable["t90"].between(target_low, target_high).astype(int)
    features = select_feature_columns(usable)
    usable = usable.dropna(axis=1, how="all")
    features = [col for col in features if col in usable.columns]
    usable = usable.dropna(subset=features, how="all")

    good = usable[usable["target_class"] == 1].copy()
    bad = usable[usable["target_class"] == 0].copy()
    if good.empty or bad.empty:
        return {"status": "not_enough_samples", "good_samples": int(len(good)), "bad_samples": int(len(bad))}

    imputer = SimpleImputer(strategy="median")
    imputer.fit(usable[features])
    scaler = StandardScaler()
    scaler.fit(imputer.transform(usable[features]))
    good_matrix = scaler.transform(imputer.transform(good[features]))
    bad_matrix = scaler.transform(imputer.transform(bad[features]))

    model = NearestNeighbors(n_neighbors=min(neighbors, len(good)), metric="euclidean")
    model.fit(good_matrix)
    distances, indices = model.kneighbors(bad_matrix)

    dcs_adjustments = []
    context_shifts = []
    dcs_features = [col for col in features if col.endswith("__last")]
    lims_features = [
        col
        for col in features
        if col in {"volatile_lab", "volatile_online", "bromine", "calcium", "calcium_stearate", "stabilizer", "mooney", "antioxidant"}
    ]
    selected_good = good.iloc[np.unique(indices.flatten())]

    for feature in dcs_features:
        bad_value = bad[feature].median()
        good_value = selected_good[feature].median()
        if pd.isna(bad_value) or pd.isna(good_value):
            continue
        dcs_adjustments.append(
            {
                "feature": feature,
                "good_median": float(good_value),
                "bad_median": float(bad_value),
                "delta_good_minus_bad": float(good_value - bad_value),
            }
        )
    for feature in lims_features:
        bad_value = bad[feature].median()
        good_value = selected_good[feature].median()
        if pd.isna(bad_value) or pd.isna(good_value):
            continue
        context_shifts.append(
            {
                "feature": feature,
                "good_median": float(good_value),
                "bad_median": float(bad_value),
                "delta_good_minus_bad": float(good_value - bad_value),
            }
        )

    dcs_adjustments = sorted(dcs_adjustments, key=lambda item: abs(item["delta_good_minus_bad"]), reverse=True)[:top_k]
    context_shifts = sorted(context_shifts, key=lambda item: abs(item["delta_good_minus_bad"]), reverse=True)[:top_k]

    return {
        "status": "ok",
        "good_samples": int(len(good)),
        "bad_samples": int(len(bad)),
        "median_neighbor_distance": float(np.median(distances)),
        "recommended_dcs_targets": dcs_adjustments,
        "context_shifts": context_shifts,
    }


def build_summary(lims: pd.DataFrame, dcs: pd.DataFrame, feature_table: pd.DataFrame, config: AnalysisConfig, column_map: dict[str, str]) -> dict[str, object]:
    t90_series = lims["t90"].dropna()
    summary = {
        "config": {
            "dcs_path": str(config.dcs_path),
            "lookback_minutes": config.lookback_minutes,
            "target_low": config.target_low,
            "target_high": config.target_high,
        },
        "lims": {
            "raw_samples_after_grouping": int(len(lims)),
            "t90_samples": int(t90_series.notna().sum()),
            "time_min": str(lims["sample_time"].min()),
            "time_max": str(lims["sample_time"].max()),
            "t90_mean": float(t90_series.mean()) if not t90_series.empty else None,
            "t90_std": float(t90_series.std(ddof=0)) if len(t90_series) > 1 else None,
            "inferred_columns": column_map,
        },
        "dcs": {
            "rows": int(len(dcs)),
            "columns": int(len(dcs.columns) - 1),
            "time_min": str(dcs["time"].min()),
            "time_max": str(dcs["time"].max()),
        },
        "joined": {
            "rows": int(len(feature_table)),
            "t90_rows": int(feature_table["t90"].notna().sum()) if "t90" in feature_table else 0,
            "in_spec_rows": int(feature_table["t90_in_spec"].sum()) if "t90_in_spec" in feature_table else 0,
        },
    }
    return summary


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze T90 mapping candidates from LIMS and DCS data.")
    parser.add_argument("--use-otr", action="store_true", help="Use the smaller merge_data_otr.csv instead of merge_data.csv.")
    parser.add_argument("--lookback-minutes", type=int, default=120)
    parser.add_argument("--target-low", type=float, default=7.5)
    parser.add_argument("--target-high", type=float, default=8.5)
    parser.add_argument("--feature-table", type=str, default="", help="Reuse a previously exported feature table CSV.")
    args = parser.parse_args()

    config = AnalysisConfig(
        dcs_path=DCS_OTR_PATH if args.use_otr else DCS_PATH,
        lookback_minutes=args.lookback_minutes,
        target_low=args.target_low,
        target_high=args.target_high,
    )

    ensure_output_dir()
    lims, column_map = load_lims_data(LIMS_PATH)
    reused_feature_table = bool(args.feature_table)
    if reused_feature_table:
        feature_table = pd.read_csv(args.feature_table, parse_dates=["sample_time"])
        dcs = load_dcs_data(config.dcs_path)
    else:
        dcs = load_dcs_data(config.dcs_path)
        feature_table = build_feature_table(lims, dcs, config.lookback_minutes)

    summary = build_summary(lims, dcs, feature_table, config, column_map)
    regression = evaluate_regression(feature_table, config.top_k_features)
    classification = evaluate_classification(
        feature_table,
        config.target_low,
        config.target_high,
        config.top_k_features,
    )
    recommendations = build_recommendations(
        feature_table,
        config.target_low,
        config.target_high,
        config.recommendation_neighbors,
        config.top_k_features,
    )

    result = {
        "summary": summary,
        "regression": regression,
        "classification": classification,
        "recommendations": recommendations,
    }

    if not reused_feature_table:
        feature_table.to_csv(OUTPUT_DIR / "t90_feature_table.csv", index=False)
    with (OUTPUT_DIR / "t90_analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
