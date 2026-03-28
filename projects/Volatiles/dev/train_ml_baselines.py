from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_linear_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_tree_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    candidates: list[tuple[str, Any]]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="dummy_mean",
            candidates=[
                (
                    "mean",
                    TransformedTargetRegressor(regressor=DummyRegressor(strategy="mean")),
                )
            ],
        ),
        ModelSpec(
            name="ridge",
            candidates=[
                (
                    f"alpha_{alpha:g}",
                    TransformedTargetRegressor(
                        regressor=build_linear_pipeline(Ridge(alpha=alpha, random_state=None))
                    ),
                )
                for alpha in [0.1, 1.0, 10.0]
            ],
        ),
        ModelSpec(
            name="elastic_net",
            candidates=[
                (
                    f"alpha_{alpha:g}_l1_{l1_ratio:g}",
                    TransformedTargetRegressor(
                        regressor=build_linear_pipeline(
                            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000, random_state=42)
                        )
                    ),
                )
                for alpha in [1e-4, 1e-3, 1e-2]
                for l1_ratio in [0.1, 0.5, 0.9]
            ],
        ),
        ModelSpec(
            name="pls",
            candidates=[
                (
                    f"components_{n_components}",
                    TransformedTargetRegressor(
                        regressor=build_linear_pipeline(PLSRegression(n_components=n_components, scale=False))
                    ),
                )
                for n_components in [2, 4, 8, 12]
            ],
        ),
        ModelSpec(
            name="extra_trees",
            candidates=[
                (
                    f"depth_{max_depth}_leaf_{min_samples_leaf}_estimators_{n_estimators}",
                    TransformedTargetRegressor(
                        regressor=build_tree_pipeline(
                            ExtraTreesRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                n_jobs=1,
                                random_state=42,
                            )
                        )
                    ),
                )
                for (n_estimators, max_depth, min_samples_leaf) in [(250, None, 2)]
            ],
        ),
        ModelSpec(
            name="hist_gbr",
            candidates=[
                (
                    f"lr_{learning_rate:g}_depth_{max_depth}_leaf_{min_samples_leaf}",
                    TransformedTargetRegressor(
                        regressor=build_tree_pipeline(
                            HistGradientBoostingRegressor(
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_leaf_nodes=leaf_nodes,
                                min_samples_leaf=min_samples_leaf,
                                l2_regularization=l2_regularization,
                                early_stopping=False,
                                random_state=42,
                            )
                        )
                    ),
                )
                for (learning_rate, max_depth, leaf_nodes, min_samples_leaf, l2_regularization) in [
                    (0.05, 6, 31, 40, 0.0),
                    (0.03, 8, 63, 80, 0.1),
                ]
            ],
        ),
    ]


def load_feature_list(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"Missing 'feature' column in {path}")
    return frame["feature"].dropna().astype(str).tolist()


def load_dataset(csv_path: Path, features: list[str], target_column: str, time_column: str) -> pd.DataFrame:
    usecols = [time_column, *features, target_column]
    frame = pd.read_csv(csv_path, usecols=usecols, parse_dates=[time_column], low_memory=False)
    frame = frame.sort_values(time_column).reset_index(drop=True)
    return frame


def split_by_time(
    frame: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, pd.DataFrame]:
    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))
    return {
        "train": frame.iloc[:train_end].copy(),
        "valid": frame.iloc[train_end:valid_end].copy(),
        "test": frame.iloc[valid_end:].copy(),
    }


def flatten_prediction(pred: Any) -> np.ndarray:
    array = np.asarray(pred)
    return array.reshape(-1)


def tail_frame(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.iloc[-max_rows:].copy()


def run_model_search(
    specs: list[ModelSpec],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in specs:
        for candidate_name, estimator in spec.candidates:
            print(f"Training {spec.name} [{candidate_name}]")
            model = clone(estimator)
            model.fit(x_train, y_train)
            valid_pred = flatten_prediction(model.predict(x_valid))
            valid_metrics = evaluate_regression(y_valid, valid_pred)
            records.append(
                {
                    "model_name": spec.name,
                    "candidate_name": candidate_name,
                    **{f"valid_{key}": value for key, value in valid_metrics.items()},
                }
            )
    result = pd.DataFrame(records)
    return result.sort_values(["valid_rmse", "valid_mae", "model_name"]).reset_index(drop=True)


def refit_and_test(
    best_row: pd.Series,
    specs: list[ModelSpec],
    x_train_valid: pd.DataFrame,
    y_train_valid: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    lookup = {
        (spec.name, candidate_name): estimator
        for spec in specs
        for candidate_name, estimator in spec.candidates
    }
    estimator = clone(lookup[(best_row["model_name"], best_row["candidate_name"])])
    estimator.fit(x_train_valid, y_train_valid)
    test_pred = flatten_prediction(estimator.predict(x_test))
    metrics = evaluate_regression(y_test, test_pred)
    pred_frame = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": test_pred,
            "residual": y_test - test_pred,
        }
    )
    return estimator, metrics, pred_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML baselines for Volatiles output.csv")
    parser.add_argument("--csv", type=Path, default=Path("data/output.csv"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("dev/artifacts/data_audit/recommended_features.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("dev/artifacts/ml_baselines"))
    parser.add_argument("--time-column", type=str, default="time")
    parser.add_argument("--target-column", type=str, default="Y_cal")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--search-train-max-rows", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    features = load_feature_list(args.features.resolve())
    frame = load_dataset(
        csv_path=args.csv.resolve(),
        features=features,
        target_column=args.target_column,
        time_column=args.time_column,
    )
    splits = split_by_time(frame, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    x_train = splits["train"][features]
    y_train = splits["train"][args.target_column].to_numpy(dtype=np.float64)
    x_valid = splits["valid"][features]
    y_valid = splits["valid"][args.target_column].to_numpy(dtype=np.float64)
    x_test = splits["test"][features]
    y_test = splits["test"][args.target_column].to_numpy(dtype=np.float64)
    x_train_valid = pd.concat([x_train, x_valid], axis=0, ignore_index=True)
    y_train_valid = np.concatenate([y_train, y_valid], axis=0)

    search_train_split = tail_frame(splits["train"], args.search_train_max_rows)
    x_train_search = search_train_split[features]
    y_train_search = search_train_split[args.target_column].to_numpy(dtype=np.float64)

    specs = model_specs()
    leaderboard = run_model_search(specs, x_train_search, y_train_search, x_valid, y_valid)
    best_row = leaderboard.iloc[0]
    best_model, test_metrics, test_predictions = refit_and_test(
        best_row=best_row,
        specs=specs,
        x_train_valid=x_train_valid,
        y_train_valid=y_train_valid,
        x_test=x_test,
        y_test=y_test,
    )

    test_predictions.insert(0, args.time_column, splits["test"][args.time_column].reset_index(drop=True))
    leaderboard.to_csv(outdir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(outdir / "best_model_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_model, outdir / "best_model.joblib")

    summary = {
        "feature_file": str(args.features.resolve()),
        "feature_count": len(features),
        "best_model_name": str(best_row["model_name"]),
        "best_candidate_name": str(best_row["candidate_name"]),
        "valid_metrics": {
            "mae": float(best_row["valid_mae"]),
            "rmse": float(best_row["valid_rmse"]),
            "r2": float(best_row["valid_r2"]),
        },
        "test_metrics": test_metrics,
        "split_summary": {
            name: {
                "rows": int(len(split)),
                "time_min": str(split[args.time_column].iloc[0]),
                "time_max": str(split[args.time_column].iloc[-1]),
            }
            for name, split in splits.items()
        },
        "search_train_rows": int(len(search_train_split)),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Volatiles ML baselines",
        "",
        "## Data split",
        f"- feature count: {len(features)}",
        f"- search train rows: {summary['search_train_rows']}",
    ]
    for name, split_summary in summary["split_summary"].items():
        report_lines.append(
            f"- {name}: {split_summary['rows']} rows, {split_summary['time_min']} -> {split_summary['time_max']}"
        )
    report_lines.extend(
        [
            "",
            "## Best validation model",
            f"- model: {summary['best_model_name']}",
            f"- candidate: {summary['best_candidate_name']}",
            f"- valid MAE/RMSE/R2: "
            f"{summary['valid_metrics']['mae']:.6f} / {summary['valid_metrics']['rmse']:.6f} / "
            f"{summary['valid_metrics']['r2']:.6f}",
            "",
            "## Test metrics",
            f"- test MAE/RMSE/R2: "
            f"{summary['test_metrics']['mae']:.6f} / {summary['test_metrics']['rmse']:.6f} / "
            f"{summary['test_metrics']['r2']:.6f}",
            "",
            "## Top leaderboard rows",
        ]
    )
    for row in leaderboard.head(10).itertuples(index=False):
        report_lines.append(
            f"- {row.model_name} [{row.candidate_name}]: "
            f"valid_rmse={row.valid_rmse:.6f}, valid_mae={row.valid_mae:.6f}, valid_r2={row.valid_r2:.6f}"
        )
    (outdir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Finished baseline search. Best model: {summary['best_model_name']} [{summary['best_candidate_name']}]")
    print(
        "Validation RMSE {:.6f}, Test RMSE {:.6f}".format(
            summary["valid_metrics"]["rmse"],
            summary["test_metrics"]["rmse"],
        )
    )


if __name__ == "__main__":
    main()
