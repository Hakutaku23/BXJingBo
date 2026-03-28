from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DISALLOWED_PREFIX_COLUMNS = 7
EPS = 1e-12


@dataclass
class ReservoirSampler1D:
    size: int
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._data = np.empty(self.size, dtype=np.float64)
        self._seen = 0
        self._filled = 0

    def update(self, values: np.ndarray) -> None:
        finite_values = values[np.isfinite(values)]
        for value in finite_values:
            self._seen += 1
            if self._filled < self.size:
                self._data[self._filled] = value
                self._filled += 1
                continue

            replace_idx = self._rng.integers(0, self._seen)
            if replace_idx < self.size:
                self._data[replace_idx] = value

    def values(self) -> np.ndarray:
        return self._data[: self._filled].copy()


class FeatureAudit:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        width = len(feature_names)
        self.total_rows = 0
        self.feature_valid_count = np.zeros(width, dtype=np.int64)
        self.feature_sum = np.zeros(width, dtype=np.float64)
        self.feature_sum_sq = np.zeros(width, dtype=np.float64)
        self.feature_min = np.full(width, np.inf, dtype=np.float64)
        self.feature_max = np.full(width, -np.inf, dtype=np.float64)

        self.pair_count = np.zeros(width, dtype=np.int64)
        self.pair_sum_x = np.zeros(width, dtype=np.float64)
        self.pair_sum_x2 = np.zeros(width, dtype=np.float64)
        self.pair_sum_y = np.zeros(width, dtype=np.float64)
        self.pair_sum_y2 = np.zeros(width, dtype=np.float64)
        self.pair_sum_xy = np.zeros(width, dtype=np.float64)

        self.target_valid_count = 0
        self.target_sum = 0.0
        self.target_sum_sq = 0.0
        self.target_min = np.inf
        self.target_max = -np.inf

        self.time_min: str | None = None
        self.time_max: str | None = None

    def update(self, time_values: pd.Series, features: pd.DataFrame, target: pd.Series) -> None:
        self.total_rows += len(features)

        chunk_time_min = time_values.min()
        chunk_time_max = time_values.max()
        if isinstance(chunk_time_min, str):
            self.time_min = chunk_time_min if self.time_min is None else min(self.time_min, chunk_time_min)
        if isinstance(chunk_time_max, str):
            self.time_max = chunk_time_max if self.time_max is None else max(self.time_max, chunk_time_max)

        x = features.to_numpy(dtype=np.float64, copy=False)
        y = target.to_numpy(dtype=np.float64, copy=False)

        valid_x = np.isfinite(x)
        valid_y = np.isfinite(y)

        self.feature_valid_count += valid_x.sum(axis=0, dtype=np.int64)

        x_zeroed = np.where(valid_x, x, 0.0)
        self.feature_sum += x_zeroed.sum(axis=0, dtype=np.float64)
        self.feature_sum_sq += np.square(x_zeroed).sum(axis=0, dtype=np.float64)

        if valid_x.any():
            chunk_min = np.where(valid_x, x, np.inf).min(axis=0)
            chunk_max = np.where(valid_x, x, -np.inf).max(axis=0)
            self.feature_min = np.minimum(self.feature_min, chunk_min)
            self.feature_max = np.maximum(self.feature_max, chunk_max)

        valid_y_count = int(valid_y.sum())
        self.target_valid_count += valid_y_count
        if valid_y_count:
            y_valid = y[valid_y]
            self.target_sum += float(y_valid.sum(dtype=np.float64))
            self.target_sum_sq += float(np.square(y_valid).sum(dtype=np.float64))
            self.target_min = min(self.target_min, float(y_valid.min()))
            self.target_max = max(self.target_max, float(y_valid.max()))

        if valid_y.any():
            pair_mask = valid_x & valid_y[:, None]
            self.pair_count += pair_mask.sum(axis=0, dtype=np.int64)
            self.pair_sum_x += np.where(pair_mask, x, 0.0).sum(axis=0, dtype=np.float64)
            self.pair_sum_x2 += np.where(pair_mask, np.square(x), 0.0).sum(axis=0, dtype=np.float64)
            y_col = y[:, None]
            self.pair_sum_y += np.where(pair_mask, y_col, 0.0).sum(axis=0, dtype=np.float64)
            self.pair_sum_y2 += np.where(pair_mask, np.square(y_col), 0.0).sum(axis=0, dtype=np.float64)
            self.pair_sum_xy += np.where(pair_mask, x * y_col, 0.0).sum(axis=0, dtype=np.float64)

    def to_frame(self) -> pd.DataFrame:
        valid_count = self.feature_valid_count.astype(np.float64)
        mean = np.divide(
            self.feature_sum,
            valid_count,
            out=np.full_like(self.feature_sum, np.nan),
            where=valid_count > 0,
        )
        variance = np.divide(
            self.feature_sum_sq,
            valid_count,
            out=np.full_like(self.feature_sum_sq, np.nan),
            where=valid_count > 0,
        ) - np.square(mean)
        variance = np.where(variance < 0.0, 0.0, variance)
        std = np.sqrt(variance)

        n = self.pair_count.astype(np.float64)
        numerator = n * self.pair_sum_xy - self.pair_sum_x * self.pair_sum_y
        denom_x = n * self.pair_sum_x2 - np.square(self.pair_sum_x)
        denom_y = n * self.pair_sum_y2 - np.square(self.pair_sum_y)
        denominator = np.sqrt(np.maximum(denom_x, 0.0) * np.maximum(denom_y, 0.0))
        target_corr = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > EPS,
        )

        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "valid_count": self.feature_valid_count,
                "missing_count": self.total_rows - self.feature_valid_count,
                "missing_rate": 1.0 - (valid_count / max(self.total_rows, 1)),
                "mean": mean,
                "std": std,
                "min": np.where(np.isfinite(self.feature_min), self.feature_min, np.nan),
                "max": np.where(np.isfinite(self.feature_max), self.feature_max, np.nan),
                "pair_count": self.pair_count,
                "target_corr": target_corr,
                "abs_target_corr": np.abs(target_corr),
            }
        )
        df["is_constant"] = df["std"].fillna(0.0) <= EPS
        return df.sort_values(["abs_target_corr", "valid_count"], ascending=[False, False]).reset_index(drop=True)

    def target_summary(self, sample_values: np.ndarray) -> dict[str, float | int | str | None]:
        target_mean = self.target_sum / self.target_valid_count if self.target_valid_count else math.nan
        target_variance = (
            self.target_sum_sq / self.target_valid_count - target_mean**2
            if self.target_valid_count
            else math.nan
        )
        target_std = math.sqrt(max(target_variance, 0.0)) if self.target_valid_count else math.nan
        quantiles = {}
        if sample_values.size:
            q_values = np.quantile(sample_values, [0.01, 0.05, 0.5, 0.95, 0.99])
            quantiles = {
                "q01": float(q_values[0]),
                "q05": float(q_values[1]),
                "q50": float(q_values[2]),
                "q95": float(q_values[3]),
                "q99": float(q_values[4]),
            }
        return {
            "row_count": self.total_rows,
            "time_min": self.time_min,
            "time_max": self.time_max,
            "target_valid_count": self.target_valid_count,
            "target_missing_count": self.total_rows - self.target_valid_count,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "target_min": None if not math.isfinite(self.target_min) else float(self.target_min),
            "target_max": None if not math.isfinite(self.target_max) else float(self.target_max),
            **quantiles,
        }


def load_schema(csv_path: Path, excluded_features: set[str] | None = None) -> tuple[str, list[str], str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if len(header) < 9:
        raise ValueError("The CSV does not contain the expected feature layout.")
    time_column = header[0]
    feature_columns = header[DISALLOWED_PREFIX_COLUMNS:-1]
    if excluded_features:
        feature_columns = [name for name in feature_columns if name not in excluded_features]
    target_column = header[-1]
    return time_column, feature_columns, target_column


def first_pass(
    csv_path: Path,
    time_column: str,
    feature_columns: list[str],
    target_column: str,
    chunk_size: int,
    target_sample_size: int,
    seed: int,
) -> tuple[FeatureAudit, dict[str, float | int | str | None], pd.DataFrame]:
    audit = FeatureAudit(feature_columns)
    target_sampler = ReservoirSampler1D(size=target_sample_size, seed=seed)

    usecols = [time_column, *feature_columns, target_column]
    chunk_iter = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False)
    for chunk in chunk_iter:
        target_sampler.update(chunk[target_column].to_numpy(dtype=np.float64, copy=False))
        audit.update(
            time_values=chunk[time_column],
            features=chunk[feature_columns],
            target=chunk[target_column],
        )

    summary = audit.target_summary(target_sampler.values())
    screening = audit.to_frame()
    return audit, summary, screening


def second_pass_sample(
    csv_path: Path,
    selected_features: list[str],
    target_column: str,
    chunk_size: int,
    total_rows: int,
    sample_rows: int,
    seed: int,
) -> pd.DataFrame:
    if not selected_features:
        return pd.DataFrame(columns=[target_column])

    sample_fraction = min(1.0, sample_rows / max(total_rows, 1))
    usecols = [*selected_features, target_column]
    samples: list[pd.DataFrame] = []

    chunk_iter = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False)
    for chunk_idx, chunk in enumerate(chunk_iter):
        sampled = chunk.sample(frac=sample_fraction, random_state=seed + chunk_idx)
        if not sampled.empty:
            samples.append(sampled)

    if not samples:
        return pd.DataFrame(columns=usecols)

    sample_df = pd.concat(samples, ignore_index=True)
    if len(sample_df) > sample_rows:
        sample_df = sample_df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)
    return sample_df


def greedy_correlation_prune(
    ranking: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    max_feature_corr: float,
    max_selected: int,
) -> list[str]:
    kept: list[str] = []
    for feature in ranking["feature"]:
        if len(kept) >= max_selected:
            break
        if not kept:
            kept.append(feature)
            continue
        corr_with_kept = corr_matrix.loc[feature, kept].fillna(0.0).abs()
        if bool((corr_with_kept <= max_feature_corr).all()):
            kept.append(feature)
    return kept


def build_recommendations(
    screening: pd.DataFrame,
    max_missing_rate: float,
    min_pair_count_ratio: float,
    top_k: int,
    max_feature_corr: float,
    max_selected: int,
    sample_corr: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    min_pair_count = int(screening["valid_count"].max() * min_pair_count_ratio)
    screened = screening.copy()
    screened["is_recommended"] = False
    screened["passes_basic_filter"] = (
        (screened["missing_rate"] <= max_missing_rate)
        & (screened["pair_count"] >= min_pair_count)
        & (~screened["is_constant"])
        & screened["target_corr"].notna()
    )

    ranked = screened.loc[screened["passes_basic_filter"]].copy()
    ranked = ranked.sort_values(["abs_target_corr", "pair_count"], ascending=[False, False]).head(top_k)
    if ranked.empty:
        return screened, []

    recommended = greedy_correlation_prune(
        ranking=ranked,
        corr_matrix=sample_corr.reindex(index=ranked["feature"], columns=ranked["feature"]),
        max_feature_corr=max_feature_corr,
        max_selected=max_selected,
    )
    screened["is_recommended"] = screened["feature"].isin(recommended)
    return screened, recommended


def write_markdown_report(
    out_path: Path,
    summary: dict[str, float | int | str | None],
    screening: pd.DataFrame,
    recommended_features: list[str],
    max_missing_rate: float,
    max_feature_corr: float,
) -> None:
    top10 = screening.head(10)
    lines = [
        "# Volatiles output.csv audit",
        "",
        "## Dataset summary",
        f"- rows: {summary['row_count']}",
        f"- time range: {summary['time_min']} -> {summary['time_max']}",
        f"- target valid rows: {summary['target_valid_count']}",
        f"- target mean/std: {summary['target_mean']:.6f} / {summary['target_std']:.6f}",
        f"- target min/max: {summary['target_min']} / {summary['target_max']}",
        "",
        "## Screening rules",
        f"- ignore columns 1-7 and treat column 413 as the target",
        f"- basic filter: missing_rate <= {max_missing_rate:.2f}, non-constant, valid pairs retained",
        f"- ranking metric: absolute Pearson correlation against target",
        f"- redundancy pruning: sample feature-feature correlation <= {max_feature_corr:.2f}",
        "",
        "## Top single-feature signals",
    ]
    for row in top10.itertuples(index=False):
        lines.append(
            f"- {row.feature}: abs_corr={row.abs_target_corr:.4f}, "
            f"missing_rate={row.missing_rate:.4f}, std={row.std:.4f}"
        )
    lines.extend(
        [
            "",
            "## Recommended feature set",
            f"- count: {len(recommended_features)}",
        ]
    )
    for feature in recommended_features:
        lines.append(f"- {feature}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and screen Volatiles output.csv")
    parser.add_argument("--csv", type=Path, default=Path("data/output.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("dev/artifacts/data_audit"))
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--target-sample-size", type=int, default=50_000)
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--max-missing-rate", type=float, default=0.40)
    parser.add_argument("--min-pair-count-ratio", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--max-feature-corr", type=float, default=0.98)
    parser.add_argument("--max-selected", type=int, default=40)
    parser.add_argument("--exclude-features", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    excluded_features = set(args.exclude_features)
    time_column, feature_columns, target_column = load_schema(csv_path, excluded_features=excluded_features)
    _, summary, screening = first_pass(
        csv_path=csv_path,
        time_column=time_column,
        feature_columns=feature_columns,
        target_column=target_column,
        chunk_size=args.chunk_size,
        target_sample_size=args.target_sample_size,
        seed=args.seed,
    )

    preselected = screening.head(args.top_k)["feature"].tolist()
    sample_df = second_pass_sample(
        csv_path=csv_path,
        selected_features=preselected,
        target_column=target_column,
        chunk_size=args.chunk_size,
        total_rows=int(summary["row_count"]),
        sample_rows=args.sample_rows,
        seed=args.seed,
    )
    sample_corr = sample_df[preselected].corr().abs() if not sample_df.empty else pd.DataFrame()

    screening, recommended_features = build_recommendations(
        screening=screening,
        max_missing_rate=args.max_missing_rate,
        min_pair_count_ratio=args.min_pair_count_ratio,
        top_k=args.top_k,
        max_feature_corr=args.max_feature_corr,
        max_selected=args.max_selected,
        sample_corr=sample_corr,
    )

    screening.to_csv(outdir / "feature_screening.csv", index=False, encoding="utf-8-sig")
    screening.loc[screening["is_recommended"]].to_csv(
        outdir / "recommended_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not sample_corr.empty:
        sample_corr.to_csv(outdir / "sample_feature_correlation.csv", encoding="utf-8-sig")

    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "time_column": time_column,
                "disallowed_columns": DISALLOWED_PREFIX_COLUMNS,
                "feature_count": len(feature_columns),
                "excluded_features": sorted(excluded_features),
                "target_column": target_column,
                "recommended_feature_count": len(recommended_features),
                "recommended_features": recommended_features,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(
        out_path=outdir / "report.md",
        summary=summary,
        screening=screening,
        recommended_features=recommended_features,
        max_missing_rate=args.max_missing_rate,
        max_feature_corr=args.max_feature_corr,
    )

    print(f"Audit finished. Outputs written to: {outdir}")
    print(f"Recommended feature count: {len(recommended_features)}")


if __name__ == "__main__":
    main()
