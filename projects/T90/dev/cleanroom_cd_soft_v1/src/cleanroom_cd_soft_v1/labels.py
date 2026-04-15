from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def centered_desirability_point(values: pd.Series, center: float, tolerance: float) -> pd.Series:
    return (1.0 - (values.astype(float) - center).abs() / tolerance).clip(lower=0.0, upper=1.0)


def centered_desirability_uncertain(
    values: pd.Series,
    center: float,
    tolerance: float,
    error_half_width: float,
    integration_points: int,
) -> pd.Series:
    deltas = np.linspace(-float(error_half_width), float(error_half_width), int(integration_points))
    matrix = values.to_numpy(dtype=float)[:, None] + deltas[None, :]
    scores = np.maximum(0.0, 1.0 - np.abs(matrix - center) / tolerance)
    return pd.Series(scores.mean(axis=1), index=values.index, dtype=float)


def generalized_bell_desirability(values: pd.Series | np.ndarray, center: float, width: float, p: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + (np.abs(arr - center) / float(width)) ** float(p))


def generalized_bell_uncertain_bounded_uniform(
    values: pd.Series,
    center: float,
    width: float,
    p: float,
    error_half_width: float,
    integration_points: int,
) -> tuple[pd.Series, pd.Series]:
    if float(error_half_width) <= 0.0:
        scores = generalized_bell_desirability(values, center=center, width=width, p=p)
        return (
            pd.Series(scores, index=values.index, dtype=float),
            pd.Series(np.zeros(len(values), dtype=float), index=values.index, dtype=float),
        )
    offsets = np.linspace(-float(error_half_width), float(error_half_width), int(integration_points))
    matrix = values.to_numpy(dtype=float)[:, None] + offsets[None, :]
    scores = generalized_bell_desirability(matrix, center=center, width=width, p=p)
    mean = scores.mean(axis=1)
    variance = scores.var(axis=1)
    return (
        pd.Series(mean, index=values.index, dtype=float),
        pd.Series(np.sqrt(np.maximum(variance, 0.0)), index=values.index, dtype=float),
    )


def bounded_interval_pass_score(values: pd.Series, spec_low: float, spec_high: float, error_half_width: float) -> pd.Series:
    y = values.astype(float)
    if float(error_half_width) <= 0.0:
        return ((y >= spec_low) & (y <= spec_high)).astype(float)
    lower = y - float(error_half_width)
    upper = y + float(error_half_width)
    overlap = (np.minimum(upper, spec_high) - np.maximum(lower, spec_low)).clip(lower=0.0)
    return (overlap / (2.0 * float(error_half_width))).clip(lower=0.0, upper=1.0)


def add_noise_aware_labels(samples: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    target_cfg = config["target_spec"]
    label_cfg = config["labels"]
    center = float(target_cfg["center"])
    tol = float(target_cfg["tolerance_half_width"])
    spec_low = float(target_cfg["spec_low"])
    spec_high = float(target_cfg["spec_high"])
    default_error = float(label_cfg["measurement_error_default"])
    process_sigma = float(label_cfg["process_sigma"])
    align_sigma = float(label_cfg["align_sigma"])
    error_combination = str(label_cfg.get("error_combination", "additive"))
    if error_combination == "additive":
        error_eff = default_error + process_sigma + align_sigma
    else:
        error_eff = max(default_error, process_sigma, align_sigma)

    result = samples.copy()
    y = result["t90"].astype(float)
    result["error_eff"] = error_eff
    cd_method = str(label_cfg.get("cd_method", "linear_uncertain"))
    if cd_method in {"generalized_bell", "generalized_bell_uncertain", "bell", "bell_uncertain"}:
        width = float(label_cfg.get("bell_width", 0.30))
        p = float(label_cfg.get("bell_p", 6.0))
        cd_uncertainty_method = str(label_cfg.get("cd_uncertainty_method", "point"))
        result["cd_raw"] = generalized_bell_desirability(y, center=center, width=width, p=p)
        if cd_uncertainty_method == "bounded_uniform":
            result["cd_mean"], result["cd_std"] = generalized_bell_uncertain_bounded_uniform(
                y,
                center=center,
                width=width,
                p=p,
                error_half_width=error_eff,
                integration_points=int(label_cfg["integration_points"]),
            )
        else:
            result["cd_mean"] = result["cd_raw"]
            result["cd_std"] = 0.0
    else:
        result["cd_raw"] = centered_desirability_point(y, center, tol)
        result["cd_mean"] = centered_desirability_uncertain(
            y,
            center=center,
            tolerance=tol,
            error_half_width=default_error,
            integration_points=int(label_cfg["integration_points"]),
        )
        result["cd_std"] = (result["cd_raw"] - result["cd_mean"]).abs()
    result["cd_std_ref"] = result["cd_std"]

    pass_method = str(label_cfg.get("p_pass_soft_method", "bounded_uniform"))
    if pass_method == "hard":
        result["p_pass_soft"] = ((y >= spec_low) & (y <= spec_high)).astype(float)
    else:
        result["p_pass_soft"] = bounded_interval_pass_score(y, spec_low, spec_high, error_eff)
    result["p_fail_soft"] = 1.0 - result["p_pass_soft"]
    result["is_in_spec_obs"] = ((y >= spec_low) & (y <= spec_high)).astype(int)
    result["is_out_spec_obs"] = 1 - result["is_in_spec_obs"]
    result["is_center_band_obs"] = ((y >= float(target_cfg["center_band_low"])) & (y <= float(target_cfg["center_band_high"]))).astype(int)

    for error in label_cfg["measurement_error_scenarios"]:
        error_value = float(error)
        suffix = str(error_value).replace(".", "p")
        scenario_error = error_value + process_sigma + align_sigma if error_combination == "additive" else max(error_value, process_sigma, align_sigma)
        if pass_method == "hard":
            result[f"p_pass_soft_e{suffix}"] = ((y >= spec_low) & (y <= spec_high)).astype(float)
        else:
            result[f"p_pass_soft_e{suffix}"] = bounded_interval_pass_score(y, spec_low, spec_high, scenario_error)
        result[f"true_outspec_prob_e{suffix}"] = 1.0 - result[f"p_pass_soft_e{suffix}"]
        if cd_method in {"generalized_bell", "generalized_bell_uncertain", "bell", "bell_uncertain"}:
            if str(label_cfg.get("cd_uncertainty_method", "point")) == "bounded_uniform":
                result[f"cd_mean_e{suffix}"], result[f"cd_std_e{suffix}"] = generalized_bell_uncertain_bounded_uniform(
                    y,
                    center=center,
                    width=float(label_cfg.get("bell_width", 0.30)),
                    p=float(label_cfg.get("bell_p", 6.0)),
                    error_half_width=scenario_error,
                    integration_points=int(label_cfg["integration_points"]),
                )
            else:
                result[f"cd_mean_e{suffix}"] = result["cd_raw"]
                result[f"cd_std_e{suffix}"] = 0.0
        else:
            result[f"cd_mean_e{suffix}"] = centered_desirability_uncertain(
                y,
                center=center,
                tolerance=tol,
                error_half_width=error_value,
                integration_points=int(label_cfg["integration_points"]),
            )

    distance_to_boundary = pd.concat([(y - spec_low).abs(), (y - spec_high).abs()], axis=1).min(axis=1)
    result["distance_to_spec_boundary"] = distance_to_boundary
    for band in label_cfg["boundary_band_list"]:
        band_value = float(band)
        suffix = str(band_value).replace(".", "p")
        result[f"boundary_band_{suffix}"] = (distance_to_boundary <= band_value).astype(int)

    repeat_weight = np.ones(len(result), dtype=float)
    if "t90_repeat_std" in result.columns:
        repeat_std = result["t90_repeat_std"].fillna(0.0).astype(float)
        repeat_weight = np.exp(-((repeat_std / float(label_cfg["repeat_std_tolerance"])) ** 2)).clip(0.0, 1.0)
    result["repeat_consistency_weight"] = repeat_weight

    boundary_weight = np.ones(len(result), dtype=float)
    boundary_cfg = label_cfg["boundary_weight"]
    if bool(boundary_cfg["enabled"]):
        boundary_weight = np.where(
            distance_to_boundary <= float(boundary_cfg["band"]),
            float(boundary_cfg["weight_inside_band"]),
            1.0,
        )
    result["boundary_weight"] = boundary_weight
    out_spec_cfg = label_cfg.get("out_spec_weight", {})
    if bool(out_spec_cfg.get("enabled", False)):
        out_spec_weight = np.where(result["is_out_spec_obs"].astype(int) == 1, float(out_spec_cfg.get("multiplier", 2.5)), 1.0)
    else:
        out_spec_weight = np.ones(len(result), dtype=float)
    result["out_spec_weight"] = out_spec_weight

    for col in ["align_confidence", "state_confidence"]:
        if col not in result.columns:
            result[col] = 1.0
    weight = (
        result["align_confidence"].astype(float)
        * result["state_confidence"].astype(float)
        * result["repeat_consistency_weight"].astype(float)
        * result["boundary_weight"].astype(float)
        * result["out_spec_weight"].astype(float)
    )
    result["sample_weight"] = weight.clip(
        lower=float(label_cfg["minimum_sample_weight"]),
        upper=float(label_cfg.get("maximum_sample_weight", 1.0)),
    )
    return result
