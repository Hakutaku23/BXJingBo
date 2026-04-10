from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def normal_cdf(values: pd.Series | np.ndarray | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


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


def centered_desirability_gaussian(values: pd.Series, center: float, tolerance: float, alpha: float) -> pd.Series:
    sigma = tolerance / math.sqrt(2.0 * math.log(1.0 / alpha))
    arr = values.to_numpy(dtype=float)
    return pd.Series(np.exp(-((arr - center) ** 2) / (2.0 * sigma**2)), index=values.index, dtype=float)


def generalized_bell_desirability(values: pd.Series | np.ndarray, center: float, width: float, p: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + (np.abs(arr - center) / float(width)) ** float(p))


def generalized_bell_uncertain_normal(
    values: pd.Series,
    center: float,
    width: float,
    p: float,
    sigma_eff: float,
    integration_points: int,
    integration_sigma_span: float,
) -> tuple[pd.Series, pd.Series]:
    offsets = np.linspace(
        -float(integration_sigma_span) * float(sigma_eff),
        float(integration_sigma_span) * float(sigma_eff),
        int(integration_points),
    )
    weights = np.exp(-0.5 * (offsets / float(sigma_eff)) ** 2)
    weights = weights / weights.sum()
    matrix = values.to_numpy(dtype=float)[:, None] + offsets[None, :]
    scores = generalized_bell_desirability(matrix, center=center, width=width, p=p)
    mean = scores.dot(weights)
    variance = ((scores - mean[:, None]) ** 2).dot(weights)
    return (
        pd.Series(mean, index=values.index, dtype=float),
        pd.Series(np.sqrt(np.maximum(variance, 0.0)), index=values.index, dtype=float),
    )


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
    sigma_eff = math.sqrt(default_error**2 + process_sigma**2 + align_sigma**2)

    result = samples.copy()
    y = result["t90"].astype(float)
    result["sigma_eff"] = sigma_eff
    cd_method = str(label_cfg.get("cd_method", "linear_uncertain"))
    if cd_method in {"generalized_bell", "generalized_bell_uncertain", "bell", "bell_uncertain"}:
        width = float(label_cfg.get("bell_width", 0.30))
        p = float(label_cfg.get("bell_p", 6.0))
        sigma_span = float(label_cfg.get("integration_sigma_span", 4.0))
        result["cd_raw"] = generalized_bell_desirability(y, center=center, width=width, p=p)
        result["cd_mean"], result["cd_std"] = generalized_bell_uncertain_normal(
            y,
            center=center,
            width=width,
            p=p,
            sigma_eff=sigma_eff,
            integration_points=int(label_cfg["integration_points"]),
            integration_sigma_span=sigma_span,
        )
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
    result["cd_gaussian"] = centered_desirability_gaussian(
        y,
        center=center,
        tolerance=tol,
        alpha=float(label_cfg["gaussian_boundary_alpha"]),
    )
    result["cd_std_ref"] = result["cd_std"]

    pass_prob = normal_cdf((spec_high - y) / sigma_eff) - normal_cdf((spec_low - y) / sigma_eff)
    result["p_pass_soft"] = np.clip(pass_prob, 0.0, 1.0)
    result["p_fail_soft"] = 1.0 - result["p_pass_soft"]
    result["is_in_spec_obs"] = ((y >= spec_low) & (y <= spec_high)).astype(int)
    result["is_out_spec_obs"] = 1 - result["is_in_spec_obs"]
    result["is_center_band_obs"] = ((y >= float(target_cfg["center_band_low"])) & (y <= float(target_cfg["center_band_high"]))).astype(int)

    for error in label_cfg["measurement_error_scenarios"]:
        error_value = float(error)
        suffix = str(error_value).replace(".", "p")
        scenario_sigma = math.sqrt(error_value**2 + process_sigma**2 + align_sigma**2)
        scenario_pass = normal_cdf((spec_high - y) / scenario_sigma) - normal_cdf((spec_low - y) / scenario_sigma)
        result[f"p_pass_soft_e{suffix}"] = np.clip(scenario_pass, 0.0, 1.0)
        result[f"true_outspec_prob_e{suffix}"] = 1.0 - result[f"p_pass_soft_e{suffix}"]
        if cd_method in {"generalized_bell", "generalized_bell_uncertain", "bell", "bell_uncertain"}:
            result[f"cd_mean_e{suffix}"], result[f"cd_std_e{suffix}"] = generalized_bell_uncertain_normal(
                y,
                center=center,
                width=float(label_cfg.get("bell_width", 0.30)),
                p=float(label_cfg.get("bell_p", 6.0)),
                sigma_eff=scenario_sigma,
                integration_points=int(label_cfg["integration_points"]),
                integration_sigma_span=float(label_cfg.get("integration_sigma_span", 4.0)),
            )
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

    for col in ["align_confidence", "state_confidence"]:
        if col not in result.columns:
            result[col] = 1.0
    weight = (
        result["align_confidence"].astype(float)
        * result["state_confidence"].astype(float)
        * result["repeat_consistency_weight"].astype(float)
        * result["boundary_weight"].astype(float)
    )
    result["sample_weight"] = weight.clip(lower=float(label_cfg["minimum_sample_weight"]), upper=1.0)
    return result
