"""Reusable core utilities for the T90 V3 workspace."""

from .data_contract import (
    SourcePaths,
    TargetSpec,
    add_out_of_spec_labels,
    build_dcs_feature_table,
    infer_lims_column_map,
    load_dcs_frame,
    load_lims_samples,
    load_ph_frame,
    summarize_numeric_window,
)

V3_CORE_STATUS = "phase1_bootstrap"

__all__ = [
    "SourcePaths",
    "TargetSpec",
    "add_out_of_spec_labels",
    "build_dcs_feature_table",
    "infer_lims_column_map",
    "load_dcs_frame",
    "load_lims_samples",
    "load_ph_frame",
    "summarize_numeric_window",
    "V3_CORE_STATUS",
]
