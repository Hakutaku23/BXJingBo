# Scripts

This directory stores runnable experiment scripts for the tabular framework
validation workspace.

- `run_autogluon_stage1_quickcheck.py`
  Stage 1 quick validation on uncleaned-source data with minimal necessary preprocessing.
- `run_autogluon_stage0_baseline.py`
  Stage-based `S0` baseline constructor aligned with `task.md`, producing the unified 120-minute causal snapshot table, results, feature catalog, leaderboard, and feature importance.
- `run_autogluon_stage1_lag_scale.py`
  `S1` lag-scale package validation on top of the `S0` baseline, now restricted to `centered_desirability` and `five_bin`.
- `run_autogluon_stage2_dynamic_morphology.py`
  `S2` controlled dynamic-morphology validation that compares the best `S1` lag-scale package against the same package plus hand-crafted dynamic features.
- `run_autogluon_stage3_regime_state.py`
  `S3` controlled regime/state validation that compares the best `S1` lag-scale package against several state-feature packages fit only within each training fold.
- `run_autogluon_stage4_interactions.py`
  `S4` controlled process-interaction validation that adds a limited set of hand-crafted interaction packages on top of the currently best validated input package for each task.
- `run_autogluon_stage5_quality.py`
  `S5` controlled data-quality / sensor-health validation that adds quality feature packages on top of each task's current best validated input package.
- `run_autogluon_stage6_centered_quality.py`
  `S6` centered-quality validation that adds centered-process features only for `centered_desirability` on top of the best validated centered input package from `S5`.
- `run_autogluon_stage7_final_selection.py`
  `S7` final controlled re-selection that consolidates validated packages from `S1` to `S6` and locks one final recipe for `centered_desirability` and `five_bin`.
- `run_autogluon_stage2_feature_engineering.py`
  Shared Stage 2 engineered snapshot experiment for mixed `high_risk` and `centered_desirability`.
- `run_autogluon_stage2_desirability.py`
  Desirability-only Stage 2 branch using a continuous soft target.
- `run_autogluon_stage2_high_risk.py`
  High-risk-only Stage 2 diagnostic branch using a hard binary label.
- `run_autogluon_stage2_soft_probability.py`
  Soft out-of-spec probability branch using a fuzzy continuous risk target derived from T90 distance to the spec boundary.
- `run_autogluon_stage2_soft_probability_tuning.py`
  Controlled local tuning for the soft probability branch across lookback, boundary softness, and top-k feature count.
- `run_autogluon_stage2_soft_probability_label_family.py`
  Controlled comparison of fuzzy label mapping families while holding the X-side Stage 2 engineering fixed.
- `run_autogluon_stage2_soft_probability_x_enrichment.py`
  Controlled comparison of additional X-side process-statistic feature bundles for the validated soft-label branch.
- `run_autogluon_stage2_soft_probability_feature_distillation.py`
  Controlled feature distillation experiment for the best range-position family under the validated soft-label setup.
