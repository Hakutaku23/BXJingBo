# Scripts

This directory stores runnable experiment scripts for the tabular framework
validation workspace.

- `run_autogluon_stage1_quickcheck.py`
  Stage 1 quick validation on uncleaned-source data with minimal necessary preprocessing.
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
