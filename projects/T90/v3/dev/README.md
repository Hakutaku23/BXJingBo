# V3 Dev

This directory stores V3-only research and development work.

Current Phase 1 priority:

- same-sample out-of-spec warning

Recommended layout:

- `experiments/` for named experiment scripts
- `artifacts/` for generated metrics and figures
- `baselines/` for reproducible baseline pipelines

Do not mix old root-level `projects/T90/dev/` scripts into this directory.

Current script index:

- `experiments/phase1_warning_data_audit.py`: audit V3 Phase 1 data coverage, label balance, and window alignment feasibility.
- `experiments/phase1_dcs_feature_screening.py`: run V3-native DCS feature and sensor screening without relying on the legacy point list.
- `experiments/phase1_future_window_warning_experiment.py`: compare future-window warning labels against the current same-sample warning baseline.
- `experiments/phase1_dual_head_current_future_experiment.py`: validate a dual-head design where current alarm and future warning use independent DCS windows and independent models.
- `experiments/phase1_dual_head_model_family_benchmark.py`: benchmark different model families independently for the current-alarm head and the future-warning head.
- `experiments/phase1_alarm_probability_calibration_experiment.py`: evaluate whether the current-alarm head should output calibrated out-of-spec probability instead of only hard labels.
- `experiments/phase1_three_class_probability_experiment.py`: evaluate direct three-class probabilities for below-spec / in-spec / above-spec using the current short-window feature chain.
- `experiments/phase1_three_class_feature_research.py`: re-search window, sensors, stats, and model family after changing the current head to a three-class probability target.
- `experiments/phase1_future_head_lag_diagnosis.py`: diagnose lag and causal window choices for the future-head before entering EWMA/distillation design.
- `experiments/phase1_current_head_ewma_distillation_experiment.py`: validate whether longer causal EWMA-weighted windows can improve the current three-class head beyond the short-window baseline.
- `experiments/phase1_current_head_multiscale_weighted_fusion_experiment.py`: compare short-window baseline against multiscale window fusion and window-level weighted aggregation for the current three-class head.
- `experiments/paper_faithful_ewma_current_head_experiment.py`: run the paper-faithful current-head EWMA applicability test with explicit tau, recursive EWMA condensation, and train-fold-only sensor re-screening.
- `baselines/phase1_same_sample_logistic_window_scan.py`: all-point same-sample logistic baseline across candidate windows.
- `baselines/phase1_compact_warning_modeling.py`: compact warning baselines using screened sensors and compact stat sets.
- `baselines/phase1_warning_alarm_policy_experiment.py`: convert model probabilities into warning/alarm two-level policies and compare threshold strategies.
- `baselines/phase1_short_window_policy_compare.py`: compare short-window policy candidates such as 8 min and 10 min.
- `baselines/phase1_event_level_warning_evaluation.py`: evaluate short-window warning models at bad-segment/event level rather than only sample level.
