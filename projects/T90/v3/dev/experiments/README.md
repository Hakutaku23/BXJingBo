# V3 Experiments

Place controlled V3 experiment scripts here.

Each experiment should state:

- the objective
- the label definition
- the input data assumption
- the main metric
- the output artifact path

Current experiment scripts:

- `phase1_warning_data_audit.py`: data coverage and alignment audit for the current Phase 1 warning task.
- `phase1_dcs_feature_screening.py`: DCS feature/sensor screening under the current task definition.
- `phase1_future_window_warning_experiment.py`: future-window warning comparison under the current 8-minute compact mainline.
- `phase1_dual_head_current_future_experiment.py`: dual-head experiment for "current alarm + future warning" with independent window search.
- `phase1_dual_head_model_family_benchmark.py`: independent model-family benchmark for the current-alarm head and the future-warning head.
- `phase1_alarm_probability_calibration_experiment.py`: probability-quality experiment for the current-alarm head, including raw vs calibrated probability comparison.
- `phase1_three_class_probability_experiment.py`: three-class probability experiment for below-spec / in-spec / above-spec outputs.
- `phase1_three_class_feature_research.py`: three-class-specific re-search of windows, sensors, stat sets, and model family for the current head.
