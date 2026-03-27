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
