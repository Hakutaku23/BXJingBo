# T90 Dev Analysis

This directory is for research and validation only. It is not part of the delivery boundary.

## Current script

- `analyze_t90_mapping.py`
  Loads the private LIMS and DCS data, cleans the LIMS workbook, aligns each sample to a historical DCS window, and evaluates three development-stage strategies:
  - T90 regression for rough explanatory power
  - in-spec classification for `7.5 <= T90 <= 8.5`
  - nearest-good-sample recommendation for APC-oriented setpoint guidance
- `recommend_cabr_ranges.py`
  Treats calcium and bromine as the only controllable variables and the remaining observed process state as context. It outputs conditional feasible ranges for calcium and bromine under the current context, with calcium as the primary recommended control handle.
- `window_size_experiment.py`
  Sweeps multiple DCS window sizes in development mode and compares recommendation quality by aligned sample count, calcium/bromine error, in-spec range coverage, and a heuristic composite score.
- `multiscale_window_experiment.py`
  Builds combined feature sets from multiple DCS windows such as `15+30` or `15+60`, then compares whether multiscale context improves calcium/bromine recommendation quality over single-window baselines.
- `ph_lag_experiment.py`
  Validates whether the dedicated PH signal `B4-AI-C53001A.PV.F_CV.xlsx` leads LIMS `T90` by a stable lag window, using lagged PH features, correlation scans, and a time-ordered ridge baseline.
- `ph_augmented_window_experiment.py`
  Uses the current `50min` DCS recommendation baseline, augments it with lagged PH features, and checks whether any PH lag actually improves calcium/bromine recommendation quality.
- `ph_segmented_window_experiment.py`
  Splits the current `50min` DCS + PH lag experiment by time segment, then checks whether different periods prefer different PH lags.
- `stage_identifier_experiment.py`
  Builds a DCS-only stage identifier on the current `50min` baseline, then decides stage by stage whether PH should be enabled and which lag is worth keeping.
- `stage_aware_prototype.py`
  Replays a second-phase prototype policy: use `50min DCS` to identify the current stage first, then choose `PH off` or `PH(120min) on` by stage and compare that policy against the pure `50min DCS` baseline.

## Data assumptions

- LIMS data comes from `data/t90-溴丁橡胶.xlsx`.
- DCS and PH data come from DCS exports, currently represented by `data/merge_data.csv` or the smaller `data/merge_data_otr.csv`.
- LIMS rows are not one-row-per-sample. The script first groups rows by sampling time and keeps the first non-null value for each indicator.
- The recommendation idea is intentionally case-based. It does not assume T90 can be predicted with high precision from DCS alone.

## Recommended workflow

Full candidate-point scan:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\analyze_t90_mapping.py --lookback-minutes 120
```

Fast iteration with the smaller DCS subset:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\analyze_t90_mapping.py --use-otr --lookback-minutes 120
```

Reuse an already exported feature table:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\analyze_t90_mapping.py --feature-table projects\T90\dev\artifacts\t90_feature_table.csv
```

Recommend calcium and bromine ranges for the latest out-of-spec sample:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\recommend_cabr_ranges.py --feature-table projects\T90\dev\artifacts\t90_feature_table.csv
```

Compare multiple DCS window sizes:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\window_size_experiment.py --windows 5,10,15,20,30,45,60
```

Compare multiscale window combinations:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\multiscale_window_experiment.py --combinations 15,20,15+30,15+60,20+60
```

Validate the PH-to-T90 lag on all offline data:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\ph_lag_experiment.py --min-lag 0 --max-lag 360 --step 5 --feature-window 30
```

Check whether lagged PH helps the `50min` DCS recommendation baseline:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\ph_augmented_window_experiment.py --lags 0,30,60,90,120,150,180,210,240,270,300 --ph-feature-window 50
```

Check whether different time segments prefer different PH lags:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\ph_segmented_window_experiment.py --lags 120,240,300 --segment-freq Q --ph-feature-window 50
```

Build a DCS-only stage identifier and derive a PH enablement policy:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\stage_identifier_experiment.py --lags 120,240,300 --stage-counts 2,3,4,5,6 --ph-feature-window 50
```

Replay the stage-aware second-phase prototype against the pure `50min DCS` baseline:

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\stage_aware_prototype.py --ph-lags 120,240,300 --stage-counts 2,3,4,5,6 --ph-feature-window 50
```

## Output files

- `artifacts/t90_feature_table.csv`
  Joined sample-level feature table built from LIMS values and lookback-window DCS aggregates.
- `artifacts/t90_analysis_summary.json`
  Summary metrics, top DCS candidates, top LIMS context features, and APC-oriented recommendation deltas.
- `artifacts/cabr_range_recommendation.json`
  Context-conditioned calcium and bromine feasible ranges for a target sample.
- `artifacts/window_size_experiment_summary.csv`
  Side-by-side comparison table for each tested window size.
- `artifacts/window_size_experiment_summary.json`
  Best-window summary and full per-window metrics.
- `artifacts/window_size_experiment_metrics.png`
  Visual comparison of aligned samples, calcium/bromine error, in-spec range coverage, and composite score across window sizes.
- `artifacts/multiscale_window_experiment_summary.csv`
  Comparison table for each tested multiscale combination.
- `artifacts/multiscale_window_experiment_summary.json`
  Best-combination summary and full per-combination metrics.
- `artifacts/multiscale_window_experiment_metrics.png`
  Visual comparison of recommendation quality across multiscale combinations.
- `artifacts/ph_lag_experiment_summary.csv`
  Per-lag PH-to-T90 validation table including aligned sample count, correlation metrics, and time-ordered ridge metrics.
- `artifacts/ph_lag_experiment_summary.json`
  Best-lag summary, top-ranked lag candidates, and whether the observed peak sits inside the process-claimed lag window.
- `artifacts/ph_lag_experiment_metrics.png`
  Visual scan of sample count, PH-to-T90 correlation, and regression quality across lag minutes.
- `artifacts/ph_augmented_window_experiment_summary.csv`
  Comparison table for each tested PH lag after augmenting the current `50min` DCS recommendation baseline.
- `artifacts/ph_augmented_window_experiment_summary.json`
  Best-lag summary, baseline comparison, and whether the process-claimed PH lag window improves recommendation quality.
- `artifacts/ph_augmented_window_experiment_metrics.png`
  Visual comparison of recommendation error, in-spec coverage, and composite score after adding lagged PH features.
- `artifacts/ph_segmented_window_experiment_summary.csv`
  Segment-by-segment comparison table for the `50min` baseline and each tested PH lag.
- `artifacts/ph_segmented_window_experiment_summary.json`
  Best lag in each segment, winning-lag consensus across segments, and support level for the process-claimed lag range.
- `artifacts/ph_segmented_window_experiment_composite_heatmap.png`
  Heatmap of composite-score improvement versus the `50min` baseline for each segment and PH lag.
- `artifacts/ph_segmented_window_experiment_calcium_heatmap.png`
  Heatmap of in-spec calcium coverage improvement versus the `50min` baseline for each segment and PH lag.
- `artifacts/stage_identifier_experiment_summary.csv`
  Stage-by-stage comparison table for the DCS-only baseline and each tested PH lag.
- `artifacts/stage_identifier_experiment_summary.json`
  Chosen stage count, stage profiles, best PH policy by stage, and whether PH is worth carrying into the second development phase.
- `artifacts/stage_identifier_experiment_composite_heatmap.png`
  Heatmap of composite-score improvement versus the `50min` baseline for each identified stage and PH lag.
- `artifacts/stage_identifier_experiment_calcium_heatmap.png`
  Heatmap of in-spec calcium coverage improvement versus the `50min` baseline for each identified stage and PH lag.
- `artifacts/stage_aware_prototype_summary.csv`
  Final side-by-side metric table for the pure `50min DCS` baseline and the stage-aware second-phase prototype.
- `artifacts/stage_aware_prototype_summary.json`
  Chosen stage count, stage-aware PH policy, and the prototype-versus-baseline comparison report.
- `artifacts/stage_aware_prototype_metrics.png`
  Visual side-by-side comparison of error, coverage, and composite score for the baseline and the stage-aware prototype.

## How to read the output

- `regression.top_features.dcs_candidate`
  DCS points that best explain T90 variation in the current random-forest baseline.
- `classification.top_features.dcs_candidate`
  DCS points that best separate in-spec and out-of-spec samples.
- `recommendations.recommended_dcs_targets`
  Median differences between out-of-spec samples and their nearest in-spec neighbors in standardized feature space. This is the current best proxy for APC target adjustment direction.
- `recommendations.context_shifts`
  LIMS-side disturbance or composition changes that appear alongside T90 shifts and should be used as context rather than APC direct control targets.
- `cabr_range_recommendation.json`
  A local-context recommendation report. The most directly actionable field is `calcium_range_given_current_bromine`, because calcium is the primary shop-floor control handle you described.

## Important caveats

- These results are development-stage heuristics, not causal proof.
- Candidate DCS points must still be reviewed with process engineers before they are treated as APC manipulable variables.
- LIMS variables such as bromine, calcium, and volatile content are likely disturbance/context signals, not necessarily APC direct control points.

## Current dev baseline

- The current second-phase R&D baseline for single-window online context is `50min`.
- `60min` is still worth keeping as a comparison point because it remains slightly better on calcium MAE alone.
- Delivery defaults remain unchanged until these dev-only findings are intentionally promoted.
