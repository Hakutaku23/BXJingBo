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

## Output files

- `artifacts/t90_feature_table.csv`
  Joined sample-level feature table built from LIMS values and lookback-window DCS aggregates.
- `artifacts/t90_analysis_summary.json`
  Summary metrics, top DCS candidates, top LIMS context features, and APC-oriented recommendation deltas.
- `artifacts/cabr_range_recommendation.json`
  Context-conditioned calcium and bromine feasible ranges for a target sample.

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
