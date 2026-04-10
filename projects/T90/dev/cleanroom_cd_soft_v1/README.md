# cleanroom_cd_soft_v1

This directory is a self-contained cleanroom experiment area for validating a
noise-aware T90 labeling and modeling workflow.

The current line is development support only. It does not modify or replace the
stable T90 delivery boundary.

## Current Exp-001 Scope

Exp-001 builds the auditable data foundation:

- load LIMS T90 samples from `projects/T90/data/t90-溴丁橡胶.xlsx`;
- load DCS histories from `projects/T90/data/merge_data.csv` and the optional
  supplemental DCS file;
- construct causal DCS windows before each LIMS sample time;
- detect and remove redundant DCS point columns before feature construction;
- aggregate duplicate LIMS records at the same sample time;
- build noise-aware labels:
  - `cd_mean`;
  - `p_pass_soft`;
  - `sample_weight`;
  - `state_confidence`;
  - `align_confidence`;
- create a time-ordered purged split;
- write all artifacts to `outputs/<run_id>/`.

CQDI and RADI are treated as downstream deployability diagnostics. Because prior
runs produced zero scores for both, Exp-001 records the label and split evidence
needed to recalibrate those metrics, but does not use CQDI/RADI as a gate.

## CPU-Only Run

Use the local `autoGluon` conda environment:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\prepare_dataset.py --config configs\base.yaml
```

For a quick smoke run:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\prepare_dataset.py --config configs\base.yaml --max-samples 50 --run-tag smoke
```

Quick supervised validation:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\train_supervised.py --config configs\base.yaml --prepared-run-dir outputs\<run_id> --run-tag exp002_quick
```

Quick validation with optional LIMS context features:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\train_supervised.py --config configs\base.yaml --prepared-run-dir outputs\<run_id> --run-tag exp002_lims_context --include-lims-context --no-autogluon
```

## Main Outputs

Each run writes:

- `config_snapshot.yaml`
- `labeled_samples.csv`
- `feature_table.csv`
- `split_indices.csv`
- `label_summary.json`
- `alignment_quality_report.json`
- `data_boundary_report.json`
- `leakage_report.json`
- `exp001_report.md`

Supervised quick runs additionally write:

- `quick_model_results.csv`
- `quick_model_scored_test_rows.csv`
- `quick_model_summary.json`
- `exp002_quick_model_report.md`

Label/input analysis runs write:

- `label_distribution_dual_axis.png`
- `input_dimension_report.json`
- `input_dimension_report.md`

## Current Feature Controls

The default configuration now applies two redundancy controls:

- DCS point-level deduplication before window feature construction;
- Historian-oriented DCS cleaning before window feature construction:
  - hold-last-value fill;
  - MAD/Hampel spike replacement with time interpolation;
  - Savitzky-Golay smoothing;
- train-only feature screening before modeling:
  - missing-rate filter;
  - non-constant filter;
  - target-correlation candidate ranking on train only;
  - feature-correlation pruning inside the selected candidate pool.

LIMS context features are available only when explicitly requested with
`--include-lims-context`.

The current default label is `generalized_bell_uncertain`: a generalized bell
desirability mapping integrated over the effective measurement/process/alignment
uncertainty.

## Current Window Recommendation

For the next A/B deep dive, use `240 min` as the default DCS window length and
keep `0 / 15 / 60 min` lag candidates during feature selection. The 240-minute
window is currently the most balanced choice under sensor noise and drift:
shorter windows can improve some ranking signals, but they are less stable on
regression error and probability calibration.

Run the two comparable lines as:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\train_supervised.py --prepared-run-dir outputs\<prepared_run_id> --run-tag win240_dcs --window-minutes 240 --no-autogluon
& D:\miniconda3\envs\autoGluon\python.exe scripts\train_supervised.py --prepared-run-dir outputs\<prepared_run_id> --run-tag win240_limsctx --include-lims-context --window-minutes 240 --no-autogluon
```

Label visualization and input dimension report:

```powershell
& D:\miniconda3\envs\autoGluon\python.exe scripts\analyze_labels_and_inputs.py --prepared-run-dir outputs\<prepared_run_id> --main-a-model-run-dir outputs\<dcs_model_run_id> --main-b-model-run-dir outputs\<lims_context_model_run_id>
```
