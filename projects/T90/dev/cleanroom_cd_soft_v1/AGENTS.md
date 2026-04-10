# AGENTS.md

## Scope

- This file applies to `cleanroom_cd_soft_v1/` and all of its subdirectories.
- This directory is a **cleanroom experiment area** used to validate a new labeling and modeling track.
- Code, documentation, and outputs in this directory **must not depend on code from legacy experiment directories, notebook state, or any implicit runtime environment**.

## Required Reading Before You Start

Before modifying code or adding a new experiment, you must read and confirm the following files:

1. `README.md`
2. `docs/experiment_plan_cn.md`
3. `docs/centered_desirability_method.md`
4. `configs/base.yaml`

If the documents above are inconsistent with the implementation, **update the documents first, then continue with the experiment**.

## Objective of This Directory

This directory is for exactly one purpose:

- Validate a clean and auditable new workflow under measurement error, fuzzy qualification boundaries, and noisy samples:
  - use `centered_desirability` as the primary supervision target;
  - use `soft target` as the auxiliary probability supervision target;
  - use `uncertainty` / `sample_weight` as the noise-modeling and down-weighting mechanism;
  - use `DCS self-supervised pretraining + LIMS supervised fine-tuning` as the primary training paradigm.

## Working Rules

### 1. Dependencies and Implementation

- Do not directly `import` trainers, feature engineering code, model definitions, or evaluation scripts from legacy experiment directories.
- Reusing ideas is allowed, but the implementation must be rewritten inside this directory and remain clearly traceable.
- All primary workflows must be runnable directly through `scripts/*.py`.
- Notebooks may be used only for visualization or temporary analysis. They must not contain the primary workflow.

### 2. Data and Splitting

- Do not use random splits as the basis for primary conclusions.
- Train / validation / test splits must prioritize the following order:
  1. group by batch / operating condition / work order;
  2. then apply time-based train-before-test splitting;
  3. then apply purge windows to adjacent samples to avoid near-neighbor leakage.
- All DCS windows derived from the same LIMS sample must stay on the same side of the split.

### 3. Labels and Supervision

- Do not treat LIMS labels as deterministic ground truth.
- Every supervised task must support at least one of the following mechanisms:
  - `sample_weight`
  - `heteroscedastic uncertainty`
  - softening for boundary samples
  - `unknown / abstain` handling
- The computation of `centered_desirability` must explicitly account for:
  - measurement error;
  - alignment error;
  - possible false-state / non-stationary samples;
  - asymmetric tolerance on the left and right sides.
- The authoritative definition of the computation is `docs/centered_desirability_method.md`.

### 4. Model and Experiment Order

- Do not start with a complex model before label governance is in place.
- The experiment order is fixed:
  1. data alignment and label construction;
  2. strict data splitting;
  3. interpretable strong baselines;
  4. self-supervised representations;
  5. joint supervision;
  6. calibration and boundary-band evaluation.
- Model capacity may be increased only after both strong baselines and lightweight temporal models are already stable.

### 5. Outputs and Reproducibility

- All artifacts must be written to `outputs/<run_id>/`.
- Each `run_id` must be unique. Recommended format: `YYYYMMDD_HHMMSS_<tag>`.
- Every run must preserve at least:
  - a configuration snapshot;
  - split indices;
  - label-construction parameters;
  - training logs;
  - evaluation results;
  - key figures.

## Documentation and Write-Back Requirements

Before starting any new experiment, you must first add a UTF-8 encoded Chinese entry to `docs/experiment_plan_cn.md` covering:

- experiment objective
- data scope
- alignment strategy
- label definition
- splitting strategy
- metric definition
- expected risks

After completing any experiment, you must add a UTF-8 encoded Chinese entry to `docs/experimental_procedure_cn.md` covering:

- actual execution commands
- output directory
- key parameters
- differences from the previous experiment
- conclusions
- failure points / uncertainties

If you modify the definition of `centered_desirability`, `soft target`, `sample_weight`, or `uncertainty`, you must also update all of the following:

- `docs/centered_desirability_method.md`
- `configs/base.yaml`
- comments in the related implementation files

## Recommended Execution Order

1. `scripts/prepare_dataset.py`
2. `scripts/train_ssl.py`
3. `scripts/train_supervised.py`
4. `scripts/evaluate.py`

## Result Evaluation Principles

Do not draw conclusions from a single metric. At minimum, check all of the following:

- `pass probability`: ROC-AUC / PR-AUC / Brier / ECE / NLL
- `centered_desirability`: MAE / RMSE / Spearman
- `boundary band metrics`: separate evaluation inside the `±0.10 / ±0.15 / ±0.20` bands
- `calibration`: reliability curves / bin-wise calibration error
- `robustness`: stability across different time ranges / batches / operating conditions
- `uncertainty usefulness`: whether high-uncertainty samples cover more boundary misclassifications and noisy samples

## Final Deliverables

The output of this directory is not a single pass/fail label. It is a set of more useful outputs:

- pass probability
- fail probability
- continuous quality score
- predictive uncertainty
- risk score for human-review prioritization
