# Single Experiment Specification: Paper-Faithful EWMA Data Distillation Applicability Test (Current Head)

Version: v1.1  
Audience: Codex / in-project developers  
Experiment role: run one controlled, single-purpose, auditable experiment inside the existing project, and answer only one question: is the EWMA data distillation method proposed in the paper applicable to the current project difficulty?

---

## 1. The Only Question This Experiment Must Answer

Under the current project conditions:

- DCS data is sampled every 1 minute;
- laboratory results are sampled every 4 hours;
- laboratory timestamps are uncertain;
- real process lag exists;
- label noise and boundary ambiguity are substantial;

if we strictly follow the core idea of the reference paper and compress each laboratory sample's historical DCS window into a recursive EWMA condensed sample, then use that representation for current-head alarming / three-class modeling:

does it produce a stable, repeatable improvement in alarm-related performance relative to the existing short-window statistics baseline?

This experiment does not answer whether the future head works, whether multiscale weighted fusion works, or whether more complex models are better.

---

## 2. Why This Single Experiment Must Be Rebuilt

The current three scripts expose different pain points and therefore cannot serve as a clean answer to whether the paper method works.

### 2.1 Problem in the current EWMA script
The current `phase1_current_head_ewma_distillation_experiment.py` uses `lambda ** delta_minutes` and computes a normalized weighted average inside a window ending at `decision_time`, without explicitly introducing candidate lag `tau`. This means it is inspired by the paper, but it does not strictly implement the paper's recursive EWMA data distillation, and it does not truly operationalize timeline-alignment relief.

### 2.2 Problem in the multiscale weighted fusion script
`phase1_current_head_multiscale_weighted_fusion_experiment.py` is fundamentally a manual weighted fusion of multiple short-window statistical feature tables, with default window groups such as `8,15,30` and `8,15,45`. This is not the paper's EWMA condensed-sample mechanism. It is closer to window-level late fusion than to data distillation over raw high-frequency sequences.

### 2.3 Problem in the future-head lag diagnosis script
`phase1_future_head_lag_diagnosis.py` introduces `lag_minutes` and causal windows, but still uses ordinary window statistics, and it performs global feature/sensor screening before time-series evaluation. It cannot directly prove whether the paper's EWMA method works. Its value is limited to the insight that explicit lag diagnosis should be preserved.

### 2.4 What the paper actually emphasizes
The paper's key contribution is not a deeper model, but using EWMA data distillation to compress a high-frequency input window into a condensed sample, so that more high-frequency information can be used without increasing the number of labeled samples, while also mitigating timeline alignment problems. The paper also explicitly notes that the attenuation parameter must be tuned case by case, and that EWMA has an inherent low-pass filtering effect.

---

## 3. Strict Scope of This Experiment

### 3.1 Current head only
This experiment evaluates current head only:

- use the current laboratory sample's own label;
- do not create future horizon labels;
- do not perform future out-of-spec prediction;
- do not build future-head event detection.

### 3.2 Only test whether the paper method is applicable
This experiment compares only two representations:

1. Baseline: the currently allowed short-window statistics baseline;
2. Paper-faithful EWMA Distillation: a recursive EWMA condensed sample implemented according to the paper's intent.

Do not introduce:
- multiscale manual weighted fusion;
- complex multi-head structures;
- future-head labels;
- deep models;
- probability post-processing;
- hysteresis;
- event consolidation.

### 3.3 Only allow a limited model family
To isolate the effect of data representation, models must remain fixed and simple:

- `multinomial_logistic`
- `random_forest_balanced`

No additional models are allowed.

---

## 4. Strict Definitions

### 4.1 Decision time
For each laboratory sample, `decision_time` is defined as that sample's `sample_time`.  
Do not modify the original `sample_time`; all temporal shifts must be expressed through `tau`.

### 4.2 Current-head target
Reuse the existing project three-class target:

- `below_spec`
- `in_spec`
- `above_spec`

Do not convert this experiment to future-head labels, soft labels, or dual-binary targets.

### 4.3 Candidate lag `tau`
`tau` means the process-effect endpoint truly corresponding to the current laboratory label, shifted backward from `decision_time`.

Definition:
- `t_end = decision_time - tau`
- `t_start = t_end - W`

`tau` must be an explicit experiment axis and must not be omitted.

### 4.4 Window length `W`
`W` is the historical window length used to construct the condensed sample for the current laboratory label.

### 4.5 EWMA attenuation coefficient `lambda`
`lambda` is used in recursive EWMA:

- closer to 1 means longer memory;
- smaller values place more emphasis on recent values.

In this experiment, `lambda` must be tied to the recursive process, not to a normalized exponential kernel of the form `lambda ** delta_minutes`.

### 4.6 Paper-consistent condensed sample
For one variable with window values ordered from old to new as `x_1, x_2, ..., x_K`, define:

- `s_1 = x_1`
- `s_k = lambda * s_(k-1) + (1 - lambda) * x_k`

The final `s_K` is the EWMA condensed feature for that variable under `(tau, W, lambda)`.

This definition must be used for the main feature in this experiment. It must not be replaced by a normalized exponentially weighted mean.  
`pandas.Series.ewm(alpha=1-lambda, adjust=False).mean().iloc[-1]` may be used as an equivalent recursive implementation.

---

## 5. A/B Structure of the Single Experiment

### 5.1 Baseline A
Use the existing short-window statistical features as the control group:

- do not introduce `tau` search;
- use the current project's current-head baseline window;
- keep the existing baseline statistic family;
- evaluate only with `multinomial_logistic` and `random_forest_balanced`.

### 5.2 Treatment B
Use paper-faithful EWMA Distillation:

- for each laboratory sample, construct a recursive condensed sample under `(tau, W, lambda)`;
- use a single window only, with no multiscale manual weighted fusion;
- evaluate current head only;
- keep the same two simple models.

### 5.3 Core comparison
Determine whether Treatment B shows:
- stable improvement;
- consistency across time folds;
- gains that are not due to a single lucky fold.

---

## 6. Explicitly Forbidden Practices

The following are forbidden in this experiment:

1. Reusing the current EWMA script's `lambda ** delta_minutes` plus normalized weighted average.
2. Fixing the window to `decision_time` without explicit `tau`.
3. Using future-head labels.
4. Using multiscale window weighted fusion.
5. Performing global feature/sensor screening before time-series evaluation.
6. Random shuffle splitting.
7. Sharing fitted scaling or imputation parameters across train/test boundaries.
8. Crossing obvious regime transitions without marking them.
9. Changing labels, model family, and feature design simultaneously so that attribution becomes impossible.
10. Using any deep learning model.

---

## 7. Data and Preprocessing Requirements

### 7.1 Data sources
Reuse the existing project data sources:

- main DCS data;
- supplemental DCS data if available;
- LIMS / laboratory samples.

### 7.2 Missing values
Strict requirements:

- a raw window may contain a small amount of missing values;
- if one variable has too few valid points in a window, mark that variable's feature as missing for that sample;
- final imputation may happen only inside the model pipeline, fitted on the training fold;
- do not perform global imputation before splitting.

### 7.3 Time order
Must guarantee:

- ascending DCS time;
- ascending laboratory sample time;
- all windows are causal;
- no feature uses data after `decision_time`.

### 7.4 Regime transitions
If obvious regime transitions can be identified:

- samples may be retained;
- but it must be recorded whether a sample window crosses a regime boundary;
- at minimum output a boolean flag `crosses_regime_boundary`;
- primary interpretation should prioritize the subset with no regime crossing.

If regime boundaries cannot yet be identified reliably, this must be explicitly stated as a known limitation in the result summary.

---

## 8. Feature Re-screening Rules (new)

Because the dataset actually contains many features, and the historical feature selection may have been unsuitable, this experiment may re-screen features, but only under strict constraints.

### 8.1 Why re-screening is allowed
The purpose of this experiment is to test whether the paper-style EWMA representation is applicable, not whether the previously selected feature list was correct.  
If the feature pool itself is poorly chosen, an otherwise useful EWMA method may be masked by an unsuitable input subset.

### 8.2 Two-layer screening only
Feature screening is allowed only in two layers.

#### Layer 1: unsupervised pre-cleaning
Label-free cleaning is allowed across the full DCS variable pool, for example:

- remove globally constant variables;
- remove variables with excessively high missingness;
- remove variables that are frozen for long periods with almost no variation;
- remove obvious duplicate columns or duplicate names.

This layer must not use any label information.

#### Layer 2: supervised re-screening
Label-aware re-screening is allowed using current-head labels, but only if all of the following hold:

- it is performed strictly inside each training fold;
- the test fold must not participate in any screening decision;
- baseline and treatment must use exactly the same sensor list inside the same fold;
- sensor screening must occur during the training-stage workflow, not once on the full dataset.

### 8.3 Recommended screening object
Prefer screening the raw sensor list rather than performing complex selection over already-expanded derived columns.  
That means deciding which DCS variables to keep first, and then constructing:

- baseline statistical features;
- EWMA recursive features

from the same retained sensors.

### 8.4 Recommended screening methods
Only simple, auditable methods are allowed:

- univariate screening scores;
- train-fold-only ranking by correlation / AUC / mutual information;
- fixed top-k sensors;
- or a fixed score threshold.

Not allowed:

- using full-data embedded importance and then back-selecting;
- large wrapper search;
- genetic algorithms;
- AutoML-style feature search.

### 8.5 top-k constraint
If top-k sensor screening is used, the candidate set must be fixed in advance to a very small set, for example:

- `topk_sensors ∈ {20, 40, 80}`

and `topk_sensors` must be recorded as an explicit experiment axis in the result table.

### 8.6 Comparability requirement
Within the same fold, under the same `topk_sensors`, and under the same sensor list:

- Baseline A and Treatment B must share the same sensor set;
- baseline cannot use one sensor list while EWMA uses another.

Otherwise the difference can no longer be attributed to representation alone.

---

## 9. Feature Definitions

### 9.1 Baseline features
Reuse the existing current-head baseline statistics only. Do not add new statistic families.

### 9.2 Treatment primary features
For each sensor under `(tau, W, lambda)`, output:

- `{sensor}__ewma_recursive`

This is the only mandatory main feature.

### 9.3 Optional auxiliary treatment features
Only the following three optional features may be added, all from the same `(tau, W)` window:

- `{sensor}__last`
- `{sensor}__last_minus_ewma_recursive`
- `{sensor}__ewm_std_recursive`

Where:

- `ewm_std_recursive` may be defined from the same recursive idea or as a weighted variance of residuals around the recursive EWMA;
- if implementation cost is high, start with only `ewma_recursive` and `last_minus_ewma_recursive`;
- but the exact formula must be documented in the report.

### 9.4 Disallowed features
This experiment does not allow:

- multi-window concatenation;
- multiscale concatenation;
- manual window-level fusion;
- future-informed differences;
- any leakage-related derived feature.

---

## 10. Fixed Parameter Grid

To prevent uncontrolled expansion of the search space, the parameter grid is fixed.

### 10.1 Candidate `tau`
Unit: minutes

- 0
- 120
- 240
- 360
- 480

### 10.2 Candidate `W`
Unit: minutes

- 60
- 120
- 240
- 360
- 480

### 10.3 Candidate `lambda`
If recursion is done at 1-minute granularity, use:

- 0.97
- 0.985
- 0.995

Do not use:

- 0.75
- 0.85
- 0.92

Reason: at 1-minute granularity, those values decay too fast and make nominally long windows behave like very short recent windows.

### 10.4 Candidate `topk_sensors` (if re-screening is enabled)
Unit: number of sensors

- 20
- 40
- 80

If the total number of valid sensors is clearly below 80, clip automatically to the available upper bound, but still record the actual value used.

---

## 11. Training and Validation Design

### 11.1 Splitting
Use:

- `TimeSeriesSplit(n_splits=5)`

If the sample count cannot support 5 folds, reducing to 4 is allowed, but never below 3.

### 11.2 Where feature screening may happen
If supervised re-screening is enabled, the execution order must be:

1. split by time into train/test fold;
2. compute sensor screening scores on the training fold only;
3. select the sensor list for that fold;
4. construct both baseline and EWMA features for train/test, but retain only the sensors selected in that fold;
5. train and evaluate the model on that fold.

Do not screen on the full dataset before cross-validation.

### 11.3 Model training
Inside the model pipeline, the following are allowed:

- `SimpleImputer(strategy="median")`
- `StandardScaler()` for Logistic only
- fixed model hyperparameters

Not allowed:

- large-scale tuning;
- Bayesian optimization;
- random search;
- complex AutoML.

---

## 12. Evaluation Metrics

Primary metrics:

- `macro_f1`
- `balanced_accuracy`

Secondary metrics:

- `weighted_f1`
- `multiclass_log_loss`
- `multiclass_brier_score`

### 12.1 Decision rule
A `(tau, W, lambda, model, topk_sensors)` combination is considered better than baseline only if all of the following hold:

1. `macro_f1` is higher than baseline;
2. `balanced_accuracy` is higher than baseline;
3. the gain is not confined to a single fold;
4. at least one of the two simple models behaves stably;
5. performance on the subset without regime-boundary crossing does not degrade.

---

## 13. Required Outputs

Codex must generate the following files.

### 13.1 Result table
- `paper_faithful_ewma_current_head_results.csv`

Each row must correspond to one experiment combination and include:

- model_name
- tau_minutes
- window_minutes
- lambda
- topk_sensors
- feature_family
- samples
- macro_f1
- balanced_accuracy
- weighted_f1
- multiclass_log_loss
- multiclass_brier_score
- valid_fold_count
- crosses_regime_boundary_ratio (if available)

### 13.2 Best-combination summary
- `paper_faithful_ewma_current_head_summary.json`

Must include:

- baseline performance;
- whether feature re-screening was enabled;
- screening strategy description;
- best EWMA combination;
- whether it beats baseline;
- where the advantage comes from;
- whether the result is unstable;
- known limitations.

### 13.3 Feature-row export
- `paper_faithful_ewma_current_head_feature_rows.csv`

At minimum include:

- decision_time
- tau_minutes
- window_minutes
- lambda
- topk_sensors (if enabled)
- each sensor's EWMA feature
- current-head target

### 13.4 Audit note
- `paper_faithful_ewma_current_head_audit.md`

Must explicitly document:

- whether recursive EWMA was strictly used;
- whether `tau` was explicitly used;
- whether feature re-screening was enabled;
- if enabled, whether screening was performed only inside training folds;
- whether regime transitions were identified;
- whether any samples were dropped due to insufficient windows.

---

## 14. Codex Execution Order

Codex must execute in the following order and must not skip steps:

1. reuse the project's existing data-loading and current-head label logic;
2. write a new, independent sensor pre-cleaning function (unsupervised);
3. write a new, independent train-fold-only sensor re-screening function if enabled;
4. write a new, independent feature-construction function:
   - input: `labeled`, `dcs`, `sensors`, `tau_minutes`, `window_minutes`, `lambda_value`
   - output: recursive-EWMA feature table;
5. run baseline first;
6. run the fixed-grid EWMA experiment second;
7. output the unified result table;
8. generate the summary files last.

Do not opportunistically refactor future head, rewrite multiscale fusion, or introduce a new target system during this experiment.

---

## 15. Definition of Success and Failure

### 15.1 Success
If the EWMA treatment yields stable improvement over baseline across multiple time folds and at least one simple model, conclude:

the EWMA data distillation method from the paper appears applicable to the current project and is worth taking to the next round of experiments.

### 15.2 Failure
If the EWMA treatment:

- does not show stable improvement over baseline;
- improves only in one fold or one accidental combination;
- remains unstable even under more reasonable `tau/W/lambda`;

then conclude:

under the current data conditions, the paper method does not yet show clear applicability, and the next step should not be greater complexity, but rather revisiting label quality, regime segmentation, or task framing itself.

Important:  
“Failure” does not mean the paper is wrong. It only means that, under the current project, current data, and current label system, there is not enough evidence to justify further investment in this direction.

---

## 16. What Is Allowed After This Experiment

Only after this experiment is completed and a clear conclusion is documented may the project move to:

- probability alarm targets;
- multiscale EWMA;
- future head;
- regime-aware segmentation;
- multi-head or more complex models.

Before that, all such extensions should remain paused.

---

## 17. Final Instruction to Codex

You are not doing a stronger-model search. You are doing a method-transferability test.

Therefore you must:

- keep the experimental question single-purpose;
- implement the paper's core mechanism strictly;
- allow feature re-screening, but prevent all leakage;
- avoid unrelated extensions;
- not change the task definition merely to get a better result;
- make the result answerable in plain terms:
  - was the paper method actually implemented correctly?
  - is the paper method better than the current baseline on this project?

End of document.
