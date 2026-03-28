# Soft Sensor Alarm Experiment Plan and Validation Objectives
Version: v1.2

## 1. Purpose

This document defines the experiment plan, validation objectives, implementation scope, and acceptance criteria for the soft sensor alarm project, for subsequent collaboration with Codex.

Project context:

- DCS process variables are sampled every **1 minute**.
- Laboratory results are available every **4 hours** (about **4 samples per day**).
- Even if laboratory timestamps are recorded precisely enough, the process still has **intrinsic process lag**.
- Manual laboratory measurements contain **non-negligible assay error**.
- Most samples are concentrated in the normal operating range, while truly out-of-spec samples are relatively rare.
- The business objective is **reliable alarming**, not necessarily high-accuracy point prediction.

The core working hypothesis for this phase is:

> The main bottlenecks in current model performance are more likely to come from sample construction, label uncertainty, and time alignment, rather than from insufficient model complexity alone.

Therefore, this experiment plan should prioritize **data-side redesign** before major model-side escalation.

Current V3 status snapshot:

- The **current-head** should now be treated as a **three-class probability problem**:
  - `below_spec`
  - `in_spec`
  - `above_spec`
- The **future-head** should now be treated as a **binary future-risk probability problem**:
  - whether out-of-spec will occur within a future horizon
- The current-head and future-head are allowed to use:
  - different DCS windows
  - different feature subsets
  - different model families
- Existing V3 evidence suggests:
  - short windows remain competitive for the current-head;
  - broader lag / distilled representations are more promising for the future-head;
  - data construction and target design still appear more important than immediately escalating model complexity.
- The referenced deep soft-sensing literature should be treated as **method inspiration**, not as a requirement to adopt deep semi-supervised models in the current phase.

---

## 2. Core Research Questions

### RQ1. Sample construction
Can alarm performance be improved by constructing **distilled DCS representations** for each laboratory sample, instead of using single-point alignment or simple window matching?

### RQ2. Time alignment and lag
Can better performance be obtained by explicitly searching over **candidate process lags** and **aggregation windows**, instead of assuming a fixed alignment rule?

### RQ3. Label uncertainty
Can practical alarm usefulness be improved by replacing hard classification with **probability-based alarming**, **soft labels**, or **confidence-weighted training**?

### RQ4. Operating regime effects
If data distillation and modeling are performed **within stable operating regimes**, rather than by fitting a single model across mixed regimes, will performance improve?

### RQ5. Model capacity and data representation
After data redesign, which model family offers the best balance among performance, robustness, interpretability, and maintenance cost?

---

## 3. Project Goals

Build and compare a series of alarm-oriented soft sensor experiment pipelines, then determine:

1. Whether **data distillation** improves alarm performance relative to the current baseline.
2. Whether **multi-lag / multi-scale features** outperform fixed-window features.
3. Whether **probability alarm models** outperform hard three-class models.
4. Whether **regime-aware training** improves robustness.
5. Which experimental pipeline should be recommended for the next development stage.

---

## 4. Non-Goals

The following are not primary goals in this phase:

- Directly building a complete production-ready online service.
- Solving all process mechanism issues in one step.
- Guaranteeing an exact and unique physical lag value.
- Pursuing the best possible regression RMSE at all costs.
- Prioritizing highly complex deep learning models before data issues are clarified.

---

## 5. Data Assumptions

Codex should assume that the following data sources exist or can be prepared.

### 5.1 DCS data
- Timestamped process variables at 1-minute frequency.
- Continuous variables, setpoints, valve positions, flows, temperatures, pressures, etc.
- Possible missing values, outliers, sensor freezes, and operational discontinuities.

### 5.2 Laboratory data
- Timestamped laboratory results at roughly 4-hour intervals.
- One or more quality indices used for alarming.
- Possible assay noise and timestamp uncertainty.

---

## 6. Experimental Principles

All experiments should follow these principles:

1. **Time order must be preserved**, and future information must never leak into the past. Even under regime-aware experiments, causal ordering remains the default.
2. **Validation must match time-series data** or **windowed data**, preferably with rolling / expanding time splits.
3. **Feature generation must be causal**.
4. **Alarm effectiveness takes priority over nominal classification accuracy**.
5. **Prioritize data-construction experiments before materially increasing model complexity**.
6. **Every experiment must record assumptions, parameters, and outputs in a reproducible manner**.
7. The current-head and future-head may use different windows, feature sets, and models if that improves practical usefulness.

---

## 7. Baseline Definitions

Before proposing improvements, Codex should reproduce or closely approximate the current baselines.

### Baseline B0: Current window-based classification baseline
Expected characteristics:
- Fixed lag or a selected alignment rule.
- Window-based statistical features.
- Current-head **three-class probability** target, such as:
  - below_spec
  - in_spec
  - above_spec
- Future-head **binary probability** target, such as:
  - future qualified
  - future unqualified

### Baseline B1: Current-head three-class probability baseline
Expected characteristics:
- Short-window or compact-window DCS features.
- Three-class probability output for:
  - below_spec
  - in_spec
  - above_spec
- Downstream decision rules may aggregate:
  - `p_below + p_above`
  - `p_above`
  - `p_below`

### Baseline B2: Future-head binary probability baseline
Expected characteristics:
- A causal DCS representation aligned to a future horizon.
- Output probability that out-of-spec will occur within the future horizon.
- Alarming is derived from probability and policy thresholds.

### Baseline B3: Naive point-matching baseline
Optional diagnostic baseline:
- Match each laboratory sample to a single DCS timestamp or a very narrow window.
- Use this to demonstrate the effect of mismatch and lag.

---

## 8. Experiment Stages

Prioritize the **current-head** model design first. After that is completed, consider the **future-head** design.

## Stage 0. Data audit and reproducibility setup

### Objective
Establish a reliable experimental foundation.

### Tasks
- Check timestamp consistency and timezone handling.
- Quantify missingness, outliers, sensor freezes, and long constant runs.
- Summarize the laboratory target distribution, class balance, and sample density near thresholds.
- Check duplicate records and impossible timestamps.
- Build reusable data split templates and experiment configuration templates.

### Deliverables
- Data audit report.
- Target distribution plots and class counts.
- Preliminary assessment of sample imbalance.
- Reusable dataset registry and split definitions.

### Validation target
- Experimental data is reproducible.
- No obvious leakage or timestamp corruption remains.

---

## Stage 1. Alignment and lag diagnosis

### Objective
Determine whether a fixed alignment assumption is defensible, and separate:
- short-window current-head behavior
- lag-sensitive future-head behavior

### Tasks
- Evaluate correlation or predictive utility across a grid of candidate lags.
- Inspect lag-response patterns for important process variables.
- Compare lag behavior across different operating regimes where possible.
- Determine whether a single global lag is usable, or whether lag is clearly regime-dependent.
- Run a separate short-window search for the current-head, instead of assuming that hour-level lag logic applies equally to the current-head.

### Suggested lag search range
Use two branches:

For the current-head short-window search:
- 8 min
- 10 min
- 12 min
- 15 min
- 20 min
- 30 min
- 45 min

For the future-head lag-sensitive search:
- 0 h
- 1 h
- 2 h
- 4 h
- 6 h
- 8 h
- 12 h

Refine later if needed.

### Deliverables
- Lag sensitivity tables.
- Diagnostic plots by variable and by regime.
- Recommended candidate lag ranges for downstream experiments.

### Validation target
- Determine whether a single-lag assumption is defensible.
- If not, justify multi-lag feature construction.

---

## Stage 2. Distilled sample construction (core stage)

### Objective
Construct **distilled DCS representations** for each laboratory sample, especially for the future-head, instead of relying only on simple window statistics.

### Core idea
For each laboratory timestamp, generate one or more causal condensed samples from preceding DCS data using:

- candidate lag: `tau`
- window length: `W`
- EWMA decay: `lambda`

Formally:

`x_condensed(t_lab; tau, W, lambda)`

### Experimental axes
- candidate lag `tau`
- window length `W`
- EWMA decay `lambda`
- variable subset or feature family

### Initial grid recommendation
Candidate lag `tau`:
- 0 h, 2 h, 4 h, 6 h, 8 h

Window length `W`:
- 10 min, 20 min, 1 h, 2 h, 4 h, 6 h, 8 h

Decay `lambda`:
- 0.60, 0.75, 0.85, 0.92, 0.97

### Feature families
At minimum compare:
1. simple short-window statistics
2. EWMA distilled features
3. multi-scale distilled features

### Example feature components
For each variable and each `(tau, W)`:
- EWMA value
- mean
- std
- min
- max
- range
- slope or linear trend
- recent change
- stability indicator

### Deliverables
- Feature generation module.
- Distillation experiment matrix.
- Comparison table versus current window features.

### Validation target
Show whether distilled sample construction improves alarm-related metrics relative to current window features, especially for the future-head. If the current-head short-window baseline remains stronger, keep the simpler design there.

---

## Stage 3. Multi-scale and multi-lag feature integration

### Objective
Test whether one laboratory result should be represented jointly by multiple time scales and multiple lag hypotheses.

### Motivation
One laboratory result may be influenced simultaneously by:
- short-term recent state,
- medium-term accumulated effects,
- longer-cycle process drift.

### Tasks
Construct combined feature sets using:
- short-horizon distilled features
- medium-horizon distilled features
- long-horizon distilled features
- multiple lag anchors

### Example configurations
- short only
- medium only
- long only
- short + medium
- medium + long
- short + medium + long
- single lag vs. multi-lag

### Deliverables
- Multi-scale feature ablation results.
- Single-lag vs. multi-lag comparison results.

### Validation target
Determine whether multi-scale representation materially improves alarm detection.

---

## Stage 4. Alarm-oriented label redesign

### Objective
Redesign brittle hard labels into target forms that are more suitable for alarming.

### Candidate target designs

#### T1: Current-head three-class probability target
- below_spec
- in_spec
- above_spec

#### T2: Future-head binary probability target
- probability that out-of-spec will occur within a future horizon

#### T3: Two independent current alarm probabilities
- probability of low-limit violation
- probability of high-limit violation

#### T4: Soft labels around threshold neighborhoods
Instead of strict boundaries, use smoother supervision near thresholds.

#### T5: Confidence-weighted labels
Reduce the training influence of samples that are very close to thresholds.

### Recommended implementation sequence
1. Lock the current-head three-class probability target as the current baseline.
2. Lock the future-head binary probability target as the current future baseline.
3. Implement soft-label or confidence-weighted variants around thresholds.
4. If needed, compare the current three-class target against two independent current alarm probabilities.

### Deliverables
- Clear target definition file.
- Threshold and gray-zone definitions.
- Comparison between hard classification and probability alarming.

### Validation target
Determine whether probability alarming is more robust and more useful than hard class assignment, and whether current-head three-class probability should remain the main representation.

---

## Stage 5. Regime-aware modeling

### Objective
Avoid distillation and unified modeling across incompatible operating states.

### Tasks
- Define regime segmentation using available metadata or change-point logic.
- Compare the following:
  - global model on mixed regimes
  - global model with regime features
  - regime-specific models
- Try to avoid distillation windows crossing major regime transitions.

### Possible regime indicators
- load bands
- product grade / mode
- key setpoints
- operating mode
- campaign or operating phase
- startup / shutdown / transient markers

### Deliverables
- Regime segmentation method.
- Regime-specific data summaries.
- Performance comparison between mixed-regime and regime-aware pipelines.

### Validation target
Determine whether regime-aware design improves robustness and reduces false alarms.

---

## Stage 6. Model family comparison

### Objective
After data and label redesign, compare model families under the same experimental setup.

### Candidate models
The first round should remain practical and interpretable:
- Logistic regression / ridge classifier
- Gradient boosting
- Random forest (optional)
- PLS / PLS-DA if suitable
- Calibrated classifiers
- Simple ordinal models or two-head models

Before simple models clearly saturate, deep models should not be prioritized.
LightGBM or deeper models should remain optional and should only be introduced after the data-side redesign is clearly justified.

### Rules
- Use the same train/test splits.
- Use identical features within the same experiment group to ensure fair comparison.
- Separate feature ablation from model ablation.

### Deliverables
- Model comparison table.
- Performance-versus-complexity summary.
- Recommended shortlist of candidate models.

### Validation target
After correcting data construction, identify the best model family.

---

## Stage 7. Alarm post-processing and decision logic

### Objective
Evaluate whether post-processing improves business usefulness.

### Candidate methods
- probability smoothing over time
- hysteresis
- minimum persistence rule
- separate entry and exit thresholds
- event-level consolidation

### Deliverables
- Alarm logic comparison report.
- Point-level and event-level evaluation results.

### Validation target
Find a decision layer that improves alarm stability without introducing unacceptable delay.

---

## 9. Validation Framework

## 9.1 Data splitting strategy

Strict time-ordered splitting should be used.

Recommended options:
- rolling-origin validation
- expanding window validation
- blocked time-series cross-validation

Random shuffle split should not be used, unless temporal order truly does not need to be preserved.

### Required split outputs
- training period
- test period
- optional multiple folds

### Additional requirement
If the process exhibits seasonal or campaign structure, the test set should include realistic operating variation.

---

## 9.2 Core evaluation metrics

Because the business objective is alarming, alarm-related metrics should take priority over generic accuracy.

### Primary metrics
For the current-head:
- macro F1
- balanced accuracy
- class-wise recall for:
  - below_spec
  - in_spec
  - above_spec
- confusion matrix

For the future-head:
- recall for out-of-spec events
- precision for out-of-spec alarms
- F1 for alarmed class or event
- PR-AUC for rare alarms
- false alarm rate
- missed alarm rate
- balanced accuracy

### Probability quality metrics
- Brier score
- calibration error
- reliability plots
- multiclass log loss where applicable

### Event-level metrics
- event detection recall
- event detection precision
- average detection delay
- early warning lead time, if applicable

### Secondary metrics
- overall accuracy
- macro F1
- confusion matrix
- MAE/RMSE can be added if regression pipelines are also tested

---

## 9.3 Decision criteria

A candidate pipeline should not be selected on a single metric alone.

A recommended pipeline should satisfy most of the following:
- improved rare-event recall relative to baseline
- acceptable false alarm rate
- stable performance across multiple time folds
- reasonable calibration when probabilities are output
- interpretable behavior, or at least behavior that is operationally diagnosable
- manageable implementation complexity

---

## 10. Experiment Matrix

Codex should implement experiments in the following order.

### Group A: Diagnostic experiments
- A1: data audit
- A2: lag sensitivity study
- A3: class imbalance and threshold-neighborhood analysis

### Group B: Feature construction experiments
- B1: current window-statistics baseline
- B2: EWMA distilled features
- B3: multi-scale distilled features
- B4: single-lag vs. multi-lag

### Group C: Target redesign experiments
- C1: current-head three-class probability
- C2: future-head binary probability
- C3: two independent current alarm probabilities
- C4: soft labels
- C5: confidence-weighted training

### Group D: Regime-aware experiments
- D1: mixed-regime global model
- D2: global model + regime features
- D3: regime-specific models
- D4: distillation constrained within regime

### Group E: Model comparison experiments
- E1: logistic / ridge
- E2: gradient boosting
- E3: PLS family, if suitable
- E4: calibrated versions of the better models

### Group F: Alarm-logic experiments
- F1: raw probability threshold alarming
- F2: time smoothing
- F3: hysteresis
- F4: persistence rule

---

## 11. Minimum Output Requirements

Each experiment run should produce at least:

- experiment ID
- dataset version
- target definition
- lag / window / decay parameters
- feature family description
- model configuration
- split definition
- validation metrics
- test metrics
- confusion matrix or event summary table
- saved predictions
- summary notes

A final comparison table should aggregate the major experiment runs.

---

## 12. Recommended Directory Structure

```text
v3/dev/
  experiments/
  baselines/
  artifacts/
  docs/
```

Suggested artifacts:
- YAML or JSON configuration files
- experiment registry in CSV/Parquet
- metrics summary table
- plots for lag sensitivity, PR curves, calibration curves, and alarm timelines

---

## 13. Key Risks and Checks

### Risk 1: Temporal leakage
Check whether every laboratory sample uses only data at or before the allowed causal cutoff.

### Risk 2: Regime mixing
Check whether distillation windows cross major operating-state transitions.

### Risk 3: Label instability near thresholds
Where possible, inspect repeated laboratory tests or neighboring samples to quantify boundary noise.

### Risk 4: Severe class imbalance
Use PR-oriented metrics and event-level analysis, not accuracy alone.

### Risk 5: Overfitting to one time period
Use multiple time folds and compare stability.

### Risk 6: Excessive feature dimensionality
Keep feature generation modular and track feature dimensionality continuously.

---

## 14. Priority Recommendation

If resources are limited, execute in the following order:

1. Stage 0: data audit
2. Stage 4: current-head three-class probability baseline and re-search
3. Stage 1: future-head lag diagnosis
4. Stage 2: EWMA distilled features for the future-head
5. Stage 4: future-head binary probability modeling
6. Stage 3: multi-scale / multi-lag integration
7. Stage 5: regime-aware experiments
8. Stage 7: alarm decision logic
9. Stage 6: broader model comparison

Rationale:
The largest expected gains are more likely to come from better sample construction and target definition, rather than from immediately switching to more complex models. Current evidence also suggests that the current-head and future-head should not be forced into the same task definition or feature construction path.

---

## 15. Final Conclusions Required in This Phase

By the end of this phase, Codex should support recommendations on the following:

1. What the best feature-construction strategy is.
2. What the most suitable lag / window / decay search scope is.
3. Whether multi-scale features are necessary.
4. Whether the current-head should remain a three-class probability design.
5. Whether the future-head should remain a binary probability design.
6. Whether regime-aware training is worth its added complexity.
7. Which candidate pipeline should be implemented next.

---

## 16. Execution Instructions for Codex

Codex should treat this document as an execution blueprint.

When details are ambiguous, follow these principles:
- preserve causality;
- avoid leakage;
- prefer simpler and auditable designs first;
- log all assumptions explicitly;
- separate data-construction effects from model effects;
- optimize for alarm usefulness rather than nominal accuracy.

End of document.
