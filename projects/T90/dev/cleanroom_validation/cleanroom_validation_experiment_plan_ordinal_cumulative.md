# Cleanroom Experiment Plan — Threshold-Oriented Ordinal / Cumulative Probability Validation

## Purpose

This cleanroom experiment is designed to answer one narrow question:

Under the current T90 problem setting, is threshold-oriented ordinal / cumulative probability modeling more appropriate than the existing hard three-class formulation for current-head alarm-related decisions?

This plan intentionally treats **task formulation** as the first validation target.  
It does **not** yet treat EWMA data distillation as the primary method under test.

---

## Why this should come before EWMA validation

The current project difficulty is dominated by label-boundary and decision-structure problems:

- many samples lie near the specification boundaries;
- laboratory measurements contain assay noise;
- hard boundaries such as 8.2 and 8.7 can make neighboring samples receive unstable labels;
- the business goal is closer to alarming / acceptability judgment than to pure point prediction.

Because of this, the first cleanroom question should be:

> Should the task continue to be modeled as a hard flat three-class problem, or should it be reformulated as an ordered threshold-probability problem?

EWMA mainly addresses:
- multi-rate sampling,
- timeline mismatch,
- and lag-aware condensed representation.

It does not directly solve poor boundary handling.  
Therefore, EWMA should be postponed until after the task formulation itself is validated.

---

## Experiment boundary

This phase validates only one method family:

- threshold-oriented ordinal / cumulative probability modeling

This phase should not include:

- paper-faithful EWMA as the main treatment
- multiscale weighted fusion
- future-head prediction
- fuzzy regression as the main line
- deep models
- alarm post-processing
- multi-head engineering extensions

These may become later cleanroom branches, but they are out of scope for this first validation.

---

## Task definition

- Task: current-head decision modeling
- Label source: current laboratory sample
- Goal: determine whether ordered threshold probability modeling is a better task definition than hard three-class classification

### Current hard three-class baseline
The current baseline task is assumed to be:

- `below_spec`
- `in_spec`
- `above_spec`

### New cleanroom treatment task
The cleanroom treatment reformulates the problem into ordered threshold events.

Recommended threshold set for the first version:

- `y < 8.0`
- `y < 8.2`
- `y < 8.7`
- `y < 8.9`

These thresholds can later be revised if business rules require it, but they must remain fixed throughout one cleanroom run.

---

## Core hypothesis

If the main problem is boundary instability rather than model capacity, then a threshold-oriented ordinal / cumulative formulation should:

- produce more stable probabilities near boundaries;
- better separate truly unacceptable material from borderline-but-acceptable material;
- better reflect business decisions than a flat hard three-class classifier.

---

## Data policy

### Allowed
- broad raw / near-raw DCS variable pool
- current LIMS / current-head label source
- offline DCS metadata reference files for interpretation

### Not allowed as default inherited assumptions
- previous selected sensor lists
- previous ranked top features
- previous locked lag/window conclusions
- previous experiment summaries as fixed truth
- previous model recommendations as fixed truth

---

## Cleanroom starting rule

This experiment should intentionally restart from minimal inherited assumptions.

That means:

- use a broad candidate sensor pool;
- allow fresh feature screening;
- reuse trusted data access and target construction utilities where appropriate;
- do not inherit prior handcrafted feature packages as mandatory inputs.

---

## Method families to compare

This phase should compare only the following three formulations.

### Baseline A — Hard three-class classification
A standard current-head hard three-class model:

- below_spec
- in_spec
- above_spec

This is the current control group.

### Baseline B — Optional nested manual two-stage classification
This is optional and only included if desired as a diagnostic baseline.

Example structure:
- first classify into coarse groups such as `<8.2`, `8.2–8.7`, `>8.7`
- then subdivide the outer groups into more detailed acceptability thresholds

This baseline is optional because it is mainly useful for comparing against the user's original intuition, not because it is the preferred final formulation.

### Treatment C — Ordinal / cumulative probability modeling
This is the main treatment.

Instead of directly predicting one flat class, model cumulative threshold probabilities such as:

- `P(y < 8.0)`
- `P(y < 8.2)`
- `P(y < 8.7)`
- `P(y < 8.9)`

Then derive interval probabilities from cumulative probabilities.

Example:
- `P(y < 8.0)`
- `P(8.0 ≤ y < 8.2) = P(y < 8.2) - P(y < 8.0)`
- `P(8.2 ≤ y < 8.7) = P(y < 8.7) - P(y < 8.2)`
- `P(8.7 ≤ y < 8.9) = P(y < 8.9) - P(y < 8.7)`
- `P(y ≥ 8.9) = 1 - P(y < 8.9)`

A downstream business rule may then collapse these interval probabilities back into:
- acceptable
- warning
- unacceptable

but that collapse must happen only after the cumulative probabilities are produced.

---

## Recommended implementation style

Use one of the following simple and auditable approaches.

### Option 1: Independent threshold probability models with monotonicity audit
Train one binary probability model per threshold event:

- `y < 8.0`
- `y < 8.2`
- `y < 8.7`
- `y < 8.9`

Then:
- audit monotonicity,
- record violations such as `P(y < 8.0) > P(y < 8.2)`,
- and optionally apply simple monotonic correction in post-processing if explicitly documented.

### Option 2: Shared ordinal formulation
If the codebase supports a simple ordinal formulation without adding heavy dependencies, it may be used.

However, the priority is not elegance.  
The priority is to obtain an auditable threshold-probability experiment quickly and cleanly.

Therefore, Option 1 is acceptable as the first cleanroom implementation, as long as monotonic consistency is explicitly checked and reported.

---

## Feature policy

This phase is about task formulation first, not about advanced feature engineering.

### Allowed feature family for the first cleanroom run
Use only simple, auditable causal window statistics.

Examples:
- mean
- std
- min
- max
- last
- range
- delta

No EWMA in this phase.

### Why no EWMA yet
If both task formulation and feature representation are changed at the same time, it becomes hard to attribute the result.  
This phase must isolate the effect of reformulating the prediction target.

---

## Feature screening policy

Because the historical feature subset may have been unsuitable, cleanroom re-screening is allowed.

### Stage 1: unsupervised pre-cleaning
Allowed on the full raw sensor pool:
- remove constant variables
- remove nearly constant frozen variables
- remove duplicate columns
- remove variables with extreme missingness

No label information may be used here.

### Stage 2: supervised sensor screening
Allowed only inside each training fold:
- univariate ranking by correlation / AUC / mutual information
- fixed top-k sensor selection
- or fixed score-threshold selection

Rules:
- screening must be fit on the training fold only
- the held-out fold must not influence screening
- all compared formulations inside the same fold must use the same screened sensor set

### Recommended top-k options
If top-k screening is used, keep it small and fixed:
- 20
- 40
- 80

Record `topk_sensors` as an explicit experiment axis.

---

## Models

Keep the model family simple to isolate task-definition effects.

Allowed model family:
- logistic-style classifiers
- random forest balanced

Recommended minimum comparison:
- multinomial logistic for the hard three-class baseline
- binary logistic for each cumulative threshold model
- random forest balanced as a secondary family

Do not introduce new complex model classes in this phase.

---

## Decision time and causality

- `decision_time` is the sample's current laboratory time
- all features must be causal
- no future information may be used
- this phase may use simple fixed causal windows
- lag-search may be minimal, but should not become the main experiment axis yet

### Recommended fixed first-pass window
For the first cleanroom ordinal run, choose one conservative causal window family and keep it fixed.

For example:
- 60 minutes
- 120 minutes
- 240 minutes

This phase is not trying to optimize lag/window exhaustively.  
It is trying to test whether the threshold-oriented formulation is superior under a reasonable and auditable feature setup.

---

## Validation

Use time-ordered evaluation only.

Recommended:
- `TimeSeriesSplit(n_splits=5)` by default
- if sample count is too small, allow 4 folds but not fewer than 3

No random shuffle split.

If operating regime boundaries are detectable:
- record whether a sample window crosses a regime boundary
- report subset behavior on non-crossing samples where possible

---

## Metrics

Because the purpose is alarm-oriented threshold handling, metrics should focus on both discrimination and calibration.

### Primary metrics
- macro_f1 on the collapsed business decision output
- balanced_accuracy on the collapsed business decision output
- PR-AUC for rare unacceptable outcomes
- average precision for key unacceptable thresholds

### Threshold-model-specific metrics
- Brier score for each threshold probability
- calibration error for each threshold probability
- monotonicity violation rate across cumulative thresholds

### Boundary-focused diagnostics
At minimum report performance on boundary-neighborhood subsets, such as:
- `7.9–8.3`
- `8.6–8.8`

These ranges may be adjusted, but boundary-neighborhood diagnostics are mandatory.

### Recommended business-level collapse
After cumulative probabilities are produced, define a simple decision collapse, for example:

- unacceptable if `P(y < 8.0)` is above threshold, or `P(y ≥ 8.9)` is above threshold
- warning if boundary-neighborhood intervals dominate
- acceptable otherwise

The exact collapse rule must be fixed and documented before comparing models.

---

## Success criteria

Threshold-oriented ordinal / cumulative modeling is considered supported only if most of the following hold:

- it improves macro_f1 over hard three-class baseline;
- it improves balanced_accuracy over hard three-class baseline;
- it improves boundary-neighborhood stability;
- it shows better calibration near operational thresholds;
- gains are not confined to a single lucky fold;
- at least one simple model family behaves stably.

If these conditions are not met, the conclusion should be that task reformulation did not yet justify replacing the hard three-class baseline.

---

## What this phase must not conclude

This phase must not claim:
- that EWMA is useless;
- that fuzzy methods are useless;
- that future-head tasks are useless;
- that no better feature representation exists.

It may conclude only whether ordinal / cumulative reformulation appears more appropriate than hard three-class classification under a simple cleanroom setup.

---

## Required outputs

- `ordinal_cumulative_results.csv`
- `ordinal_cumulative_summary.json`
- `ordinal_cumulative_feature_rows.csv`
- `ordinal_cumulative_audit.md`

### The audit note must state
- what thresholds were used
- whether cumulative probabilities were modeled independently or jointly
- whether monotonicity was checked
- whether monotonic correction was applied
- whether supervised screening was fold-internal only
- what historical assumptions were intentionally ignored

---

## Recommended first deliverables inside the cleanroom directory

- `plans/ordinal_cumulative_validation.md`
- `configs/ordinal_cumulative_current_head.yaml`
- `scripts/run_ordinal_cumulative_current_head.py`
- `reports/ordinal_cumulative_current_head_audit.md`

---

## Next phase, only if this phase succeeds

If ordinal / cumulative validation succeeds, then the next cleanroom experiment may compare:

- ordinal / cumulative + simple window statistics
versus
- ordinal / cumulative + paper-faithful EWMA distilled features

That second experiment would then isolate the additional value of EWMA after task formulation has already been improved.

---

## Naming recommendation

Keep using the directory:

`projects/T90/dev/cleanroom_validation/`

This remains the correct location, because the cleanroom principle still applies even though the primary method under validation has changed.

