# Cleanroom Experiment Plan — Risk-Constrained Centered Quality Validation

## Purpose

This experiment validates a new task formulation for T90:

> Instead of treating all in-spec values as equally good, model quality as a combination of:
> 1. constraint satisfaction risk, and
> 2. centered desirability around the process target (for example 8.45).

The central question is:

Under the current T90 problem setting, does a **risk-constrained centered-quality formulation** better reflect the real process objective than threshold-only current-head classification?

This experiment should be treated as a separate cleanroom branch. It is not a direct continuation of the ordinal/cumulative branch and not a direct continuation of the conformal/reject branch.

---

## Why a separate branch is justified

The previous cleanroom work established that:

- hard three-class classification is too brittle near boundaries;
- ordinal / cumulative modeling is directionally better, but still not fully satisfactory;
- distributional + conformal + reject improves honesty about uncertainty, but the operational tradeoff is still difficult.

A deeper issue now appears to be:

- the business objective is not simply “inside 8.2–8.7 is equally good”;
- in real process control, values closer to the target center (for example 8.45) may be better than values merely inside the admissible range;
- therefore, current formulations may still flatten an inherently graded quality target into a coarse constraint problem.

This motivates a new branch that explicitly models:
- safety / spec compliance,
- and centered quality desirability.

---

## Recommended dedicated directory

Use a dedicated branch directory:

`projects/T90/dev/centered_quality_validation/`

Reason:
- this idea changes the task objective itself;
- it should not be mixed into the existing ordinal/cumulative or distributional/reject branch too early;
- it will likely produce separate plans, configs, scripts, and reports.

---

## Core hypothesis

If the process truly prefers the target center rather than only the legal interval, then a centered-quality objective should:

- distinguish “acceptable but marginal” from “acceptable and high-quality”;
- reduce the artificial flattening of all in-spec samples into one class;
- better align model behavior with real process desirability;
- provide a cleaner basis for later alarm and optimization decisions.

---

## Scope

This phase validates only one method family:

- risk-constrained centered-quality modeling

This phase should not include as primary treatments:

- paper-faithful EWMA
- PH augmentation
- future-head prediction
- multiscale weighted fusion
- deep sequence models
- complex end-to-end policy optimization

These may be evaluated later, only after the task definition itself is validated.

---

## Task definition

### Current business background
The admissible range may be written as:

- target center: `8.45`
- tolerance half-width: `0.25`
- admissible interval: `8.20–8.70`

The project should validate whether this process is better described by:
- a pure threshold task,
or
- a centered desirability task under hard risk constraints.

### New formulation
Split the task into two coupled targets:

#### Target A — Risk / feasibility
Model whether the sample violates or threatens the specification bounds.

Recommended outputs:
- `P(y < 8.2)`
- `P(y > 8.7)`
or equivalent risk-oriented outputs.

#### Target B — Centered quality desirability
Model how desirable the current value is relative to the target center.

Recommended center:
- `target_center = 8.45`

Recommended first desirability score:
- a smooth score that is maximal at `8.45`
- decreases as `|y - 8.45|` grows
- reaches a low value near the specification edges
- and may be clipped or strongly penalized outside spec

---

## Recommended target designs

This branch should compare at least three target designs.

### Baseline A — Frozen threshold-only reference
Use the strongest currently frozen threshold-oriented cleanroom reference as the control anchor.

This is not the optimization target in this branch.  
It is the comparison baseline.

### Treatment B — Centered quality score only
Use one continuous centered-quality score, for example:

#### Piecewise-linear desirability
`q(y) = max(0, 1 - |y - 8.45| / 0.25)`

Interpretation:
- `q(8.45) = 1`
- score decreases linearly toward 0 near the edges
- values outside spec are clipped to 0

Alternative smooth forms are allowed if documented:
- Gaussian-like desirability
- quadratic utility
- bell-shaped desirability score

### Treatment C — Joint risk + centered quality
This is the main treatment.

Model both:
- risk outputs
- centered desirability score

Then derive business decisions from both together.

This should be treated as the main experimental target.

---

## Why not use “premium rate” as the only target

A simple binary “premium / non-premium” label is allowed as a diagnostic baseline, but it should not be the main target.

Reason:
- it still discards distance information;
- it may create a new rare-event problem;
- it is less expressive than a centered desirability score.

Therefore, if premium-style labeling is tested, it should be secondary, not primary.

---

## Centered desirability design

The desirability score must be fixed before comparison.

### Recommended first version
Use the linear score:

`q(y) = max(0, 1 - |y - center| / tolerance_half_width)`

with:
- `center = 8.45`
- `tolerance_half_width = 0.25`

### Optional second version
Use a smooth score:

`q(y) = exp(- (y - center)^2 / (2 * sigma^2))`

If this is used, `sigma` must be fixed before experimentation and documented in the audit.

### Hard constraint
Do not tune the desirability formula on outer test folds.

---

## Risk target design

The first version should stay simple and auditable.

### Recommended first version
Use two binary risk heads:
- lower-spec risk
- upper-spec risk

Equivalent formulations are acceptable, as long as they remain interpretable.

### Why this is useful
The process may have asymmetric risk behavior on the low side and high side.  
Separating the two directions is more informative than one flat “out-of-spec” label.

---

## Business decision mapping

The decision layer must be fixed before model comparison.

### Recommended first business mapping
Use four operational states:

- `unacceptable`
- `acceptable`
- `premium`
- `retest`

### Example mapping logic
1. If lower-risk or upper-risk exceeds a hard threshold -> `unacceptable`
2. Else if desirability score is very high -> `premium`
3. Else if risk is low but desirability is only moderate -> `acceptable`
4. Else if uncertainty is too high or signals disagree -> `retest`

This is only a recommended structure.  
The exact thresholds must be fixed before the cleanroom run and documented.

### Important note
This branch is not trying to replace process control logic.  
It is trying to validate whether a centered-quality objective better matches the process than threshold-only classification.

---

## Feature policy

This phase is about target definition first, not advanced feature engineering.

### Allowed feature family for the first run
Use the strongest current simple baseline representation:

- simple 120min causal window statistics
- sensor-identity de-dup
- fold-local sensor screening

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
Changing both task formulation and feature representation would destroy attribution.  
This branch must isolate the effect of the centered-quality target.

---

## Feature screening policy

Reuse the same cleanroom rules.

### Stage 1: unsupervised pre-cleaning
Allowed on the full sensor pool:
- remove constant variables
- remove nearly constant frozen variables
- remove duplicate columns
- remove variables with extreme missingness

No label information may be used here.

### Stage 2: supervised screening
Allowed only inside each training fold:
- univariate ranking
- fixed top-k sensor selection
- or fixed score-threshold selection

Rules:
- screening must be fit on the training fold only
- the held-out fold must not influence screening
- all compared methods inside the same fold must use the same screened sensor set

### Recommended first-pass top-k
- `topk_sensors = 40`

---

## Models

Keep models simple to isolate target-design effects.

Allowed first-pass models:
- ridge / linear regression for centered desirability
- logistic models for risk heads
- random forest as a secondary robustness model

If a joint multi-output wrapper is easy to audit, it may be used.  
Otherwise, separate simple heads are acceptable for the first implementation.

Do not introduce complex model families in this phase.

---

## Decision time and causality

- `decision_time` is the current laboratory sample time
- all features must be causal
- no future information may be used
- use the same 120min causal window as the strongest current simple baseline

This phase is not trying to optimize lag/window yet.  
It is trying to validate the task objective.

---

## Validation

Use time-ordered evaluation only.

Recommended:
- `TimeSeriesSplit(n_splits=5)` by default
- if sample count is too small, allow 4 folds but not fewer than 3

No random shuffle split.

If operating-regime boundaries are detectable:
- record whether a sample window crosses a regime boundary
- report subset behavior on non-crossing samples where possible

---

## Metrics

This phase must evaluate both:
- threshold-risk behavior
- centered-quality usefulness

### Risk metrics
- lower-risk AP
- upper-risk AP
- unacceptable recall
- unacceptable miss rate

### Centered-quality metrics
- MAE or RMSE on desirability score
- rank correlation between predicted desirability and observed desirability
- premium precision / recall if a premium diagnostic threshold is defined

### Decision metrics
If the branch includes the 4-state decision layer, report:
- macro_f1 over `unacceptable / acceptable / premium / retest`
- balanced_accuracy over covered decisions if retest is active
- premium precision
- unacceptable false-clear rate

### Diagnostic metrics
- distribution of predicted desirability inside the in-spec region
- whether the model truly prefers the center over the margins
- whether samples near 8.45 receive higher predicted quality than samples near 8.2 or 8.7

---

## Success criteria

This branch is considered supported only if most of the following hold:

- the centered-quality target distinguishes center-quality from edge-quality in a stable way;
- the joint risk + desirability formulation better matches process intuition than threshold-only classification;
- premium-like samples can be identified without unacceptable growth in false-clear risk;
- the decision layer remains interpretable and auditable;
- gains are not confined to one lucky fold.

If these conditions are not met, the conclusion should be that the centered-quality objective does not yet justify replacing the current frozen threshold-oriented reference.

---

## What this phase must not conclude

This branch must not claim:
- that ordinal / cumulative was useless;
- that conformal / reject was useless;
- that EWMA is useless;
- that future-head is unnecessary.

It may conclude only whether:
- a centered-quality objective is a better task definition for the real process target;
- and whether joint risk + desirability is more meaningful than threshold-only decision logic.

---

## Required outputs

- `centered_quality_results.csv`
- `centered_quality_summary.json`
- `centered_quality_feature_rows.csv`
- `centered_quality_audit.md`

### The audit note must state
- what target center was used
- what desirability formula was used
- whether premium diagnostics were included
- how risk heads were defined
- what decision mapping was used
- whether supervised screening was fold-internal only
- what historical assumptions were intentionally ignored

---

## Recommended first deliverables inside the new directory

- `plans/centered_quality_validation.md`
- `configs/centered_quality_current_head.yaml`
- `scripts/run_centered_quality_current_head.py`
- `reports/centered_quality_current_head_audit.md`

---

## Next phase, only if this phase succeeds

If the centered-quality branch succeeds, the next experiments may compare:

- centered-quality + simple 120min stats
versus
- centered-quality + paper-faithful EWMA
or
- centered-quality + PH lag-aware augmentation

Only then should feature-side augmentation be evaluated again.

---

## Naming recommendation

Use the dedicated directory:

`projects/T90/dev/centered_quality_validation/`

This is recommended because the task objective itself has changed, and the branch should remain isolated until its value is clear.
