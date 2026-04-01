# Cleanroom Experiment Plan — Distributional Interval Prediction with Reject Option

## Purpose

This cleanroom experiment is designed to answer one narrow question:

Under the current T90 problem setting, is a distributional interval prediction approach with an explicit reject / retest option more appropriate than the current threshold-oriented ordinal / cumulative classification line for boundary-sensitive current-head decisions?

This plan intentionally treats **predictive uncertainty and decision abstention** as the next validation target.  
It assumes that the project has already learned the following from prior cleanroom work:

- hard three-class classification is too brittle near boundaries;
- ordinal / cumulative probability modeling is directionally better but still not satisfactory;
- forcing every sample into a final class still re-hardens ambiguous boundary cases;
- simple-window features remain the strongest stable feature baseline so far.

This phase does **not** treat EWMA, PH augmentation, or future-head prediction as the primary method under test.

---

## Why this should come after ordinal / cumulative validation

The previous cleanroom stage established that task reformulation matters.  
However, it also showed a recurring limitation:

- boundary-friendly formulations improved some aspects of decision behavior,
- but the model was still required to collapse each sample into a final class,
- which reintroduced hard decisions on samples that may genuinely be ambiguous.

Under the current project conditions:

- many samples cluster near 8.2 and 8.7;
- laboratory measurements contain assay noise;
- neighboring values near thresholds may not deserve fully different hard labels;
- the business goal is alarm usefulness, not forced point certainty.

Because of this, the next cleanroom question should be:

> Should the model move from threshold classification toward interval-distribution prediction, and should the decision layer explicitly allow abstain / retest outcomes for uncertain boundary cases?

This phase therefore shifts the modeling target from:
- hard class prediction
or
- cumulative threshold probabilities alone

toward:
- interval probability distribution over ordered T90 bins,
- followed by an explicit decision policy with a reject / retest state.

---

## Experiment boundary

This phase validates only one method family:

- distributional interval prediction with reject option

This phase should not include:

- paper-faithful EWMA as the main treatment
- PH lag-aware augmentation as the main treatment
- multiscale weighted fusion
- future-head prediction
- deep models
- end-to-end complex sequence architectures
- stage-aware branch splitting as the main line

These may become later cleanroom branches, but they are out of scope for this first validation.

---

## Core hypothesis

If the remaining project difficulty is driven more by **irreducible boundary ambiguity** than by insufficient model capacity, then a distributional formulation with reject option should:

- reduce high-confidence hard mistakes near boundaries;
- preserve or improve core acceptable-zone discrimination;
- provide a more truthful representation of uncertainty around 8.2 and 8.7;
- better support operational decisions such as accept / warning / unacceptable / retest.

---

## Task definition

- Task: current-head decision modeling
- Label source: current laboratory sample
- Goal: determine whether interval-distribution prediction plus reject option is a better decision formulation than current ordinal / cumulative collapse

### Reference baseline to keep
The frozen reference line from the previous cleanroom stage is:

- simple 120min causal window statistics
- fold-local topk sensor screening
- sensor-identity de-dup
- monotonic ordinal / cumulative probabilities
- inner-threshold symmetric hard weighting on 8.2 / 8.7 only

This is the comparison anchor, not the thing being rewritten.

### New cleanroom treatment task
The new treatment predicts a probability distribution over ordered T90 intervals.

Recommended first-bin scheme:

- `bin_1: y < 8.0`
- `bin_2: 8.0 <= y < 8.2`
- `bin_3: 8.2 <= y < 8.7`
- `bin_4: 8.7 <= y < 8.9`
- `bin_5: y >= 8.9`

These bins may later be revised if business rules change, but they must remain fixed throughout one cleanroom run.

---

## Label design

This phase should not use one-hot targets as the primary treatment target.

Instead, use **soft distribution labels** over the ordered bins.

### Principle
If a sample lies near a threshold, its supervision should not be concentrated entirely in one bin.

Examples:
- a sample close to 8.2 may place probability mass on both `bin_2` and `bin_3`
- a sample close to 8.7 may place probability mass on both `bin_3` and `bin_4`

### Recommended first implementation
Use a simple local distribution rule around the observed T90 value.

Allowed initial approaches:
- triangular soft labels over neighboring bins
- Gaussian-like kernel over bin centers
- small-radius piecewise linear mass redistribution around thresholds

The exact rule must be fixed before comparison and documented in the audit.

### Hard constraint
The soft-label rule must depend only on the observed current T90 value and fixed bin definitions.  
It must not be tuned per fold using test outcomes.

---

## Method families to compare

This phase should compare only the following formulations.

### Baseline A — Frozen cleanroom ordinal / cumulative reference
Use the strongest currently frozen reference line as-is.

### Baseline B — Distributional prediction without reject option
Predict the 5-bin interval probability distribution, but always collapse to one final business decision:
- acceptable
- warning
- unacceptable

This baseline isolates whether distributional supervision alone already helps.

### Treatment C — Distributional prediction with reject / retest option
Predict the same 5-bin interval distribution, then allow a fourth operational state:
- acceptable
- warning
- unacceptable
- retest

This is the main treatment.

---

## Recommended modeling style

The priority is auditability, not sophistication.

### Option 1: Direct multinomial distribution model over 5 bins
Train a simple probabilistic classifier on the 5-bin target.

Allowed:
- multinomial logistic
- balanced random forest probabilities

If soft targets are not directly supported by the chosen implementation, approximate with one of:
- sample replication with weighted bin targets
- custom cross-entropy if already lightweight and auditable
- label-smoothing style training with fixed target vectors

### Option 2: Conditional distribution via adjacent or cumulative reconstruction
If the codebase already supports a simple ordered formulation that reconstructs interval probabilities cleanly, it may be used.

However, the first cleanroom implementation should prefer the simpler auditable route.

---

## Reject / retest decision layer

This is the key new component.

The model must be allowed to abstain on uncertain samples.

### Required operational outputs
At minimum produce:
- `P(bin_1)`
- `P(bin_2)`
- `P(bin_3)`
- `P(bin_4)`
- `P(bin_5)`

Then derive:
- `acceptable_prob = P(bin_3)`
- `warning_prob = P(bin_2) + P(bin_4)`
- `unacceptable_prob = P(bin_1) + P(bin_5)`

### New decision state
Add:
- `retest`

### Recommended first reject policy
Use a simple documented rule based on uncertainty.

For example, retest if any of the following hold:
- top class probability is below a fixed confidence threshold
- central mass is spread across both sides of a key threshold
- entropy is above a fixed threshold
- acceptable and unacceptable probabilities are both non-trivial
- warning mass is high but not dominant enough for direct warning

The exact rule must be fixed before model comparison.

### Hard constraint
Do not optimize the reject rule on the outer test folds.

Reject thresholds may be chosen:
- from the training fold only,
- or fixed a priori.

---

## Feature policy

This phase is still about output formulation first, not about advanced feature engineering.

### Allowed feature family for the first run
Use only the current strongest simple and auditable baseline feature family:

- simple 120min causal window statistics

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
Changing both uncertainty formulation and feature representation at the same time would destroy attribution.  
This phase must isolate whether moving from threshold classification to distributional prediction + reject is itself beneficial.

---

## Feature screening policy

Keep the same cleanroom rules.

### Stage 1: unsupervised pre-cleaning
Allowed on the full raw sensor pool:
- remove constant variables
- remove nearly constant frozen variables
- remove duplicate columns
- remove variables with extreme missingness

No label information may be used here.

### Stage 2: supervised sensor screening
Allowed only inside each training fold:
- univariate ranking
- fixed top-k sensor selection
- or fixed score-threshold selection

Rules:
- screening must be fit on the training fold only
- the held-out fold must not influence screening
- all compared formulations inside the same fold must use the same screened sensor set

### Recommended top-k
Keep the current strong reference:
- `topk_sensors = 40`

A top-k sensitivity check may come later, but is not part of the first run.

---

## Models

Keep the model family simple to isolate formulation effects.

Allowed model family:
- multinomial logistic
- balanced random forest

Recommended first-pass comparison:
- multinomial logistic for 5-bin distribution prediction
- multinomial logistic for collapsed baseline where needed
- balanced random forest as a secondary robustness family

Do not introduce new complex model classes in this phase.

---

## Decision time and causality

- `decision_time` is the current laboratory sample time
- all features must be causal
- no future information may be used
- use the same 120min causal window as the frozen reference line for the first run

This phase is not trying to optimize lag/window.  
It is trying to test whether a distributional target plus reject option is superior under the strongest currently known simple representation.

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

This phase must evaluate both **decision quality** and **uncertainty usefulness**.

### Primary decision metrics
For non-rejected predictions:
- macro_f1
- balanced_accuracy
- warning average precision
- unacceptable average precision

### Coverage / abstention metrics
Must report:
- reject rate
- decision coverage
- macro_f1 at covered samples only
- unacceptable miss rate among non-rejected samples
- boundary-neighborhood reject rate

### Distribution-quality metrics
Must report:
- multiclass Brier score over the 5 bins
- calibration error over the 5 bins if feasible
- negative log loss if feasible
- entropy statistics

### Boundary-focused diagnostics
At minimum report:
- `7.9–8.3`
- `8.6–8.8`

For each boundary subset, report:
- reject rate
- high-confidence non-warning rate
- unacceptable false-clear rate
- acceptable confidence behavior

---

## Business decision mapping

The business layer must be fixed before comparison.

### Recommended first mapping
From the 5-bin distribution:
- acceptable = `P(bin_3)`
- warning = `P(bin_2) + P(bin_4)`
- unacceptable = `P(bin_1) + P(bin_5)`

Then apply reject rule first:
- if reject condition holds -> `retest`
- else output the largest among acceptable / warning / unacceptable

### Why this is required
This keeps the decision layer:
- simple
- interpretable
- comparable to the existing cleanroom reference

---

## Success criteria

Distributional prediction with reject option is considered supported only if most of the following hold:

- it lowers high-confidence hard mistakes near the boundaries;
- it produces a useful reject set concentrated in ambiguous samples rather than easy samples;
- covered-sample macro_f1 is not worse than the frozen ordinal / cumulative reference;
- unacceptable miss risk among non-rejected samples is not worse, and ideally better;
- calibration / Brier behavior improves or at least remains acceptable;
- gains are not confined to one lucky fold.

If these conditions are not met, the conclusion should be that the added uncertainty layer did not yet justify replacing the current frozen cleanroom reference.

---

## What this phase must not conclude

This phase must not claim:
- that EWMA is useless;
- that PH is useless;
- that future-head is useless;
- that ordinal / cumulative was a mistake;
- that fuzzy ideas are unnecessary.

It may conclude only whether:
- interval-distribution prediction is more appropriate than threshold-only classification,
- and whether an explicit reject / retest option improves decision quality under the current simple feature baseline.

---

## Required outputs

- `distributional_reject_results.csv`
- `distributional_reject_summary.json`
- `distributional_reject_feature_rows.csv`
- `distributional_reject_audit.md`

### The audit note must state
- what bin definitions were used
- what soft-label rule was used
- whether the distribution target was direct or reconstructed
- what reject rule was used
- how reject thresholds were chosen
- whether supervised screening was fold-internal only
- what historical assumptions were intentionally ignored

---

## Recommended first deliverables inside the cleanroom directory

- `plans/distributional_reject_option_validation.md`
- `configs/distributional_reject_option_current_head.yaml`
- `scripts/run_distributional_reject_option_current_head.py`
- `reports/distributional_reject_option_current_head_audit.md`

---

## Next phase, only if this phase succeeds

If the distributional + reject line succeeds, then the next cleanroom experiment may compare:

- distributional + reject + simple 120min stats
versus
- distributional + reject + paper-faithful EWMA
or
- distributional + reject + PH lag-aware augmentation

That next phase would then isolate whether feature-side augmentation adds value after the output formulation has already improved.

---

## Naming recommendation

Keep using the directory:

`projects/T90/dev/cleanroom_validation/`

This remains the correct location, because the cleanroom principle still applies even though the primary method under validation has changed.
