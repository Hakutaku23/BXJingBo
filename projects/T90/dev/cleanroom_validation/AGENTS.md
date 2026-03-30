# AGENTS.md — Cleanroom Validation Rules for T90

This file applies to work inside `projects/T90/dev/cleanroom_validation/` and refines the parent `projects/T90/AGENTS.md`.

## Scope
- This directory is reserved for cleanroom experiments that intentionally restart from a broad feature pool and minimal inherited assumptions.
- The default purpose is transfer-validation of methods, not incremental polishing of prior experiment branches.
- Unless the user explicitly requests comparison, do not inherit:
  - historical selected sensor lists,
  - recommended window sizes,
  - recommended lag settings,
  - tuned model hyperparameters,
  - or prior experiment conclusions as fixed constraints.

## Primary objective
- Determine whether a method is genuinely useful under the current project difficulty.
- Prefer clear falsifiable experiment design over layered engineering complexity.
- Keep the question narrow before expanding scope.

## Allowed reuse
You may reuse:
- existing raw / near-raw data loaders,
- current target-label construction logic when the task still concerns the same current-head target,
- offline reference metadata such as `projects/T90/data/卤化位点.xlsx` for DCS tag interpretation,
- conservative utility code that does not embed prior feature-engineering conclusions.

You should avoid reusing:
- previous experiment summaries as default truth,
- previous feature-selection outputs,
- prior handcrafted feature bundles,
- prior weighted fusion designs,
- or prior branch-specific experiment assumptions.

## Directory expectations
Recommended contents under this directory:
- `plans/` — experiment plans, method notes, acceptance criteria
- `configs/` — experiment-specific configs
- `scripts/` — runnable experiment scripts
- `reports/` — summaries, audits, result interpretation
- `artifacts/` — generated tables, metrics, plots, exported rows

## Experiment discipline
- Start from a broad candidate feature pool unless the user explicitly narrows it.
- Distinguish clearly between:
  - unsupervised feature pre-cleaning,
  - fold-internal supervised feature screening,
  - feature construction,
  - model fitting,
  - evaluation.
- Do not mix these stages in ways that make leakage hard to audit.
- For time-dependent tasks, preserve causality and avoid future leakage.
- If operating-regime boundaries are detectable, record whether feature windows cross them.
- When validating a literature method, implement the paper-faithful version first. Hybrid improvements belong to later phases.

## Baseline rules
- Every cleanroom experiment must define an explicit baseline.
- The baseline may be simpler than prior project baselines if the purpose is method isolation.
- Baseline and treatment must share the same data split and, when relevant, the same screened sensor set inside each fold.

## Output rules
Each experiment should leave behind:
- a clear plan,
- a runnable script or config,
- a machine-readable result table,
- and a brief audit note stating what was and was not allowed.

## Promotion rule
Nothing from this directory should be treated as delivery code by default.
Promotion into `core/`, `interface.py`, `example.py`, or `README.md` requires an explicit user decision.
