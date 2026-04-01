# AGENTS.md — Centered Quality Validation Rules for T90

This file applies to work inside `projects/T90/dev/centered_quality_validation/` and refines the parent `projects/T90/AGENTS.md`.

## Scope
- This directory is reserved for cleanroom-style experiments that validate whether a centered-quality objective is more appropriate than threshold-only task definitions.
- The default purpose is target-definition validation, not incremental polishing of ordinal/cumulative or conformal/reject branches.
- Unless the user explicitly requests comparison, do not inherit:
  - historical selected sensor lists as fixed truth,
  - prior weighted fusion structures,
  - prior EWMA conclusions as fixed constraints,
  - or prior business mappings as mandatory logic.

## Primary objective
- Determine whether a risk-constrained centered-quality target better matches the true process objective.
- Prefer a falsifiable, auditable experiment design over engineering complexity.
- Keep the branch narrow before expanding scope.

## Allowed reuse
You may reuse:
- existing raw / near-raw data loaders,
- current current-head label utilities,
- sensor-identity de-dup logic,
- conservative cleanroom feature utilities,
- offline reference metadata such as `projects/T90/data/卤化位点.xlsx` for DCS tag interpretation.

You should avoid reusing:
- prior frozen decision rules as mandatory outputs,
- previous feature-selection results as fixed truth,
- prior handcrafted feature bundles,
- previous branch conclusions as constraints on this new target.

## Directory expectations
Recommended contents under this directory:
- `plans/`
- `configs/`
- `scripts/`
- `reports/`
- `artifacts/`

## Experiment discipline
- Start from a broad candidate feature pool unless the user explicitly narrows it.
- Distinguish clearly between:
  - unsupervised feature pre-cleaning,
  - fold-internal supervised sensor screening,
  - feature construction,
  - target construction,
  - model fitting,
  - decision mapping,
  - evaluation.
- Do not mix these stages in ways that make leakage hard to audit.
- Preserve causality and avoid future leakage.
- If operating-regime boundaries are detectable, record whether feature windows cross them.
- When validating centered quality, keep feature engineering simple first. Do not introduce EWMA until the target definition itself is supported.

## Baseline rules
- Every experiment in this directory must define an explicit threshold-oriented baseline.
- Baseline and treatment must share the same data split and, when relevant, the same screened sensor set inside each fold.
- The main comparison is target definition, not feature family.

## Output rules
Each experiment should leave behind:
- a clear plan,
- a runnable script or config,
- a machine-readable result table,
- and a brief audit note stating what was and was not allowed.

## Promotion rule
Nothing from this directory should be treated as delivery code by default.
Promotion into `core/`, `interface.py`, `example.py`, or `README.md` requires an explicit user decision.
