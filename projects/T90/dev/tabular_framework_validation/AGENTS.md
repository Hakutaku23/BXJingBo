# AGENTS.md — Tabular Framework Validation Rules for T90

This file applies to work inside `projects/T90/dev/tabular_framework_validation/` and refines the parent `projects/T90/AGENTS.md`.

## Scope
- This directory is reserved for validating strong tabular frameworks such as AutoGluon and TabPFN on the current project.
- The workflow here is explicitly two-stage:
  1. quick validation from uncleaned-source data with minimal necessary preprocessing
  2. framework-specific feature engineering and data cleaning, only if stage 1 is promising
- The default purpose is framework validation, not direct productionization.

## Required dataset statement
Every experiment and report in this directory must explicitly state:

- the starting data source is currently **uncleaned**
- “uncleaned” refers to the source state, not to a ban on all preprocessing
- stage 1 may use only minimal necessary preprocessing
- stage 2 may introduce dedicated feature engineering only if stage 1 has demonstrated value

## Primary objective
- Determine whether AutoGluon and TabPFN are worth keeping as project-level tabular benchmarks.
- If they are, determine how to design a framework-specific feature-engineering workflow for them.
- Keep the workflow auditable and leakage-controlled.

## Allowed reuse
You may reuse:
- raw / near-raw data loaders
- current current-head label utilities
- simple causal snapshot feature builders
- offline reference metadata such as `projects/T90/data/卤化位点.xlsx` for signal interpretation

You should avoid reusing as fixed truth:
- prior branch-specific selected sensor lists
- prior handcrafted feature bundles
- prior best-threshold logic
- prior best-window logic

## Directory expectations
Recommended contents under this directory:
- `plans/`
- `configs/`
- `scripts/`
- `reports/`
- `artifacts/`

## Experiment discipline
- Preserve causality and avoid future leakage.
- Use time-ordered validation only.
- Clearly separate:
  - stage 1 quick validation
  - stage 2 dedicated feature engineering
- Record what was deliberately left uncleaned in stage 1.
- Record what was newly engineered or cleaned in stage 2.

## Baseline rules
- Every benchmark must define a simple baseline.
- Baseline and framework runs must use the same data split.
- Stage 2 is not allowed unless stage 1 has shown a credible positive signal.

## Output rules
Each experiment should leave behind:
- a clear plan,
- runnable configs or scripts,
- a machine-readable result table,
- and an audit note explicitly describing:
  - starting source condition,
  - stage 1 processing,
  - stage 2 processing if applicable.

## Promotion rule
Nothing from this directory should be treated as delivery code by default.
Promotion into `core/`, `interface.py`, `example.py`, or `README.md` requires an explicit user decision.
