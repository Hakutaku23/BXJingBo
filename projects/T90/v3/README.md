# T90 V3 Workspace

`v3/` is the clean third-version workspace for the T90 project.

It is intentionally separated from the current stable line under `projects/T90/` so that new task definitions, new preprocessing logic, and new model design can be explored without creating confusion about what is already deliverable.

## Positioning

- The root `projects/T90/` line remains the current stable line.
- `projects/T90/dev/` remains the historical V1/V2 research area.
- `projects/T90/prior/` stores curated prior knowledge from the historical work.
- `projects/T90/v3/` is where new design should start from first principles.

## V3 Goal

V3 should answer a cleaner version of the T90 problem than V1/V2 did.

The current V3 Phase 1 target is now fixed as:

- same-sample out-of-spec warning

Meaning:

- given the current DCS operating context,
- determine whether the current sample is likely to fall outside the configured T90 spec window.

Recommendation remains part of the full V3 scope, but it is intentionally deferred to Phase 2.

## Current Structure

- `AGENTS.md`
  V3-specific working rules.
- `config/`
  Future runtime or experiment configuration.
- `assets/`
  Future lightweight V3 assets.
- `core/`
  Candidate V3 production logic once it stabilizes.
- `interface.py`
  Intended future V3 external entry point.
- `example.py`
  Minimal usage example for the V3 interface.
- `dev/`
  V3-only experiments, baselines, and artifacts.

## Current State

This workspace has entered Phase 1 bootstrap.

- A reusable V3 data contract now lives in `core/data_contract.py`.
- The first Phase 1 config lives in `config/phase1_warning.yaml`.
- The first audit script lives in `dev/experiments/phase1_warning_data_audit.py`.
- `interface.py` still returns a structured placeholder response.
- `example.py` still demonstrates the intended invocation shape only.

## Working Principle

Use `projects/T90/prior/` to understand what has already been tried, then design V3 around the new objective rather than around legacy implementation inertia.
