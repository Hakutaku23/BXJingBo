# AGENTS.md — T90 Subproject Rules

This file applies to work performed inside `projects/T90/` and overrides or refines broader repository-level instructions.

## Scope
- This subproject is an independently delivered package inside the larger `BXJingBo` repository.
- Treat `projects/T90/` as a self-contained deliverable unless a task explicitly requires cross-project coordination.
- Do not modify sibling subprojects unless the user explicitly asks for cross-project alignment.

## Delivery Boundary
The expected delivered contents for this subproject are limited to:
- `core/` — the core algorithm package
- `interface.py` — the external interface entry point
- `example.py` — invocation example for the external interface
- `README.md` — subproject usage and delivery documentation

Everything else should be treated as development support, research material, internal tooling, or local-only assets unless explicitly promoted into the delivery boundary.

## Directory Responsibilities
### `core/`
- Keep production logic here.
- Prefer clear module boundaries and stable public APIs.
- Avoid binding external callers directly to internal helper modules when `interface.py` can provide a cleaner boundary.
- Maintain code structure that can later be optimized, cythonized, or compiled for delivery.
- Avoid unnecessary runtime coupling to notebooks, ad hoc scripts, or debug-only utilities.

### `interface.py`
- This is the primary external integration entry point.
- Keep it stable, explicit, and minimal.
- Expose a clean calling contract suitable for downstream integration.
- Avoid leaking research-only arguments, temporary debug switches, or internal path assumptions into the public interface.

### `example.py`
- Demonstrate the intended external usage path.
- Keep the example runnable, concise, and aligned with the current `interface.py` contract.
- Prefer deterministic and minimal examples over exploratory demos.

### `README.md`
- Document how to install runtime dependencies, prepare inputs, call the interface, and interpret outputs.
- Focus on externally useful usage documentation rather than internal research notes.
- Keep all commands and examples synchronized with the current code.

### `dev/`
- Use for research scripts, exploratory validation, conversion utilities, temporary benchmarking, and developer helpers.
- Code here is not part of the delivery contract unless explicitly promoted.
- Do not let `dev/` become the only place where critical production logic exists.

### `data/`
- Treat as local/private working data.
- Do not rely on committed large datasets under this directory.
- Do not document workflows that require shipping `data/` as part of final delivery.

### `data_example/`
- Use for small, safe, demonstrative example inputs or outputs.
- Keep samples minimal and suitable for documentation, smoke tests, and external demonstrations.

## Data Directory Semantics

### `data/`
- Contains data that has already been organized, transformed, or cleaned through the legacy MATLAB-based preprocessing workflow.
- Treat this directory as processed working data, not raw source data.


Specific reference under `data/`:
- `projects/T90/data/卤化位点.xlsx` is the factory-provided authoritative reference for DCS tag/point descriptions in the halogenation area.
- When performing DCS tag interpretation, feature naming, signal screening, or signal grouping/classification, prefer this file as the primary source of truth.
- This file is reference metadata / offline supporting material, not a new runtime input source.
- Conclusions distilled from this file may be promoted into `prior/feature_notes.yaml`, `v3/config/`, or small supporting documentation, but the original spreadsheet must not become a hard runtime dependency.

### `data_example/`
- Contains small, development-friendly example data used for demonstration, debugging, and reproducible examples.
- Prefer files here for `example.py` and README usage examples.
- Do not assume this directory represents the newest real-world input batch.

### `incoming_data/`
- Contains newly received raw or near-raw data for testing, validation, and preprocessing migration work.
- Treat this directory as the source side when reproducing or replacing the MATLAB preprocessing workflow in Python.
- Do not assume files here are already cleaned, standardized, or ready for direct delivery use.

## Engineering Rules
- Prefer small, targeted edits over broad rewrites.
- Preserve backward compatibility of `interface.py` unless the task explicitly allows breaking changes.
- When refactoring `core/`, keep the public interface stable or update `example.py` and `README.md` in the same change.
- Do not introduce framework-heavy abstractions unless there is a clear maintenance benefit.
- Avoid repo-wide configuration churn when the task is only about `T90`.

## Dependency Rules
- Runtime requirements must remain minimal and directly justified by the delivered interface.
- Development, testing, benchmarking, export, or conversion dependencies belong in development dependency definitions rather than runtime requirements unless they are truly required at delivery time.
- Do not add heavyweight dependencies for convenience without a clear reason.

## Cython / Compilation Readiness
- Write code in `core/` so it remains straightforward to optimize or compile later.
- Prefer explicit data flow, clear typing opportunities, and simple module boundaries.
- Do not move the only maintainable source implementation into `dev/`.
- Avoid patterns that make later compilation significantly harder unless the user explicitly prefers them.

## Validation Rules
Before considering work complete, check the following when relevant:
- Imports still resolve from the intended working directory.
- `interface.py` remains the main supported entry point.
- `example.py` still reflects the real invocation path.
- `README.md` matches the current behavior.
- No change accidentally depends on private local data under `data/`.

## Change Discipline
- If the task concerns research, keep the result in `dev/` unless it has been clearly promoted into delivery code.
- If the task concerns delivery, prefer changes in `core/`, `interface.py`, `example.py`, and `README.md`.
- Do not silently expand project scope.
- Do not invent build, test, export, or packaging commands if the repository does not already define them.
- When commands are missing, propose conservative commands and label them as suggested rather than established.

## Output Expectations
When producing code changes or implementation plans for this subproject:
- state which files belong to delivery output versus development support,
- preserve the delivery boundary,
- prefer solutions that remain maintainable by a single developer,
- avoid team-process assumptions such as mandatory CI, issue templates, or multi-owner review flows unless explicitly requested.

## Project-specific deployment constraint

This subproject must remain fully usable in a CPU-only factory environment.

Hard requirements:
- `interface.py` must run correctly on CPU-only machines;
- `example.py` must demonstrate CPU-only usage;
- `README.md` must document CPU-only setup and execution;
- `requirements/runtime.txt` must not include GPU-only packages;
- do not default to `cuda`, CUDAExecutionProvider, or any GPU-specific backend;
- any optional acceleration path must be strictly non-default and must not affect the CPU delivery path.

## Additional directory responsibility for cleanroom validation

### `dev/cleanroom_validation/`
- Use this directory for cleanroom experiments that intentionally restart analysis from scratch.
- The goal here is method-transfer validation under minimal prior assumptions, not incremental optimization of previous experiment branches.
- Work in this directory should, by default, ignore prior feature-engineering conclusions, locked sensor subsets, tuned window choices, and historical experiment recommendations unless the task explicitly asks for comparison against them.
- Prefer starting from the broad available feature pool, basic data quality filtering, and clearly stated causal assumptions.
- Reuse of offline reference metadata is allowed when it improves interpretability or signal naming consistency. In particular, `projects/T90/data/卤化位点.xlsx` remains the authoritative offline reference for DCS tag interpretation and feature naming.
- Outputs here are development support only, not delivery-boundary artifacts, unless the user explicitly promotes them.
- Do not let cleanroom experiment logic silently replace production logic in `core/`, `interface.py`, `example.py`, or `README.md` without explicit promotion.

## Additional cleanroom experiment rules
- Cleanroom experiments should prioritize isolating whether a method is intrinsically useful on this project, rather than preserving compatibility with earlier experiment scaffolding.
- When starting a cleanroom experiment, explicitly document:
  - the target task,
  - the allowed data sources,
  - whether historical feature selection is ignored,
  - the baseline used for comparison,
  - and the leakage controls used in evaluation.
- Keep cleanroom experiment code, configs, notes, and reports inside `dev/cleanroom_validation/` unless the user explicitly asks otherwise.
- If a cleanroom experiment needs feature screening, prefer:
  - unsupervised pre-filtering on the full raw feature pool,
  - followed by supervised screening strictly inside each training fold.
- Do not perform full-dataset supervised feature selection before time-series validation.
- If a cleanroom experiment is meant to validate a literature method, prefer a paper-faithful implementation first, and postpone hybrid extensions until after the first conclusion is documented.

