# AGENTS.md — Volatiles Subproject Rules

This file applies to work performed inside `projects/Volatiles/` and overrides or refines broader repository-level instructions.

## Scope
- This subproject is an independently delivered package inside the larger `BXJingBo` repository.
- Treat `projects/Volatiles/` as a self-contained deliverable unless a task explicitly requires cross-project coordination.
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
- Use this directory for processed working data.
- Do not rely on committed large datasets under this directory.
- Do not document workflows that require shipping `data/` as part of final delivery.

### `data_example/`
- Use this directory for minimal, non-private, reproducible example inputs or sample assets.
- Keep contents small and safe to ship with the subproject.
- Prefer examples that support `example.py` and README usage demonstrations.
- Do not place factory-private or large-scale working data here.

### `incoming_data/`
- Use this directory for newly received raw or near-raw data for testing, validation, and preprocessing migration work.
- Do not assume files here are already cleaned, standardized, or ready for direct delivery use.

## Data Directory Semantics

### `data/`
- Contains data that has already been organized, transformed, or cleaned through the legacy MATLAB-based preprocessing workflow.
- Treat this directory as processed working data, not raw source data.

### `data_example/`
- Contains small, development-friendly example data used for demonstration, debugging, and reproducible examples.
- Prefer files here for `example.py` and README usage examples.
- Do not assume this directory represents the newest real-world input batch.

### `incoming_data/`
- Contains newly received raw or near-raw data for testing, validation, and preprocessing migration work.
- Treat this directory as the source side when reproducing or replacing the MATLAB preprocessing workflow in Python.
- Do not assume files here are already cleaned, standardized, or ready for direct delivery use.

## Data Flow Assumption
Unless the user explicitly states otherwise, assume the intended preprocessing direction is:

`incoming_data/` -> preprocessing / cleaning logic -> `data/`

`data_example/` is for small reproducible examples, not the authoritative source of newly received raw data.

## Engineering Rules
- Prefer small, targeted edits over broad rewrites.
- Preserve backward compatibility of `interface.py` unless the task explicitly allows breaking changes.
- When refactoring `core/`, keep the public interface stable or update `example.py` and `README.md` in the same change.
- Do not introduce framework-heavy abstractions unless there is a clear maintenance benefit.
- Avoid repo-wide configuration churn when the task is only about this subproject.
- In the early stage of this subproject, prefer a minimal working implementation over premature abstraction.

## Dependency Rules
- Runtime requirements must remain minimal and directly justified by the delivered interface.
- Development, testing, benchmarking, export, or conversion dependencies belong in development dependency definitions rather than runtime requirements unless they are truly required at delivery time.
- Do not add heavyweight dependencies for convenience without a clear reason.
- If dependency files are repository-level, update the repository-level dependency definitions instead of creating ad hoc subproject-specific ones unless explicitly requested.

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
- Example usage remains runnable in a CPU-only environment.

## Change Discipline
- If the task concerns research, keep the result in `dev/` unless it has been clearly promoted into delivery code.
- If the task concerns delivery, prefer changes in `core/`, `interface.py`, `example.py`, and `README.md`.
- Do not silently expand project scope.
- Do not invent build, test, export, or packaging commands if the repository does not already define them.
- When commands are missing, propose conservative commands and label them as suggested rather than established.

## Execution Discipline
For non-trivial tasks:
- first restate the goal, affected files, and constraints;
- propose a short plan before making broad edits;
- complete one milestone at a time;
- keep diffs minimal and relevant to the requested task;
- do not add extra features, cleanup work, or refactors unless explicitly requested.

## Output Expectations
When producing code changes or implementation plans for this subproject:
- state which files belong to delivery output versus development support,
- preserve the delivery boundary,
- prefer solutions that remain maintainable by a single developer,
- avoid team-process assumptions such as mandatory CI, issue templates, or multi-owner review flows unless explicitly requested.

## Reference Source Directories
Directories such as `matlab/` or `references/` contain external-language reference implementations.
They are used for algorithm understanding and migration only, not as final delivery artifacts.
Production Python code must be placed in `core/`, then exposed through `interface.py` when appropriate.

## Project-Specific Deployment Constraint
This subproject must remain fully usable in a CPU-only factory environment.

Hard requirements:
- `interface.py` must run correctly on CPU-only machines;
- `example.py` must demonstrate CPU-only usage;
- `README.md` must document CPU-only setup and execution;
- repository-level `requirements/runtime.txt` must not include GPU-only packages for this delivery path;
- do not default to `cuda`, CUDAExecutionProvider, or any GPU-specific backend;
- any optional acceleration path must be strictly non-default and must not affect the CPU delivery path.