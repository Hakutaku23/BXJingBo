# AGENTS.md — T90 V3 Workspace Rules

This file applies to all work performed inside `projects/T90/v3/`.

## 1. Role of V3
- `v3/` is the active third-version workspace.
- It is the default location for new architecture, new experiments, new baselines, and future delivery-candidate development.
- `v3/` is not an extension of the frozen root `dev/`.
- `v3/` should be able to mature into a new delivery candidate as a self-contained workspace.

## 2. V3 Product Objective
V3 targets a **joint objective of early warning plus recommendation**.

The full intended V3 scope is:
- T90 out-of-spec early warning;
- calcium recommendation.

However, V3 development must be staged:
- **Phase 1:** focus only on out-of-spec early warning;
- **Phase 2:** develop calcium recommendation only after the warning path is sufficiently validated.

Do not treat recommendation work as the first delivery priority of V3.
The first milestone of V3 is a usable and well-validated out-of-spec warning capability.

## 3. V3 Output Contract
V3 may produce a **two-level operational output** for online use.

The allowed warning levels are:
- `warning`
- `alarm`

These levels must be explicitly defined in configuration, documentation, and evaluation logic.
Do not hard-code ambiguous alert semantics in scattered scripts.
If offline models are trained for different thresholds or policies, the deployed runtime must load the model and policy that correspond to the configured warning definition.

## 4. Target Spec Rule
The current target spec remains:
- `8.45 ± 0.25`
- equivalently, a nominal acceptable window of `8.20 ~ 8.70`

In V3, this target spec must be treated as **configurable**, not as a permanently hard-coded constant.

This means:
- the target center and tolerance should be representable in config;
- label generation and evaluation should follow the configured spec;
- if models are trained offline against a given spec, online deployment must load the model package that matches the configured spec;
- changing the spec must not silently reuse an incompatible trained model.

## 5. Input and Validation Boundary
V3 does **not** introduce a new raw online input source named `incoming_data`.
The raw input boundary for V3 should remain aligned with the existing available sources and deployment assumptions.

However:
- after V3 development is complete, data from `incoming_data` should still be used for offline validation, replay, or backtesting when appropriate;
- such validation usage does not redefine `incoming_data` as a new production input source;
- validation code and production code must keep this distinction explicit.

Do not redesign the V3 runtime interface around `incoming_data` as if it were a newly added live source.

## 6. V3 Delivery Boundary
The V3 delivery boundary is limited to:
- `core/`
- `interface.py`
- `example.py`
- `README.md`

Everything else in `v3/` is development support unless explicitly promoted.

## 7. Directory Responsibilities

### `core/`
- Holds V3 candidate production logic.
- Keep public module boundaries clear and stable.
- Do not allow essential logic to exist only in `dev/`.
- Phase-1 warning logic that is considered delivery-candidate code must eventually live here.
- Phase-2 calcium recommendation logic that is considered delivery-candidate code must also be promoted here.

### `interface.py`
- This is the intended external integration entry point for V3.
- Keep it explicit, minimal, and stable.
- Do not expose research-only arguments, temporary debug switches, or fragile path assumptions.
- The interface should support the warning output contract clearly.
- Calcium-recommendation-related interface fields may remain absent or minimal until Phase 2 begins.

### `example.py`
- Demonstrates the intended V3 usage path.
- Keep it runnable, concise, and synchronized with `interface.py`.
- During Phase 1, the example should primarily demonstrate out-of-spec warning usage.
- If Phase 2 is introduced, the example may extend to calcium recommendation, but only after the warning path is stable.

### `README.md`
- Documents V3 usage, assumptions, structure, outputs, and phase scope.
- Focus on V3 itself.
- Do not let the README become a dump of historical V1/V2 detail.
- It must clearly distinguish current implemented scope from future intended scope.
- It must clearly state that the current recommendation scope is calcium only.

### `config/`
- Centralize experiment and runtime configuration when practical.
- Prefer explicit config over scattered hard-coded settings.
- Spec definitions, label policies, warning thresholds, model-package selection rules, calcium recommendation settings, and evaluation settings should be configurable here when appropriate.

### `assets/`
- Store lightweight assets needed by V3 development or packaging.
- Do not use this directory as a large unmanaged dump.

### `dev/`
- Stores V3-only experiments, baselines, temporary analyses, and developer helpers.
- Do not mix V1/V2 historical work into this directory.
- Do not leave critical delivery logic here indefinitely.
- Calcium recommendation exploration may happen here before any promotion into `core/`.

### `dev/experiments/`
- Place controlled V3 experiment scripts and notebooks here.
- Each experiment SHOULD have a clear objective, assumptions, and traceable output path.
- Phase 1 experiments should prioritize label definition, early-warning modeling, thresholding, and robustness checks.
- Phase 2 experiments, if started, should focus on calcium recommendation only.

### `dev/artifacts/`
- Store experiment outputs, metric summaries, reports, and temporary artifacts here.
- Artifacts SHOULD be attributable to named experiments, configs, or dates.
- Do not treat artifacts as delivery assets unless explicitly promoted.

### `dev/baselines/`
- Store baseline pipelines and benchmark comparisons here.
- Keep them reproducible and clearly separate from candidate production logic.
- Phase 1 baselines should focus on out-of-spec warning before calcium recommendation quality.

## 8. Working Rules
- Build V3 from first principles when appropriate.
- Use `projects/T90/prior/` as curated prior knowledge, not as a binding implementation template.
- Reuse historical conclusions only when they still match the V3 objective and assumptions.
- Distinguish clearly between reusable prior, tentative hypothesis, and deprecated path.
- Prefer small, testable, traceable steps over broad rewrites.
- Keep the Phase 1 / Phase 2 boundary explicit in code, configs, and documentation.
- Do not let unfinished calcium recommendation work blur the warning-first milestone.

## 9. Promotion Rules
- Promote from `dev/` to `core/` only when the logic is stable enough to support a delivery candidate.
- Warning capability should reach a stable interface and evaluation standard before calcium recommendation logic is promoted as delivery-candidate code.
- If a V3 conclusion becomes reusable prior knowledge, distill it into `projects/T90/prior/` in curated form.
- Do not silently promote V3 code into the root stable line.

## 10. Dependency Rules
- Keep the V3 runtime path as small and explicit as possible.
- Prefer dependencies that remain compatible with CPU-only deployment.
- Development-only, benchmarking-only, and export-only dependencies should stay out of runtime requirements unless truly required for delivery.

## 11. Validation Rules
Before considering V3 work complete, check the following when relevant:
- `interface.py`, `example.py`, and `README.md` are synchronized;
- no essential logic exists only in `dev/`;
- experiments are traceable to configs or explicit assumptions;
- artifacts can be tied back to named experiments;
- the default execution path remains CPU-usable;
- configured spec, warning policy, and loaded model package are mutually consistent;
- offline validation clearly distinguishes production inputs from validation-only inputs such as `incoming_data`.

## 12. Deployment Constraint
V3 must remain compatible with CPU-only factory environments.
- Do not default to GPU-specific backends.
- Any optional acceleration path must be strictly non-default.
- If a dependency complicates CPU-only delivery, prefer the simpler alternative unless the benefit is clear and necessary.
