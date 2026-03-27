# AGENTS.md — T90 Subproject Governance

This file applies to all work performed inside `projects/T90/`. It refines any broader repository-level instructions for this subproject.

## 1. Scope
- Treat `projects/T90/` as a self-contained subproject.
- Do not modify sibling subprojects unless the task explicitly requires cross-project coordination.
- All work inside `projects/T90/` MUST respect the separation between the stable line, the prior-knowledge line, and the V3 line.

## 2. Project Line Definition
`projects/T90/` contains three lines with different responsibilities and change policies.

### 2.1 Stable line at the current root
The current root-level delivery line consists of:
- `core/`
- `interface.py`
- `example.py`
- `README.md`

This line is the current stable delivery candidate.

Rules:
- Treat it as frozen by default.
- Only make changes here when the task explicitly targets the current stable delivery path.
- Do not use this line as the default place for new architecture, new experiments, or broad redesign.

### 2.2 Historical research line
The historical research line is:
- `dev/`

This line contains V1/V2 development history.

Rules:
- Treat it as frozen research history by default.
- Use it for audit, review, extraction, and summarization of prior attempts.
- Do not continue new V3 experiments in this directory.
- Do not turn it into an active mixed workspace again.

### 2.3 V3 incubation line
The new incubation line is:
- `v3/`

This is the default workspace for the third-version development effort.

Rules:
- All new architecture, new experimental design, new baselines, and future delivery-candidate work SHOULD go into `v3/` by default.
- `v3/` MUST remain self-contained.
- V3 work MUST NOT depend on informal reuse of root `dev/` scripts as if they were part of the delivery boundary.

## 3. Delivery Boundary

### 3.1 Current stable delivery boundary
The current stable delivery boundary is limited to:
- `core/`
- `interface.py`
- `example.py`
- `README.md`

Everything else at the root level is non-delivery material unless explicitly promoted.

### 3.2 V3 delivery boundary
The intended V3 delivery boundary is limited to:
- `v3/core/`
- `v3/interface.py`
- `v3/example.py`
- `v3/README.md`

Everything else under `v3/` is development support unless explicitly promoted.

## 4. Default Working Rule
- If the task is about maintaining the current shipped or near-shipped path, work in the current root stable line.
- If the task is about a new approach, replacement design, exploratory implementation, or next-generation delivery, work in `v3/`.
- If the task is about historical understanding, review root `dev/` and distill reusable conclusions into `prior/` when appropriate.
- Do not place new V3 development into root `dev/`.
- Do not silently promote V3 code into the stable root line.

## 5. Directory Responsibilities

### 5.1 Root stable line

#### `core/`
- Holds the current stable production logic.
- Keep module boundaries clear.
- Prefer targeted maintenance, defect fixes, and compatibility fixes.
- Do not use it as the default landing zone for experimental redesign.

#### `interface.py`
- This is the current external integration entry point.
- Preserve backward compatibility unless the task explicitly allows a breaking change.
- Keep the interface explicit, minimal, and stable.
- Do not expose research-only switches, unstable assumptions, or developer-only path conventions.

#### `example.py`
- Demonstrates the supported stable usage path.
- Keep it runnable, concise, and synchronized with the real `interface.py` contract.
- Prefer deterministic and minimal examples over exploratory demos.

#### `README.md`
- Documents stable usage, runtime requirements, input expectations, and outputs.
- Focus on delivery-facing documentation.
- Do not turn it into a research notebook replacement.

### 5.2 Historical research line

#### `dev/`
- Stores V1/V2 research history.
- Use it for historical audit, trace-back, and extraction of prior conclusions.
- Do not place new V3 experiments here.
- Do not allow critical production logic to exist only here.

### 5.3 Prior-knowledge line

#### `prior/`
- Stores curated prior knowledge extracted from historical work.
- `prior/` MUST contain distilled conclusions, not bulk experimental dumps.
- Do not copy old research folders into `prior/` wholesale.
- Every file in `prior/` SHOULD remain interpretable without replaying the original experiment tree.

#### `prior/README.md`
- Explains what prior knowledge exists and how it should be used.
- Keep it short, navigational, and decision-oriented.

#### `prior/experiment_registry.csv`
- Records key historical attempts in compact form.
- Each retained row SHOULD capture at least:
  - objective,
  - method,
  - main inputs,
  - main conclusion,
  - whether the direction should continue.
- This file is a knowledge index, not an artifact inventory.

#### `prior/label_policy.yaml`
- Defines the authoritative label policy for T90-related judgment.
- Keep the meanings of in-spec, out-of-spec, overspec, warning, and alarm explicit.
- Do not allow different experiments to silently redefine labels without updating this file.

#### `prior/feature_notes.yaml`
- Records feature-related prior knowledge.
- Capture signals tried, windows tried, transformations tried, and stage- or context-specific conclusions.
- Record PH conclusions explicitly, including when PH helps, when it does not help, and under what setup the observation was made.
- Prefer reusable design knowledge over run-specific details.

#### `prior/references/`
- Holds small curated reference notes or supporting documents.
- Keep contents compact and decision-oriented.
- Do not use it as a large artifact archive.

### 5.4 V3 line

#### `v3/`
- This is the default active workspace for the third version.
- It MUST remain structurally independent from the frozen root `dev/` workflow.
- The long-term expectation is that `v3/` can become a new delivery candidate without inheriting historical directory noise.

#### `v3/core/`
- Holds V3 candidate production logic.
- Keep public module boundaries clear.
- Do not allow essential V3 logic to live only inside `v3/dev/`.

#### `v3/interface.py`
- This is the intended future external entry point for V3.
- Keep it explicit, stable, and minimal.
- Do not leak research-only switches or fragile path assumptions into the public interface.

#### `v3/example.py`
- Demonstrates the intended V3 usage path.
- Keep it concise, runnable, and synchronized with `v3/interface.py`.

#### `v3/README.md`
- Documents V3 usage, assumptions, structure, and outputs.
- Focus on V3 itself.
- Mention historical context only when it helps explain a design decision.

#### `v3/config/`
- Centralizes V3 runtime and experiment configuration when practical.
- Prefer explicit config over scattered hard-coded settings.

#### `v3/assets/`
- Stores lightweight assets needed by V3 development or packaging.
- Do not turn it into an unmanaged dump of large files or transient experiment outputs.

#### `v3/dev/`
- Stores V3-only experimental work.
- This directory MUST NOT mix root `dev/` historical work with V3 work.
- Use it for temporary studies, baselines, ablations, exploration, and developer-only helpers.

#### `v3/dev/experiments/`
- Place controlled V3 experiment scripts and notebooks here.
- Each experiment SHOULD have a clear goal and a traceable output path.

#### `v3/dev/artifacts/`
- Store V3 experiment outputs, reports, metric summaries, and temporary artifacts here.
- Artifacts SHOULD be attributable to explicit experiments, configs, or dates.
- Do not treat artifacts as delivery assets unless explicitly promoted.

#### `v3/dev/baselines/`
- Store baseline models and comparison pipelines here.
- Keep them reproducible and clearly separate from candidate production logic.

## 6. Engineering Rules
- Prefer small, targeted, and reversible changes.
- Keep production-worthy logic in `core/` or `v3/core/`, not only in `dev/`.
- Avoid broad framework-heavy abstractions unless they clearly reduce long-term maintenance cost.
- Avoid hard-coded path coupling between `v3/` and root historical artifacts.
- When refactoring a delivery boundary, update its paired `interface.py`, `example.py`, and `README.md` in the same change when necessary.

## 7. Knowledge Promotion Rules
- Promotion from root `dev/` into `prior/` MUST be curated.
- Promotion from `v3/dev/` into `v3/core/` means the logic is stable enough to support a delivery candidate.
- Promotion from `v3/` into the root stable line MUST be explicit and user-directed.
- When summarizing prior conclusions, distinguish clearly between:
  - confirmed reusable prior,
  - tentative observation,
  - deprecated path,
  - V3-only working hypothesis.

## 8. Dependency Rules
- Keep runtime dependencies minimal and justified by the delivery boundary.
- Do not expand stable-line runtime requirements merely for V3 convenience.
- V3 may use extra development dependencies when justified, but unnecessary heavyweight additions should still be avoided.
- Development, benchmarking, conversion, and export dependencies should remain outside runtime requirements unless they are truly required for delivery.

## 9. Validation Rules
Before considering work complete, check the following when relevant:
- the current stable `interface.py`, `example.py`, and `README.md` are still aligned;
- the V3 `interface.py`, `example.py`, and `README.md` are aligned when V3 is the target;
- no new V3 experiment was placed into the frozen root `dev/`;
- no bulk historical dump was copied into `prior/`;
- no critical production logic exists only inside a `dev/` directory;
- imports resolve from the intended working directory.

## 10. Change Discipline
- Do not silently redefine the project structure.
- Do not silently broaden the delivery boundary.
- Do not invent build, test, packaging, or export commands if they do not yet exist.
- When commands are missing, propose conservative commands and label them as suggested rather than established.
- When a task output spans stable line, prior line, and V3 line, state explicitly which files belong to which line.

## 11. Output Expectations
When producing code changes, implementation plans, or restructuring proposals for this subproject:
- state whether the work targets the stable line, the prior line, or the V3 line;
- preserve the stable root delivery boundary unless explicitly asked to replace it;
- keep `prior/` curated and compact;
- keep `v3/` self-contained;
- prefer solutions maintainable by a single developer;
- avoid process-heavy assumptions unless explicitly requested.

## 12. Deployment Constraint
This subproject must remain usable in a CPU-only factory environment.

Hard requirements:
- the current stable `interface.py` must run correctly on CPU-only machines;
- the current stable `example.py` must demonstrate CPU-only usage;
- the stable `README.md` must document CPU-only setup and execution;
- if V3 becomes a delivery candidate, `v3/interface.py`, `v3/example.py`, and `v3/README.md` must also support CPU-only execution;
- do not default to `cuda`, CUDAExecutionProvider, or any GPU-specific backend;
- any optional acceleration path must be strictly non-default and must not affect the CPU delivery path.
