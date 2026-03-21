# AGENTS.md

This file guides Codex and other coding agents when working in this repository.

Scope:
- Applies to the repository root `BXJingBo/` and all descendants.
- If a deeper `AGENTS.md` exists inside a subproject, the closer file overrides this one for that subtree.

## Deployment constraint

All deliverables intended for the manufacturer must be CPU-only.
Do not require NVIDIA GPUs, CUDA, TensorRT, or any other GPU-specific runtime as a prerequisite for build, test, inference, or deployment.

When proposing or modifying code:
- prefer CPU-compatible implementations by default;
- do not introduce GPU-only dependencies into runtime requirements;
- do not make CUDA or GPU execution the default or assumed path;
- keep examples, interface entrypoints, and documentation runnable in a CPU-only environment.

## 1. Repository nature

This is a multi-project repository maintained by a single developer for research, implementation, and final interface delivery. It is **not** a large team product repository.

Work in this repo with a clear separation between:
- research and development material
- final deliverables

Do not introduce heavyweight process, unnecessary infrastructure, large-scale restructuring, or team-oriented conventions unless explicitly requested.

## 2. Top-level structure

Typical layout:

```text
BXJingBo/
├─ projects/
│  ├─ T90/
│  │  ├─ core/            # core algorithm package; may later be optimized/compiled with Cython
│  │  ├─ data/            # local private data, ignored by .gitignore
│  │  ├─ data_example/    # minimal public example data
│  │  ├─ dev/             # research, validation, and development scripts
│  │  ├─ interface.py     # primary external entry point for delivery
│  │  ├─ example.py       # minimal usage example
│  │  └─ README.md        # project usage and interface documentation
│  └─ ...
├─ requirements/
│  ├─ dev.txt
│  └─ runtime.txt
├─ scripts/
├─ .gitattributes
├─ .gitignore
├─ IDB_requirements.txt   # dependency versions available in the target environment
└─ README.md
```

## 3. Delivery boundary

The repository itself is not the deliverable. Each subproject is delivered separately.

For a subproject such as `projects/T90/`, the default delivery set is:
- compiled deliverable form of `core/` at delivery time
- `interface.py`
- `example.py`
- `README.md`

The following are normally **not** deliverables:
- `dev/`
- `data/`
- ad hoc test scripts
- intermediate research artifacts
- files unrelated to the target subproject

Always protect the delivery boundary. Do not make the formal interface depend on `dev/`, `data/`, private local files, or developer-only setup.

## 4. Scope rules

### 4.1 Work only in the requested subproject

If the task is about `projects/T90/`, work only in that subproject and directly relevant shared dependencies.

Do not:
- modify other sibling subprojects casually
- do broad repo-wide refactors for style consistency
- extract project-specific logic into global utilities unless reuse is explicitly required

### 4.2 Keep subprojects independent

Each subproject should remain as independently deliverable as possible.

Do not create unnecessary coupling such as:
- one subproject depending on another subproject's internal implementation
- a formal interface depending on another subproject's `dev/` scripts
- root-level temporary tools becoming required runtime prerequisites

### 4.3 Do not rename delivery files without explicit instruction

Do not rename or relocate `core/`, `interface.py`, `example.py`, `README.md`, or documented input/output paths unless the task explicitly requires it.

## 5. Responsibilities by location

### 5.1 `core/`

`core/` contains the core algorithm, main data-processing flow, and stable business logic.

Requirements:
- Keep important logic in `core/`, not scattered across scripts.
- Prefer clear, testable, stable implementations.
- Keep module boundaries and public entry points explicit.

For future Cython optimization/compilation:
- avoid unnecessary runtime metaprogramming, monkey patching, and excessive reflection
- avoid deeply interactive or notebook-style side effects in core logic
- avoid fragile runtime import tricks and hidden initialization
- separate public API from internal implementation where practical

Important:
- Do **not** move the only working Python implementation into `dev/`.
- If Cython experiments are needed, keep experimental code clearly separated, but preserve a maintainable canonical implementation path.

### 5.2 `interface.py`

`interface.py` is the primary external entry point of a subproject.

Requirements:
- keep signatures as stable as possible
- make parameters, return values, and exception behavior explicit
- do not expose temporary research knobs as formal API unless necessary
- do not require callers to understand `dev/` or internal repository layout

Design preference:
- prefer explicit inputs and structured outputs
- when the existing project style uses dictionaries for interface input/output, follow that style consistently
- do not break an existing public API only for stylistic cleanup

### 5.3 `example.py`

`example.py` is the minimal runnable usage example, not an arbitrary script.

Requirements:
- keep it aligned with the current `interface.py`
- prefer data from `data_example/` when example input files are needed
- make it short and directly useful for external callers
- do not disguise research scripts as examples

### 5.4 `README.md`

A subproject `README.md` is part of the delivery package.

Requirements:
- update it when interface, parameters, paths, inputs/outputs, or run instructions change
- focus on usage, I/O expectations, dependencies, and caveats
- avoid mixing long research notes into delivery documentation

### 5.5 `dev/`

`dev/` is for research, validation, debugging, and development scripts.

Requirements:
- experimental code is allowed
- formal delivery must not depend on `dev/`
- mature logic should be promoted into `core/` or the interface layer instead of living permanently in `dev/`

### 5.6 `data/`

`data/` is private local data and is ignored by `.gitignore`.

Rules:
- assume external users do not have it
- do not hardcode formal interfaces against private files in `data/`
- do not commit large raw data, caches, exports, or sensitive samples
- `interface.py` must not require `data/` to function along the delivery path

### 5.7 `data_example/`

`data_example/` contains minimal public, reproducible example data.

Rules:
- prefer it for `example.py` and validation that should be reproducible
- keep samples small but complete enough to validate the documented flow
- add or update examples when interface inputs change materially

## 6. Paths and environment assumptions

- Use robust relative path handling.
- Do not hardcode personal absolute paths.
- Do not assume a specific IDE, shell, or launch directory.
- Do not bake one-off local environment setup into the formal interface.

## 7. Dependency management

Repository-level dependency files usually include:
- `requirements/dev.txt` for development, research, testing, and build helpers
- `requirements/runtime.txt` for the minimal runtime dependencies needed by delivered interfaces

Rules:
- classify each added dependency before updating requirements
- keep runtime dependencies minimal and stable
- do not add heavyweight packages for small convenience features
- prefer the standard library or existing dependencies when reasonable
- if version alignment matters for delivery, check `IDB_requirements.txt`
- if a package is only needed for research, testing, or compilation support, do not place it in `runtime.txt` unless runtime truly requires it

## 8. Commands, verification, and non-invention rule

A good agent should verify work, but should not invent project commands.

Rules:
- prefer commands already documented in subproject `README.md`, `scripts/`, or existing project files
- if no reliable command is documented, do not fabricate a build/test/lint workflow
- state clearly what was verified and what could only be checked statically

When possible, verify at least:
1. `example.py` remains consistent with `interface.py`
2. the interface can reach core functionality without depending on `dev/`
3. documentation matches the code
4. the change does not accidentally depend on `data/`

## 9. Change priorities

Default priority order:
1. keep the interface usable and the delivery boundary clear
2. keep mature logic in `core/`
3. keep `example.py` and `README.md` synchronized
4. only then improve research scripts or cosmetic style

Do not leave partial alignment:
- if the interface changes, review `example.py` and `README.md`
- if core logic is extracted, remove stale duplicate logic from `interface.py`
- if dependencies change, update the correct requirements file

## 10. Output style

This repository is maintained by one developer. Prefer complete, direct, practical changes.

When delivering work:
- provide a complete implementation when the task asks for code
- avoid large unrelated edits
- keep the summary concise
- explain:
  - which files changed
  - why they changed
  - whether the public interface changed
  - how the result was verified

## 11. Do-not rules

Unless explicitly requested, do not:
- introduce Docker, databases, queues, microservices, or other heavy infrastructure
- redesign the entire repository layout
- over-generalize a single-project problem into an unnecessary framework
- add complex CI/CD setup
- add unrelated frontend, backend, or network capabilities
- sacrifice deliverability for the appearance of being more engineered
- turn experimental scripts into formal interfaces
- let README/example drift away from the actual interface

## 12. Default decision rule

When unsure where code belongs:
- formal, reusable, deliverable capability -> `core/` or `interface.py`
- temporary validation, data analysis, debugging, or experiments -> `dev/`
- external usage demonstration -> `example.py`
- external explanation -> `README.md`
- anything that only works with private local data -> not part of the delivery path

## 13. Direct instructions to Codex

When working in this repository:
- identify the exact target subproject first
- modify the minimum necessary file set
- protect the formal interface and delivery boundary
- move mature logic toward `core/`
- check whether `example.py` and `README.md` need updates when interfaces change
- classify dependency changes into `dev.txt` or `runtime.txt`
- avoid hardcoding local paths, private data, or one-off developer workflows into formal interfaces
- prefer concise final explanations over long process narratives

## 14. Default optimization target

If the task is underspecified, optimize for:
- easier independent delivery of a single subproject
- clearer and more stable interfaces
- more core logic concentrated in `core/`
- sharper separation between research code and delivery code
- better maintainability without over-engineering
