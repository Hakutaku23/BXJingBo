# T90 Prior Package

`prior/` stores curated prior knowledge distilled from the completed V1/V2 work under `projects/T90/dev/`.

Its role is to help V3 start with explicit hypotheses and rejected paths, without copying old code into the new workspace.

## How To Use

- Treat the files here as prior knowledge, not as implementation templates.
- Re-check every prior against the V3 objective before reusing it.
- Prefer re-running a focused V3 experiment over blindly inheriting an old conclusion.

## What Is Included

- `experiment_registry.csv`
  Summary table of major experiment families, questions, best observed result, and whether the result should influence V3.
- `label_policy.yaml`
  Current label and target definitions used in V1/V2 validation.
- `feature_and_window_priors.yaml`
  Curated priors about window size, PH usage, stage structure, and current alert-model preference.
- `method_priors.md`
  Short narrative summary of what worked, what did not, and why.
- `artifact_index.csv`
  Index of the most important legacy artifacts worth consulting.

## What Is Intentionally Not Included

- Raw historical data copies
- Full experiment logs
- Delivery code snapshots
- Temporary scripts that only make sense inside the old V1/V2 workflow

## Current High-Value Priors

- For recommendation, the strongest single DCS window candidate was `50 min`.
- `60 min` slightly improved calcium point error in some experiments, but `50 min` was the better balanced recommendation window.
- PH looked useful only as an optional, stage-conditional enhancement.
- PH lags above `4 h` should not be treated as a formal deployment rule.
- For out-of-spec alerting, `single_logistic_oos` was the most stable probability source among the methods tested.
- Temporal alarm persistence reduced signal chatter, but did not solve the false-alarm problem by itself.

## Important Constraint On Reuse

Some legacy conclusions were built on top of DCS point sets that were originally screened under an earlier `15 min` framing.

That means:

- window conclusions are still useful as priors, but not final truth;
- legacy representative point sets should not be treated as fixed V3 inputs;
- any V3 feature-selection or stage-structure work should be revalidated from the underlying data rather than copied from the old pipeline.

## Recommended V3 Attitude

Start from first principles, but keep these prior conclusions visible:

1. `50 min DCS` is a strong baseline, not a final truth.
2. Same-sample out-of-spec classification was informative, but not fully satisfying as an industrial alert task.
3. Event-level or future-window warning is a reasonable V3 direction.
