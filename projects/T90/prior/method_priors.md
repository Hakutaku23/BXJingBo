# Legacy Method Priors

This file summarizes the main method lessons from the completed V1/V2 work.

## Recommendation Path

- The most stable recommendation baseline came from a single `50 min` DCS window.
- `60 min` sometimes improved calcium point error, but the overall recommendation tradeoff favored `50 min`.
- Direct multi-scale concatenation did not beat the better single-window setups.
- Stage-aware optional PH improved recommendation quality in legacy replay, but PH should remain optional.
- The legacy DCS point set should be treated as low-confidence input to V3 because it was originally screened under an earlier `15 min` framing.

## PH Usage

- PH was not a universally strong standalone predictor.
- PH became more useful as a conditional enhancement layered on top of DCS context.
- `120 min` was the most reusable lag candidate.
- `300 min` sometimes looked numerically attractive in a subset of stage experiments, but should not be promoted because it exceeds the practical `4 h` deployment comfort zone.

## Out-Of-Spec Alerting

- Same-sample out-of-spec alerting was feasible, but not fully convincing as a plant-facing single-bit alert.
- Among the methods tested, `single_logistic_oos` gave the best ranking quality.
- Committees, dual-head methods, and anomaly detectors did not clearly beat the logistic baseline.
- Temporal persistence rules reduced switching and chatter, but did not fix the false-alarm burden enough on their own.

## V3 Implication

The most promising V3 methodological shift is not "more model variety" inside the same old framing.

The stronger candidate is:

1. Redefine the alert objective.
2. Consider event-level or future-window warning.
3. Rebuild feature candidates and representative DCS points before treating `50 min` as more than a baseline prior.
4. Use legacy `50 min DCS` as a baseline reference, not as a hard constraint.
5. Keep PH optional and strictly secondary until V3 proves otherwise.
