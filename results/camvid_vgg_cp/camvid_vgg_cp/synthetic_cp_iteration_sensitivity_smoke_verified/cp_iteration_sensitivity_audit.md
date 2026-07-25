# CP iteration-budget sensitivity audit

Status: **complete**.

This diagnostic reports completed requested budgets and residual stabilization. It does not claim certified convergence because no per-iteration convergence history is available.

## Rank-level diagnostics

| Rank | Budget | Mean squared residual | Population SD | Seed range | Mean gap above bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.951135441 | 0.012102616 | 0.029139379 | 0.158100929 |
| 1 | 2 | 0.934393533 | 0.008408516 | 0.018289614 | 0.141359021 |
| 1 | 3 | 0.932540974 | 0.008640974 | 0.018473527 | 0.139506462 |
| 1 | 4 | 0.931915616 | 0.008570327 | 0.018257525 | 0.138881104 |
| 2 | 1 | 0.914352192 | 0.014740785 | 0.036025653 | 0.298886576 |
| 2 | 2 | 0.876690880 | 0.000722866 | 0.001713292 | 0.261225264 |
| 2 | 3 | 0.870180867 | 0.002689793 | 0.006586879 | 0.254715251 |
| 2 | 4 | 0.867364007 | 0.004083316 | 0.009962720 | 0.251898391 |
| 4 | 1 | 0.817974563 | 0.017104430 | 0.041844157 | 0.474141602 |
| 4 | 2 | 0.769384803 | 0.015412433 | 0.034095294 | 0.425551841 |
| 4 | 3 | 0.754450520 | 0.011889090 | 0.026652117 | 0.410617559 |
| 4 | 4 | 0.744123427 | 0.011514623 | 0.026312597 | 0.400290465 |

## Required descriptive thresholds

- Threshold decisions are deferred until the complete 10/25/50/100 canonical grid is available.

## Scientific decisions

1. Ten-iteration plateau decision: deferred.
2. CP--SVD gap-reduction decision: deferred.
3. High-rank 100-iteration gap decision: deferred.
4. Canonical-budget decision: deferred.
5. Later-experiment budget decision: deferred.
6. If the selected fitting budget changes, canonical zero-shot CP results must be rerun and separately labeled.
7. Further iteration-budget work cannot be assessed until the requested grid completes.
