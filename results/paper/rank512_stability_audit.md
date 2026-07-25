# Rank-512 CP numerical-stability audit

Status: **incomplete**.

Completed requested budgets are reported without a convergence claim. Float32 fits are evaluated in float64 through both the fitted layer reconstruction and an independent chunked CP contraction.

## Budget summary

| Precision | Budget | Mean squared residual | Seed range | Max repetition range | Degeneracy rows | Max cancellation ratio |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 200 | 0.780846359 | 0.000212194 | 0.000e+00 | 6 | 1.423e+03 |
| float32 | 400 | 0.799820189 | 0.020681335 | 0.000e+00 | 2 | 1.233e+03 |

## Scientific decisions

1. Repeatability at 200 and 400: deferred.
2. Deterioration onset: deferred.
3. Seed specificity: deferred.
4. Factor-degeneracy association: deferred.
5. Residual verification: deferred.
6. Precision effect: deferred until the labeled float64 subset is attempted.
7. Budget-800 behavior: deferred.
8. Downstream common budget: deferred.
9. Rank-512 downstream inclusion: deferred.
10. Additional fitting work: complete the requested diagnostic grid.
