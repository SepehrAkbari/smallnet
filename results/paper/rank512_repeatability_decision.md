# Rank-512 CP repeatability decision

This is an intentionally lean protocol decision, not a completed version of the original 37-row numerical-stability grid.

## Protocol validation

- Completed lean scientific rows: **12/12**.
- Exact residual repeatability across repetitions: **True**.
- Final-factor hashes are identical within every seed/budget group: **True**.
- Residual verification passed for every row: **True**.
- Nonfinite rows: **0**.
- Failed rows: **0**.

## Residual comparison

- Mean normalized squared residual at 200 iterations: `0.780846358733`.
- Mean normalized squared residual at 400 iterations: `0.799820188518`.
- Mean deterioration from 200 to 400 iterations: `0.018973829785`.
- Seed range at 200 iterations: `0.000212194308`.
- Seed range at 400 iterations: `0.020681335289`.

## Factor-scaling diagnostics

- seed 0, budget 200: max scaling spread `195.845`, max cancellation ratio `1360.07`, max absolute factor value `9.24331`, threshold-based degeneracy flag `True`.
- seed 0, budget 400: max scaling spread `1387.44`, max cancellation ratio `1233.01`, max absolute factor value `8.97856`, threshold-based degeneracy flag `True`.
- seed 1, budget 200: max scaling spread `145.625`, max cancellation ratio `1414.97`, max absolute factor value `8.86717`, threshold-based degeneracy flag `True`.
- seed 1, budget 400: max scaling spread `119785`, max cancellation ratio `143.327`, max absolute factor value `10.0386`, threshold-based degeneracy flag `False`.
- seed 2, budget 200: max scaling spread `152.963`, max cancellation ratio `1422.55`, max absolute factor value `7.35932`, threshold-based degeneracy flag `True`.
- seed 2, budget 400: max scaling spread `33410.8`, max cancellation ratio `261.051`, max absolute factor value `9.88939`, threshold-based degeneracy flag `False`.

The recorded threshold flags are descriptive diagnostics. They do not certify CP degeneracy, but the exact repeated residuals and final-factor hashes establish that the 200-to-400 deterioration is reproducible rather than an isolated corrupted run.

## Accepted decision

**Use a common CP fitting budget of 200 iterations for all downstream ranks.**

The remaining rank-512 stability budgets and float64 optimization subset will not be run. The intentionally partial full-grid audit must remain labeled incomplete.
