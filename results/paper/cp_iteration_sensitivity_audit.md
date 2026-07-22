# CP iteration-budget sensitivity audit

Status: **complete**.

This diagnostic reports completed requested budgets and residual stabilization. It does not claim certified convergence because no per-iteration convergence history is available.

## Rank-level diagnostics

| Rank | Budget | Mean squared residual | Population SD | Seed range | Mean gap above bound |
|---:|---:|---:|---:|---:|---:|
| 128 | 10 | 0.904099241 | 0.000113080 | 0.000274071 | 0.171784210 |
| 128 | 25 | 0.898148846 | 0.000083929 | 0.000188989 | 0.165833816 |
| 128 | 50 | 0.895245000 | 0.000156815 | 0.000368376 | 0.162929970 |
| 128 | 100 | 0.893318771 | 0.000174707 | 0.000420873 | 0.161003741 |
| 256 | 10 | 0.864526949 | 0.000013443 | 0.000029993 | 0.204287830 |
| 256 | 25 | 0.855216880 | 0.000012177 | 0.000028228 | 0.194977761 |
| 256 | 50 | 0.850743058 | 0.000066848 | 0.000158042 | 0.190503938 |
| 256 | 100 | 0.847853971 | 0.000097262 | 0.000237770 | 0.187614852 |
| 512 | 10 | 0.807650989 | 0.000172536 | 0.000418184 | 0.254440073 |
| 512 | 25 | 0.793902579 | 0.000110956 | 0.000270456 | 0.240691663 |
| 512 | 50 | 0.787474859 | 0.000080270 | 0.000195451 | 0.234263943 |
| 512 | 100 | 0.783410227 | 0.000108803 | 0.000257702 | 0.230199311 |

## Required descriptive thresholds

- Rank 128: mean 50-to-100 reduction `0.001926229`; less than `1e-3`: `False`. Mean relative 10-to-100 reduction `1.1924%`; less than 1%: `False`. Seed variability `increased`.
- Rank 256: mean 50-to-100 reduction `0.002889086`; less than `1e-3`: `False`. Mean relative 10-to-100 reduction `1.9286%`; less than 1%: `False`. Seed variability `increased`.
- Rank 512: mean 50-to-100 reduction `0.004064632`; less than `1e-3`: `False`. Mean relative 10-to-100 reduction `3.0014%`; less than 1%: `False`. Seed variability `decreased`.
- Rank ordering changed across complete budgets: `False`.

## Canonical ten-iteration reproduction

- rank=128, seed=0: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=128, seed=1: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=128, seed=2: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=256, seed=0: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=256, seed=1: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=256, seed=2: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=512, seed=0: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=512, seed=1: absolute residual difference `0.000e+00`, within tolerance=`True`.
- rank=512, seed=2: absolute residual difference `0.000e+00`, within tolerance=`True`.

## Scientific decisions

1. **Is ten iterations near a residual plateau?** `False` under the two requested descriptive checks: all-rank 50-to-100 change below 1e-3=`False` and all-rank absolute relative 10-to-100 change below 1%=`False`. This is residual stabilization, not certified convergence.
2. **Does a larger budget substantially reduce the CP--SVD gap?** At least one rank changes by 1% or more, so budget sensitivity is materially present under the requested diagnostic. The fractions of the ten-iteration gap removed by 100 iterations are rank 128: `6.2756%`, rank 256: `8.1615%`, rank 512: `9.5271%`. These descriptive fractions, rather than a preset conclusion, determine whether the gap reduction is material.
3. **Is the widening high-rank gap present at 100 iterations?** The 100-iteration mean gaps are rank 128: `0.161004`, rank 256: `0.187615`, rank 512: `0.230199`. The gap increases monotonically with rank=`True`; rank ordering changed=`False`.
4. **Does any rank require a larger canonical budget?** Ranks failing at least one requested stabilization check: `[128, 256, 512]`. This is a protocol decision, not a convergence certificate.
5. **Budget for later experiments:** No final common budget is justified yet; extend only ranks whose 50-to-100 change is at least 1e-3, then select a common stabilized budget.
6. **Would a budget change require rerunning canonical zero-shot results?** Yes. Any CP factors used for zero-shot evaluation must be refitted at the selected budget and the corresponding zero-shot rows regenerated. The existing ten-iteration artifacts must remain as a separately labeled protocol.
7. **Is another budget experiment necessary?** No further budget grid is required if the 50-to-100 changes are descriptively negligible for all ranks; otherwise extend only the affected ranks with the same initialization protocol.
