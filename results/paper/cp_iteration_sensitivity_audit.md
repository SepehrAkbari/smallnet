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
| 128 | 200 | 0.891966190 | 0.000134121 | 0.000318529 | 0.159651160 |
| 128 | 400 | 0.891338914 | 0.000157598 | 0.000367949 | 0.159023884 |
| 256 | 10 | 0.864526949 | 0.000013443 | 0.000029993 | 0.204287830 |
| 256 | 25 | 0.855216880 | 0.000012177 | 0.000028228 | 0.194977761 |
| 256 | 50 | 0.850743058 | 0.000066848 | 0.000158042 | 0.190503938 |
| 256 | 100 | 0.847853971 | 0.000097262 | 0.000237770 | 0.187614852 |
| 256 | 200 | 0.846052189 | 0.000071511 | 0.000169086 | 0.185813069 |
| 256 | 400 | 0.845276589 | 0.000619274 | 0.001366558 | 0.185037470 |
| 512 | 10 | 0.807650989 | 0.000172536 | 0.000418184 | 0.254440073 |
| 512 | 25 | 0.793902579 | 0.000110956 | 0.000270456 | 0.240691663 |
| 512 | 50 | 0.787474859 | 0.000080270 | 0.000195451 | 0.234263943 |
| 512 | 100 | 0.783410227 | 0.000108803 | 0.000257702 | 0.230199311 |
| 512 | 200 | 0.780846359 | 0.000089049 | 0.000212194 | 0.227635443 |
| 512 | 400 | 0.799820189 | 0.009738946 | 0.020681335 | 0.246609273 |

## Adjacent-budget diagnostics

| Rank | Lower | Upper | Mean absolute reduction | Mean relative reduction | Absolute <1e-3 | Relative <1% | Seed-range change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10 | 25 | 0.005950395 | 0.6582% | False | True | -0.000085083 |
| 128 | 25 | 50 | 0.002903846 | 0.3233% | False | True | 0.000179387 |
| 128 | 50 | 100 | 0.001926229 | 0.2152% | False | True | 0.000052497 |
| 128 | 100 | 200 | 0.001352580 | 0.1514% | False | True | -0.000102343 |
| 128 | 200 | 400 | 0.000627276 | 0.0703% | True | True | 0.000049420 |
| 256 | 10 | 25 | 0.009310069 | 1.0769% | False | False | -0.000001765 |
| 256 | 25 | 50 | 0.004473823 | 0.5231% | False | True | 0.000129814 |
| 256 | 50 | 100 | 0.002889086 | 0.3396% | False | True | 0.000079728 |
| 256 | 100 | 200 | 0.001801783 | 0.2125% | False | True | -0.000068683 |
| 256 | 200 | 400 | 0.000775600 | 0.0917% | True | True | 0.001197471 |
| 512 | 10 | 25 | 0.013748409 | 1.7023% | False | False | -0.000147728 |
| 512 | 25 | 50 | 0.006427721 | 0.8096% | False | True | -0.000075005 |
| 512 | 50 | 100 | 0.004064632 | 0.5162% | False | True | 0.000062252 |
| 512 | 100 | 200 | 0.002563868 | 0.3273% | False | True | -0.000045508 |
| 512 | 200 | 400 | -0.018973830 | -2.4299% | False | False | 0.020469141 |

## Highest-two-budget stopping diagnostics

- Rank 128, 200 to 400: mean absolute reduction `0.000627276`; relative reduction `0.0703%`; absolute below `1e-3`=`True`; relative below 1%=`True`; seed-range change `0.000049420`.
- Rank 256, 200 to 400: mean absolute reduction `0.000775600`; relative reduction `0.0917%`; absolute below `1e-3`=`True`; relative below 1%=`True`; seed-range change `0.001197471`.
- Rank 512, 200 to 400: mean absolute reduction `-0.018973830`; relative reduction `-2.4299%`; absolute below `1e-3`=`False`; relative below 1%=`False`; seed-range change `0.020469141`.
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

1. **Is 400 iterations near a descriptive residual plateau?** `False`: all 200-to-400 absolute changes below `1e-3`=`False` and all relative changes below 1%=`False`. This is residual stabilization, not convergence.
2. **Is 200 a sufficiently stable common budget?** `False` under the requested 200-to-400 checks.
3. **Is 400 necessary?** `True` under those checks; if false, 200 already provides the stable common endpoint.
4. **Does the CP--SVD gap remain large at 400?** `True` under the transparent descriptive reading that at least half of the original ten-iteration gap remains at every rank. The remaining gaps are rank 128: `0.159024`, rank 256: `0.185037`, rank 512: `0.246609`; the fractions of the original ten-iteration gap remaining are rank 128: `92.57%`, rank 256: `90.58%`, rank 512: `96.92%`. These magnitudes quantify the remaining gap without imposing an additional universal cutoff.
5. **Has rank ordering changed?** `False`. The gap still widens monotonically with rank at 400=`True`.
6. **Should later CP decompositions use 100, 200, or 400 iterations?** Use the common budget `400` under the requested adjacent-pair checks; do not use rank-specific budgets.
7. **Is one final 800-iteration check needed?** `True`. It is recommended only because at least one rank has mean absolute 200-to-400 reduction greater than or equal to `1e-3`=`True`.
