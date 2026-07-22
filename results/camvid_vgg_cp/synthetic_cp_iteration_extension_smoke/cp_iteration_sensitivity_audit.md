# CP iteration-budget sensitivity audit

Status: **complete**.

This diagnostic reports completed requested budgets and residual stabilization. It does not claim certified convergence because no per-iteration convergence history is available.

## Rank-level diagnostics

| Rank | Budget | Mean squared residual | Population SD | Seed range | Mean gap above bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 0.881815921 | 0.001508464 | 0.003229507 | 0.179105490 |
| 1 | 25 | 0.881656469 | 0.001690206 | 0.003585483 | 0.178946038 |
| 1 | 50 | 0.881656377 | 0.001690335 | 0.003585758 | 0.178945946 |
| 1 | 100 | 0.881656377 | 0.001690335 | 0.003585758 | 0.178945946 |
| 1 | 200 | 0.881656377 | 0.001690335 | 0.003585758 | 0.178945946 |
| 1 | 400 | 0.881656377 | 0.001690335 | 0.003585758 | 0.178945946 |
| 2 | 10 | 0.790677003 | 0.013024267 | 0.031675547 | 0.332300103 |
| 2 | 25 | 0.779004582 | 0.006128975 | 0.014038293 | 0.320627683 |
| 2 | 50 | 0.761991639 | 0.008104924 | 0.017241893 | 0.303614739 |
| 2 | 100 | 0.761946053 | 0.008116448 | 0.017217767 | 0.303569153 |
| 2 | 200 | 0.761944881 | 0.008114791 | 0.017214252 | 0.303567981 |
| 2 | 400 | 0.761944881 | 0.008114791 | 0.017214252 | 0.303567981 |
| 4 | 10 | 0.599401300 | 0.006966480 | 0.017055194 | 0.423721267 |
| 4 | 25 | 0.590487819 | 0.008028744 | 0.019061988 | 0.414807786 |
| 4 | 50 | 0.578845338 | 0.006755890 | 0.016400341 | 0.403165306 |
| 4 | 100 | 0.567083849 | 0.013386173 | 0.028528803 | 0.391403816 |
| 4 | 200 | 0.567007070 | 0.013327286 | 0.028412844 | 0.391327037 |
| 4 | 400 | 0.566986216 | 0.013297796 | 0.028350283 | 0.391306183 |

## Adjacent-budget diagnostics

| Rank | Lower | Upper | Mean absolute reduction | Mean relative reduction | Absolute <1e-3 | Relative <1% | Seed-range change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 25 | 0.000159452 | 0.0181% | True | True | 0.000355976 |
| 1 | 25 | 50 | 0.000000092 | 0.0000% | True | True | 0.000000275 |
| 1 | 50 | 100 | 0.000000000 | 0.0000% | True | True | 0.000000000 |
| 1 | 100 | 200 | 0.000000000 | 0.0000% | True | True | 0.000000000 |
| 1 | 200 | 400 | 0.000000000 | 0.0000% | True | True | 0.000000000 |
| 2 | 10 | 25 | 0.011672421 | 1.4763% | False | False | -0.017637254 |
| 2 | 25 | 50 | 0.017012944 | 2.1839% | False | False | 0.003203600 |
| 2 | 50 | 100 | 0.000045586 | 0.0060% | True | True | -0.000024126 |
| 2 | 100 | 200 | 0.000001172 | 0.0002% | True | True | -0.000003515 |
| 2 | 200 | 400 | 0.000000000 | 0.0000% | True | True | 0.000000000 |
| 4 | 10 | 25 | 0.008913481 | 1.4871% | False | False | 0.002006794 |
| 4 | 25 | 50 | 0.011642481 | 1.9717% | False | False | -0.002661648 |
| 4 | 50 | 100 | 0.011761489 | 2.0319% | False | False | 0.012128463 |
| 4 | 100 | 200 | 0.000076780 | 0.0135% | True | True | -0.000115959 |
| 4 | 200 | 400 | 0.000020854 | 0.0037% | True | True | -0.000062561 |

## Highest-two-budget stopping diagnostics

- Rank 1, 200 to 400: mean absolute reduction `0.000000000`; relative reduction `0.0000%`; absolute below `1e-3`=`True`; relative below 1%=`True`; seed-range change `0.000000000`.
- Rank 2, 200 to 400: mean absolute reduction `0.000000000`; relative reduction `0.0000%`; absolute below `1e-3`=`True`; relative below 1%=`True`; seed-range change `0.000000000`.
- Rank 4, 200 to 400: mean absolute reduction `0.000020854`; relative reduction `0.0037%`; absolute below `1e-3`=`True`; relative below 1%=`True`; seed-range change `-0.000062561`.
- Rank ordering changed across complete budgets: `False`.

## Scientific decisions

1. **Is 400 iterations near a descriptive residual plateau?** `True`: all 200-to-400 absolute changes below `1e-3`=`True` and all relative changes below 1%=`True`. This is residual stabilization, not convergence.
2. **Is 200 a sufficiently stable common budget?** `True` under the requested 200-to-400 checks.
3. **Is 400 necessary?** `False` under those checks; if false, 200 already provides the stable common endpoint.
4. **Does the CP--SVD gap remain large at 400?** `True` under the transparent descriptive reading that at least half of the original ten-iteration gap remains at every rank. The remaining gaps are rank 1: `0.178946`, rank 2: `0.303568`, rank 4: `0.391306`; the fractions of the original ten-iteration gap remaining are rank 1: `99.92%`, rank 2: `91.35%`, rank 4: `92.35%`. These magnitudes quantify the remaining gap without imposing an additional universal cutoff.
5. **Has rank ordering changed?** `False`. The gap still widens monotonically with rank at 400=`True`.
6. **Should later CP decompositions use 100, 200, or 400 iterations?** Use the common budget `100` under the requested adjacent-pair checks; do not use rank-specific budgets.
7. **Is one final 800-iteration check needed?** `False`. It is recommended only because at least one rank has mean absolute 200-to-400 reduction greater than or equal to `1e-3`=`False`.
