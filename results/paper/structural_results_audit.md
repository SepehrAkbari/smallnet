# Scientific audit of structural reconstruction and zero-shot results

## Scope and provenance

This audit covers the completed canonical artifacts in `results/camvid_vgg_cp/` and Figures A--C in `results/paper/figures/`. It does not rerun decomposition or evaluation. The reconstruction and zero-shot metadata identify the same dense checkpoint SHA-256 (`1576b92ebb359a519d57d489f8ee823c033588fbd7c77b1c9091606f42fb74a3`). The zero-shot metadata references dataset-validation report SHA-256 `f1bb932fa776a2065507de6b29bdc174c139a1e20f8c4172e04df74af037f57b`.

The run used a Tesla T4, random CP initialization, ranks 32, 64, 128, 256, and 512, seeds 0, 1, and 2, and a fixed requested budget of 10 CP iterations. The metadata explicitly states that tensorly-torch 0.5 does not expose an iteration history or convergence certificate through the fitted layer interface. Consequently, the completed rows establish execution of the requested budget, not formal convergence.

## Completeness and row integrity

All expected scientific rows are present and completed:

| Artifact | Expected | Observed | Duplicate scientific rows | Failed rows |
|---|---:|---:|---:|---:|
| CP reconstruction | 15 | 15 | 0 | 0 |
| Matrix-SVD reconstruction | 5 | 5 | 0 | 0 |
| CP zero-shot, validation and test | 30 | 30 | 0 | 0 |
| Matrix-SVD zero-shot, validation and test | 10 | 10 | 0 | 0 |
| Dense zero-shot references | 2 | 2 | 0 | 0 |
| Joined structural tradeoff rows | 21 | 21 | 0 | not applicable |

Each CP rank has three reconstruction seeds and three evaluation seeds on each split. Each matrix-SVD rank has one reconstruction and one evaluation on each split. Dense has one row on each split. The reconstruction, reconstruction-figure, zero-shot, and zero-shot-figure metadata contain no failures; zero-shot metadata also contains no skipped splits. Thus no failed run is hidden by the aggregate files.

## Mathematical and accounting checks

All checks pass at the configured tolerance of `1e-5`:

- Every CP squared residual exceeds the strongest unfolding lower bound. The smallest observed gap is `0.1284098341`; this is far outside the tolerance margin.
- Every matrix-SVD squared residual equals the output-mode tail. The maximum absolute discrepancy is `2.22e-16`.
- Every nonsquared relative Frobenius residual equals the square root of the squared residual. The maximum absolute discrepancy is `1.11e-16`.
- The composed matrix-SVD convolution kernel matches the truncated-SVD tensor. Maximum absolute kernel discrepancies range from `2.98e-08` to `5.22e-08`.

For tensor shape `(4096, 512, 7, 7)`, including bias, the formulas used and independently verified are:

\[
P_{\mathrm{dense}}=4096\cdot512\cdot7\cdot7+4096=102{,}764{,}544,
\]

\[
P_{\mathrm{CP}}(r)=r(4096+512+7+7+1)+4096,
\]

\[
P_{\mathrm{SVD}}(r)=r(512\cdot7\cdot7+4096)+4096.
\]

Every recorded parameter count, target-layer parameter ratio, and compression factor agrees with these formulas within floating-point tolerance. The ratio is representation parameters divided by dense target-layer parameters; the compression factor is its reciprocal.

## Results by method and rank

Standard deviations below are population standard deviations over the three configured CP seeds. The exact machine-readable values are in `tables/structural_results_summary.csv`.

| Method | Rank | Squared residual, mean (SD) | Validation present-class mIoU, mean (SD) | Test present-class mIoU, mean (SD) | Target-layer parameters | Ratio | Compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP | 32 | 0.950734 (0.000048) | 0.212581 (0.005072) | 0.191171 (0.007461) | 152,032 | 0.001479 | 675.94x |
| CP | 64 | 0.931057 (0.000133) | 0.245135 (0.003417) | 0.222970 (0.005743) | 299,968 | 0.002919 | 342.59x |
| CP | 128 | 0.904099 (0.000113) | 0.270241 (0.001514) | 0.246119 (0.000558) | 595,840 | 0.005798 | 172.47x |
| CP | 256 | 0.864527 (0.000013) | 0.289474 (0.001502) | 0.264218 (0.000925) | 1,187,584 | 0.011556 | 86.53x |
| CP | 512 | 0.807651 (0.000173) | 0.304144 (0.000838) | 0.281282 (0.000490) | 2,371,072 | 0.023073 | 43.34x |
| Output-unfolding SVD | 32 | 0.822257 | 0.324895 | 0.298720 | 937,984 | 0.009128 | 109.56x |
| Output-unfolding SVD | 64 | 0.781806 | 0.334765 | 0.308905 | 1,871,872 | 0.018215 | 54.90x |
| Output-unfolding SVD | 128 | 0.732315 | 0.349242 | 0.316782 | 3,739,648 | 0.036390 | 27.48x |
| Output-unfolding SVD | 256 | 0.660239 | 0.355529 | 0.320524 | 7,475,200 | 0.072741 | 13.75x |
| Output-unfolding SVD | 512 | 0.553211 | 0.361362 | 0.321913 | 14,946,304 | 0.145442 | 6.88x |
| Dense | -- | 0 | 0.366259 | 0.322016 | 102,764,544 | 1.000000 | 1.00x |

The matrix-SVD test result at rank 512 is effectively at the dense test reference (`0.321913` versus `0.322016`), but its validation result remains below dense (`0.361362` versus `0.366259`). This should be described as near recovery on one split, not general equality with dense performance.

## CP seed sensitivity

| Rank | Squared-residual range | Validation mIoU range | Test mIoU range |
|---:|---:|---:|---:|
| 32 | 0.000108 | 0.012380 | 0.017932 |
| 64 | 0.000326 | 0.007499 | 0.013967 |
| 128 | 0.000274 | 0.003701 | 0.001361 |
| 256 | 0.000030 | 0.003662 | 0.002232 |
| 512 | 0.000418 | 0.002000 | 0.001197 |

Weight residuals are highly repeatable across seeds at every rank. Downstream sensitivity is concentrated at the lowest ranks: rank 32 is the unstable rank for zero-shot interpretation, and rank 64 is moderately seed-sensitive on test. Ranks 128--512 are stable under these three decompositions. The larger mIoU ranges at ranks 32 and 64 despite tiny residual ranges also caution against treating small weight-residual differences as predictive of downstream quality.

Three seeds are adequate for this descriptive diagnostic stage because all individual observations are retained and the main ordering is consistent. They are not adequate for inferential variance claims or significance testing.

## Fixed ten-iteration CP budget

Mean CP squared residual decreases monotonically with rank (`0.9507`, `0.9311`, `0.9041`, `0.8645`, `0.8077`), while zero-shot mIoU improves monotonically on both splits. There is no rank reversal or seed-dependent failure suggesting severe stochastic instability. However, the mean gap above the strongest lower bound grows with rank (`0.1285`, `0.1493`, `0.1718`, `0.2043`, `0.2544`). Because loss histories and actual convergence status are unavailable, that widening gap can reflect CP representational constraints, optimization limitation from the ten-iteration budget, or both.

Ten iterations are defensible only as a clearly labeled fixed-budget protocol. They are not defensible as evidence that the reported tensors are converged or close to the best attainable CP approximations. Before the paper interprets the CP--SVD reconstruction gap as a property of the representation, a limited iteration-budget sensitivity experiment is scientifically necessary. A focused comparison at ranks 128, 256, and 512, retaining multiple seeds and reporting the same residuals at the configured 10 iterations and at one or two larger documented budgets, is sufficient; the sensitivity result must remain separately labeled if the canonical protocol is unchanged.

## Matrix-SVD control and matched compression

At every equal rank, output-unfolding SVD has a substantially lower reconstruction residual and substantially higher zero-shot mIoU than CP. Equal-rank comparisons are not compression-matched: matrix SVD uses approximately 6.3 times as many target-layer parameters as CP at the same rank.

The fairest available near-matches are:

| Matrix SVD | CP | CP / SVD parameter count | SVD residual advantage | SVD validation mIoU advantage | SVD test mIoU advantage |
|---|---|---:|---:|---:|---:|
| rank 32, 937,984 parameters | rank 256, 1,187,584 parameters | 1.266 | 0.042270 | 0.035422 | 0.034502 |
| rank 64, 1,871,872 parameters | rank 512, 2,371,072 parameters | 1.267 | 0.025845 | 0.030620 | 0.027623 |

Even though CP has about 27% more target-layer parameters in these two pairings, matrix SVD remains better in reconstruction and zero-shot mIoU. The control therefore adds scientific insight: the unfolding tail is achieved by its corresponding matrix model and the advantage is not explained only by equal-rank parameter imbalance. It should remain in the paper, explicitly called `matrix_svd_output_unfolding` or output-unfolding truncated SVD, not a tensor decomposition. Tables and captions should prioritize the two near-matched comparisons rather than equal-rank superiority claims.

## Exploratory correlations

The stored correlations were independently recomputed; maximum discrepancies were `3.33e-16` for Pearson and `1.11e-16` for Spearman. All stored coefficients below are defined:

| Scope | Quantity versus present-class mIoU | Validation Pearson / Spearman | Test Pearson / Spearman | n |
|---|---|---:|---:|---:|
| All nondense | strongest unfolding tail | -0.573 / -0.613 | -0.580 / -0.613 | 20 |
| All nondense | squared residual | -0.909 / -0.974 | -0.884 / -0.973 | 20 |
| All nondense | parameter ratio | 0.688 / 0.947 | 0.645 / 0.947 | 20 |
| CP | strongest unfolding tail | -0.935 / -0.982 | -0.938 / -0.982 | 15 |
| CP | squared residual | -0.932 / -0.961 | -0.936 / -0.957 | 15 |
| CP | parameter ratio | 0.866 / 0.982 | 0.874 / 0.982 | 15 |
| Matrix SVD | strongest unfolding tail | -0.935 / -1.000 | -0.875 / -1.000 | 5 |
| Matrix SVD | squared residual | -0.935 / -1.000 | -0.875 / -1.000 | 5 |
| Matrix SVD | parameter ratio | 0.865 / 1.000 | 0.784 / 1.000 | 5 |

The matrix-only subsets have only five observations, so their coefficients are undersized and the perfect Spearman values merely encode monotonic rank ordering. Within a fixed CP rank, the spectral bound and parameter ratio are constant across seeds, so correlations involving those quantities are undefined; residual-only rank subsets have just three observations and are also too small to interpret. Dense-only and matrix-within-rank subsets contain one observation. The all-nondense subset mixes methods with unequal parameterizations, and all reported correlations are strongly confounded by rank. These coefficients are descriptive only; they support no significance, causal, or predictive claim.

## Figures A--C

The companion CSVs match their scientific source tables exactly: Figure A has all 4,622 cumulative-energy points across the four modes with maximum numerical difference `1.11e-16`; Figure B has all 20 reconstruction rows with zero difference; Figure C has all 21 joined test observations with zero difference.

- **Figure A:** The full `0--4096` horizontal range is honest and distinguishes channel from spatial modes by color and line style. It also compresses the spatial curves and all experimental-rank markers into the far-left region, making them hard to read. Retain the full-range panel, but add a clearly labeled low-rank inset or second panel (for example ranks 1--512) rather than truncating the only axis. Label or explain the five vertical rank markers.
- **Figure B:** The vertical axis includes zero, individual CP seeds are retained, and the title and axis correctly say normalized squared Frobenius error. The lower-bound and SVD curves coincide by construction, so one is largely hidden by the other. State that equality explicitly in the caption or offset/use hollow SVD markers so both encodings remain legible. The caption must also warn that equal rank does not imply equal parameter count.
- **Figure C:** The test split and present-class mIoU are explicit; individual seeds, CP mean, matrix SVD, and dense reference are all present. The zero baseline is not truncated. No uncertainty is missing for CP because individual seed points are shown. Add the same equal-rank parameter-count caveat in the caption, and avoid describing matrix rank 512 as full recovery because validation remains below dense.

No plotted value or aggregation error requires regeneration before using the figures. The Figure A inset and Figure B overlap clarification are readability improvements that should be made before final submission.

## Explicit decisions

- **Should rank 512 remain in the paper?** Yes. It is the necessary upper endpoint: CP remains highly compressed (`43.34x` at the target layer), gives the best CP zero-shot result, and still has a large reconstruction gap. Matrix rank 512 also demonstrates test-split saturation near dense while exposing the parameter-cost difference.
- **Are three seeds adequate for this diagnostic stage?** Yes for descriptive means, ranges, and robustness of ordering; no for inferential claims.
- **Is ten CP iterations defensible?** Only as a fixed-budget protocol, not as a convergence claim. A targeted iteration-budget sensitivity check is necessary before attributing the CP--SVD gap primarily to representation.
- **Is matrix SVD useful?** Yes. It is an exact, interpretable control and remains superior in the two available near-matched-compression comparisons. It adds insight if parameter imbalance is stated plainly.
- **Which ranks should proceed to activation-distortion experiments?** CP ranks 64, 256, and 512 provide a low/middle/high structural span without redundant coverage. Include matrix ranks 32 and 64 as the closest compression-matched controls if that stage compares representation families.
- **Which ranks should proceed to controlled fine-tuning?** CP ranks 64, 256, and 512. Rank 64 preserves the high-compression failure boundary, rank 256 is an intermediate tradeoff, and rank 512 is the strongest zero-shot CP candidate. If compute is constrained, prioritize 256 and 512 and retain 64 as the boundary case.
- **Has the central claim changed?** The results do not materially weaken the claim that rank-energy is a necessary structural diagnostic and that higher candidate ranks are more plausible. They sharpen its limitation: the lower bound does not determine fitted CP error or mIoU, and representation plus optimization matter. The strong SVD control prevents interpreting the CP gap as rank insufficiency alone until iteration sensitivity is checked.

## Resource and reproducibility notes

The reconstruction metadata records peak `ru_maxrss=5,480,700` and zero-shot metadata records `6,033,500`; on Linux these correspond to approximately 5.23 GiB and 5.75 GiB, respectively, for the peak process resident set. CP fitting runtimes range from about 0.21 to 1.98 seconds per rank/seed on the T4, while each exact SVD row records about 10.46--10.61 seconds. These timings are environment-specific and should not be presented as model inference latency.

The Colab metadata reports a dirty Git worktree at commit `5f68c3406584a8abc690e9cdb84a94c13481fcac`; the artifact hashes therefore provide the stronger identity check for this audit. No result-processing code was changed and no experiment or test suite was run during this audit.

## Necessary next experiment

Before manuscript interpretation of the reconstruction gap, run only the focused CP iteration-budget sensitivity described above. No additional reconstruction ranks or seeds are required at this diagnostic stage. Activation-distortion and controlled fine-tuning should remain paused until that check determines whether the ten-iteration residuals are materially optimization-limited.
