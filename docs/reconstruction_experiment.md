# Structural reconstruction experiment

This experiment compares necessary unfolding-rank constraints, fitted CP weight reconstruction, an exact matrix low-rank control, zero-shot segmentation metrics, and representation cost. It does not fine-tune a model and does not measure activation distortion.

## Quantities

For a convolution tensor (W \in \mathbb{R}^{C_{out}\times C_{in}\times k_h\times k_w}), let (W_{(m)}) be its mode-(m) unfolding and let its singular values be σ₁ ≥ σ₂ ≥ …. The cumulative spectral energy and squared tail are

\[
E_m(r)=\frac{\sum_{j=1}^r \sigma_j(W_{(m)})^2}{\sum_j \sigma_j(W_{(m)})^2},
\qquad L_m(r)=1-E_m(r).
\]

The strongest unfolding lower bound is (L(r)=\max_m L_m(r)). Every CP-rank-(r) tensor has matrix rank at most (r) in every unfolding. Eckart–Young therefore gives the necessary inequality

\[
\frac{\lVert W-\widehat W^{CP}_r\rVert_F^2}{\lVert W\rVert_F^2} \ge L(r).
\]

The unfolding tail is a lower bound, not the fitted CP error. It does not account for the requirement that one set of CP factors must satisfy all modes simultaneously, the initializer, the finite optimization budget, or downstream segmentation behavior.

Artifacts use two deliberately distinct names:

- `actual_relative_squared_frobenius_error` is \(\lVert W-\widehat W\rVert_F^2/\lVert W\rVert_F^2\).
- `actual_relative_frobenius_error` is its square root, \(\lVert W-\widehat W\rVert_F/\lVert W\rVert_F\).

## Output-unfolding matrix-SVD control

The deterministic control reshapes the kernel to

\[
W_{(0)} \in \mathbb{R}^{C_{out}\times(C_{in}k_hk_w)}
\]

and retains its first (r) singular triplets. `MatrixLowRankConv2d` implements the result as a (k_h\times k_w) convolution from (C_{in}) to (r) channels followed by a (1\times1) convolution from (r) to (C_{out}). The stage verifies the composed kernel against (U_r\Sigma_rV_r^T) and verifies that its normalized squared residual equals (1-E_0(r)) within the configured tolerance. This method is labeled `matrix_svd_output_unfolding`; it is a matrix low-rank baseline, not a tensor decomposition.

Including bias, the target-layer parameter counts are

\[
P_{dense}=C_{out}C_{in}k_hk_w+C_{out},
\]

\[
P_{CP}=r(C_{out}+C_{in}+k_h+k_w+1)+C_{out},
\]

and

\[
P_{SVD}=r(C_{in}k_hk_w+C_{out})+C_{out}.
\]

The `+1` in the CP formula counts the rank-length component-weight vector stored by `tensorly-torch`.

## Reproducibility protocol

The canonical ranks are 32, 64, 128, 256, and 512. CP is fitted separately with seeds 0, 1, and 2. Before each fit, Python, NumPy, PyTorch CPU, and available CUDA generators are seeded and deterministic PyTorch settings are requested. The metadata records the initializer, requested `n_iter_max`, deterministic settings, actual fitting device, package versions, runtime, and failures.

The canonical configuration retains the existing random initializer and 10-iteration budget. It enables the repository's mathematically equivalent, memory-bounded hybrid MTTKRP contraction. The default TensorLy backend explicitly forms Khatri–Rao products; for the canonical tensor its largest such temporary is approximately 28 GiB at rank 512 and 14 GiB at rank 256. The hybrid uses TensorLy's fast explicit contraction when the estimated temporary is at most 512 MiB and otherwise contracts rank columns in chunks of 64. The selected backend and both limits are recorded in every CP row. This changes the contraction implementation, not the fitted objective, initializer, rank, seed, or iteration budget.

`tensorly-torch` 0.5 does not expose iteration loss history through `FactorizedConv.from_conv`; consequently, a completed run records that the requested budget completed but does not claim certified convergence or an observed iteration count. Poor residuals at this budget should be reported as evidence that the protocol may be optimization-limited, not silently addressed by increasing the budget.

Each successful or failed rank/seed is written incrementally. Re-running a subset merges by method/rank/seed, and a new failure does not replace an already completed reconstruction row.

Ranks and CP seeds are normalized to nonnegative integers whenever saved CSV rows are loaded. Matrix-SVD seeds remain empty. Malformed ranks, summary labels such as `all` or `mean`, and duplicate scientific keys are excluded from aggregation and figures while their raw rows and normalization diagnostics remain available in the CSV and metadata. Figure-generation failures are nonfatal after computation rows have been saved.

## Commands

Validate the implementation and run the dataset-free smoke test locally:

```bash
uv run python -m pytest

uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage reconstruction \
  --device cpu \
  --synthetic-smoke
```

Run the canonical reconstruction stage:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage reconstruction \
  --device cpu
```

Regenerate Figures A and B from saved rows and spectral metadata without loading the checkpoint or rerunning SVD/CP decomposition:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage reconstruction-figures \
  --device cpu
```

To resume a partially completed rank, request the intended rank and seeds again. Completed matrix-SVD and CP keys are skipped automatically; only missing or failed combinations are recomputed:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage reconstruction \
  --device cuda \
  --ranks 32 \
  --seeds 0 1 2
```

After reconstruction is complete, run the validated zero-shot comparison:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage structural-zero-shot \
  --device cpu
```

The zero-shot stage requires the canonical `dataset_validation_report.json` and `reconstruction_summary.csv`. It hashes both the dataset-validation report and dense checkpoint, uses validation and test splits, and re-fits CP with the same rank/seed before accepting a join to the reconstruction row.

## Colab or GPU execution

Use the same commands with `--device cuda`. CP fitting, the exact output-mode matrix SVD, and segmentation evaluation use the requested device where PyTorch supports the operation; an unsupported or failed accelerator SVD falls back to CPU and records the actual device. The remaining mode-wise spectra run on CPU. Because Colab sessions are ephemeral, run one rank and CP seed subset at a time and download the output directory after every completed command:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage reconstruction \
  --device cuda \
  --ranks 32 \
  --seeds 0 1 2

uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage structural-zero-shot \
  --device cuda \
  --ranks 32 \
  --seeds 0 1 2
```

Repeat with ranks 64, 128, 256, and 512. Preserve the same output directory between phases so incremental merging retains completed rows.

## CP iteration-budget sensitivity

The dedicated sensitivity stage is isolated from `reconstruction_summary.csv` and all zero-shot artifacts. Its primary protocol independently fits ranks 128, 256, and 512 with seeds 0, 1, and 2 at requested budgets 10, 25, 50, and 100. Before every fit it constructs the zero-iteration CP factors, hashes the component weights and factor matrices, verifies that the hash agrees across budgets for the same rank and seed, and then resets the seed immediately before the independent budgeted fit. It does not warm-start from a shorter run.

Run a dataset- and checkpoint-free smoke experiment locally:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage cp-iteration-sensitivity \
  --device cpu \
  --synthetic-smoke
```

Run or resume one canonical rank at a time on CUDA:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage cp-iteration-sensitivity \
  --device cuda \
  --ranks 128 \
  --seeds 0 1 2 \
  --iteration-budgets 10 25 50 100
```

Repeat with ranks 256 and 512. Every completed rank/seed/budget row is written immediately. Rerunning the same command skips completed keys, retries missing or failed keys, recalculates comparisons and aggregates, and regenerates the partial or complete figure. A figure or audit-generation exception is recorded without deleting successful computation rows.

Sensitivity outputs are distinct from the canonical ten-iteration table:

- `results/camvid_vgg_cp/cp_iteration_sensitivity_summary.csv`
- `results/camvid_vgg_cp/cp_iteration_sensitivity_rank_summary.csv`
- `results/camvid_vgg_cp/cp_iteration_sensitivity_metadata.json`
- `results/camvid_vgg_cp/cp_iteration_sensitivity_config_used.json`
- `results/paper/figures/cp_iteration_sensitivity.csv`
- `results/paper/figures/cp_iteration_sensitivity.pdf`
- `results/paper/figures/cp_iteration_sensitivity.png`
- `results/paper/cp_iteration_sensitivity_audit.md`

The rank summary reports population standard deviation, minimum, maximum, seed range, residual changes, remaining lower-bound gap, the specified descriptive thresholds, rank ordering, and seed-variability direction. Completion means the requested budget ran; it does not certify optimization convergence.

## Outputs

Canonical reconstruction artifacts:

- `results/camvid_vgg_cp/reconstruction_summary.csv`: every deterministic matrix-SVD rank and every CP rank/seed.
- `results/camvid_vgg_cp/reconstruction_rank_summary.csv`: CP mean, population standard deviation, minimum, and maximum squared residual by rank.
- `results/camvid_vgg_cp/reconstruction_metadata.json`: spectral data, versions, deterministic settings, convergence limitations, failures, and output references.
- `results/camvid_vgg_cp/reconstruction_config_used.json`: exact configuration snapshot.
- `results/camvid_vgg_cp/reconstruction_figures_metadata.json`: figure-only regeneration outputs, normalization diagnostics, and nonfatal figure failures.

Canonical mask-dependent and joined artifacts:

- `results/camvid_vgg_cp/structural_zero_shot_summary.csv`
- `results/camvid_vgg_cp/structural_zero_shot_metadata.json`
- `results/camvid_vgg_cp/structural_tradeoff_summary.csv`
- `results/camvid_vgg_cp/structural_tradeoff_correlations.json`

Paper figure data, PNGs, and PDFs are written under `results/paper/figures/` with prefixes `figure_a_`, `figure_b_`, and `figure_c_`. Figure B always labels its vertical quantity as normalized squared Frobenius error. Figure C uses present-class mIoU and retains individual CP seeds.

Failed fits or evaluations are recorded with method, rank, seed, exception, and environment metadata. A missing or failed row must not be described as a successful experiment. Correlations are labeled exploratory and descriptive: no significance tests are computed, and neither the tables nor figures establish that a spectral bound predicts mIoU.
