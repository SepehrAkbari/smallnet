# smallnet

This repository studies low-rank tensor factorization for semantic segmentation
as a reproducible diagnostic experiment, not as a state-of-the-art computer
vision benchmark. The paper focus is VGG16-FCN32s on CamVid with CP
factorization of the dominant `classifier.0` convolution.

The final paper experiment is CP-only: post-training factorization of
`classifier.0` at ranks 32, 64, 128, 256, and 512, with seeds 0, 1, and 2 and
a common 200-iteration fitting budget. It measures weight reconstruction,
isolated-layer activation distortion, zero-shot validation/test segmentation,
and parameter/MAC costs. The central claim remains modest: compression of one
dominant layer must be audited structurally and functionally, and target-layer
compression is not full-model compression.

When the config uses `FactorizedConv.from_conv` with `init: "random"` and
`n_iter_max > 0`, the random value is the tensor-decomposition initializer.
This is post-training CP factorization fitted to the dense convolution, not a
claim that the neural layer is simply randomly initialized.

## Repository Layout

- `src/model.py`: VGG16-FCN32s model definition.
- `src/dataset.py`: CamVid image and RGB-mask loader.
- `src/smallnet/`: reusable experiment utilities for config loading, data,
  evaluation, CP factorization, profiling, rank diagnostics, metadata, and
  reproducibility.
- `scripts/run_experiment.py`: canonical experiment entrypoint.
- `configs/camvid_vgg_cp.json`: default reproducible CamVid/VGG/CP config.
- `configs/camvid_vgg_cp_paper.json`: paper-oriented rank sweep config.
- `docs/dataset_validation.md`: CamVid pairing, RGB-mask, and validation
  report workflow.
- `docs/colab_runbook.md`: Colab/T4 execution and artifact download notes.
- `tests/`: lightweight tests using synthetic tensors and tiny temporary data.
- `res/`: historical regenerated result summaries and paper assets.
- `results/`: default output directory for new experiment runs.
- `results/camvid_vgg_cp/final_cp_200/`: canonical CP-only paper dataset.

Older scripts in `scripts/` are retained for compatibility and historical
result regeneration. New paper runs should use `scripts/run_experiment.py`.
Matrix-SVD development was abandoned and is outside the final paper scope.
Older fine-tuned checkpoints remain historical artifacts and are not part of
the canonical 200-iteration post-training experiment.

## Data Layout

CamVid data is not included in this repository. Place it as:

```text
data/CamVid/
  class_dict.csv
  train/
  train_labels/
  val/
  val_labels/
  test/
  test_labels/
```

Run dataset validation before regenerating paper results. The loader uses
explicit image-mask pairing and strict RGB label conversion; it does not fall
back to positional filename sorting or silently map unknown colors to class
zero. No class is silently ignored. The default config explicitly excludes the
CamVid `Void` class with `ignore_index: 30`.

## Checkpoints

Large checkpoints and datasets should not be committed. The default dense
checkpoint path is:

```text
model/best_model.pth
```

Generated CP checkpoints are saved only when enabled in the config and are
ignored under `results/**/checkpoints/`.

## Configuration

Edit `configs/camvid_vgg_cp.json` to control:

- dataset root and class dictionary path,
- image size and number of classes,
- target layer,
- CP ranks, seeds, initialization, and ALS iterations,
- training and fine-tuning epochs,
- batch size and learning rate,
- device,
- output directory,
- split list and explicit ignored class,
- profiling latency settings,
- rank-energy thresholds.

Each run writes the exact config used to `results/camvid_vgg_cp/`.

## Commands

Install dependencies with `uv sync`, then run tests:

```bash
uv run python -m pytest
```

Validate the committed final paper artifacts without CamVid, CUDA, or the
dense checkpoint:

```bash
uv run python scripts/validate_final_cp_paper_artifacts.py
```

Validate CamVid before any paper experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage validate-data \
  --device cpu
```

Dense baseline evaluation:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage dense
```

CP rank-sweep zero-shot evaluation:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage zero-shot
```

Evaluate existing fine-tuned CP checkpoints without retraining:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage eval-finetuned
```

CP rank-sweep fine-tuning:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage finetune
```

Profiling:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage profile
```

Rank-energy diagnostics:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage rank
```

Full pipeline:

```bash
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp.json --stage full
```

For quick smoke checks, add `--device cpu --max-batches 0` to evaluation stages.

## Outputs

The default output directory is `results/camvid_vgg_cp/`. Each stage writes:

- `<stage>_metadata.json`: environment metadata, git hash, config, skipped
  splits, confusion matrices, and per-class metrics where applicable,
- `<stage>_summary.csv`: one row per model/split result,
- `<stage>_config_used.json`: exact run configuration.

The `validate-data` stage writes `dataset_validation_report.json`,
`dataset_validation_summary.csv`, `dataset_class_counts.csv`, and
`dataset_validation_config_used.json`; see `docs/dataset_validation.md`.

Metadata includes timestamp, device name, PyTorch version, CUDA availability
and version, git commit hash, and dirty-worktree status when available.

The final paper-facing tables and figures are generated from the immutable
completed rows with:

```bash
uv run python scripts/build_final_cp_paper_artifacts.py
```
