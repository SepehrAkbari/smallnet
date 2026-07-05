# smallnet

This repository studies low-rank tensor factorization for semantic segmentation
as a reproducible diagnostic experiment, not as a state-of-the-art computer
vision benchmark. The paper focus is VGG16-FCN32s on CamVid with CP
factorization of the dominant `classifier.0` convolution.

The central claim is modest: low-rank tensor factorization can sharply reduce
the parameter count of a dominant FCN classifier layer, but rank-energy
diagnostics, downstream mIoU, and hardware latency can disagree. Compression
claims should therefore be audited structurally and empirically.

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
- `docs/colab_runbook.md`: Colab/T4 execution and artifact download notes.
- `tests/`: lightweight tests using synthetic tensors and tiny temporary data.
- `res/`: historical regenerated result summaries and paper assets.
- `results/`: default output directory for new experiment runs.

Older scripts in `scripts/` are retained for compatibility and historical
result regeneration. New paper runs should use `scripts/run_experiment.py`.

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

If `test_labels/` is absent, evaluation skips the test split and records the
skip reason in the JSON metadata. No class is silently ignored. The default
config explicitly excludes the CamVid `Void` class with `ignore_index: 30`.

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

Metadata includes timestamp, device name, PyTorch version, CUDA availability
and version, git commit hash, and dirty-worktree status when available.
