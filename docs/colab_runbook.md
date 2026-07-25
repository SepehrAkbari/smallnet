# Colab/T4 Runbook

Use this runbook for the full CamVid/VGG16-FCN32s CP diagnostic pipeline when
local CPU or MPS execution is too slow.

## Setup

1. Start a Colab runtime with a GPU. A Tesla T4 is sufficient for the expected
   paper runs, but latency numbers are device-specific.
2. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/SepehrAkbari/smallnet.git
   cd smallnet
   pip install uv
   uv sync
   ```

3. Place CamVid data at:

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

4. Place checkpoints at:

   ```text
   model/best_model.pth
   model/finetuned_rank_64.pth
   model/finetuned_rank_128.pth
   model/finetuned_rank_256.pth
   ```

## Commands

Run each stage separately and download outputs after each stage. Colab sessions
are ephemeral.

```bash
uv run python -m pytest
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage dense --device cuda
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage zero-shot --device cuda
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage eval-finetuned --device cuda
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage profile --device cuda
uv run python scripts/run_experiment.py --config configs/camvid_vgg_cp_paper.json --stage rank --device cuda
uv run python scripts/build_paper_artifacts.py --config configs/camvid_vgg_cp_paper.json
```

Use `--max-batches 1` only for smoke checks. Do not use smoke outputs as paper
evidence.

## Outputs To Download And Commit

Copy the *contents* of the remote results directories, not the directory
container itself. For example, the trailing slashes below prevent creating
`results/camvid_vgg_cp/camvid_vgg_cp/` or `results/paper/paper/`:

```bash
rsync -av /remote/results/camvid_vgg_cp/ results/camvid_vgg_cp/
rsync -av /remote/results/paper/ results/paper/
```

After transfer, reject accidental repeated roots before committing:

```bash
test ! -e results/camvid_vgg_cp/camvid_vgg_cp
test ! -e results/paper/paper
uv run python scripts/validate_final_cp_paper_artifacts.py
```

Download and commit small generated artifacts:

```text
results/camvid_vgg_cp/*_summary.csv
results/camvid_vgg_cp/*_metadata.json
results/camvid_vgg_cp/*_config_used.json
results/paper/MANIFEST.json
results/paper/tables/*.csv
results/paper/tables/*.tex
results/paper/figures/*.pdf
```

Do not commit:

```text
data/
model/*.pth
results/**/checkpoints/
```

The repository `.gitignore` excludes datasets and checkpoint files. Before
committing, verify:

```bash
git status --short
git check-ignore -v model/finetuned_rank_64.pth model/finetuned_rank_128.pth model/finetuned_rank_256.pth
```

## Interpretation

The zero-shot CP stage uses post-training CP factorization fitted to the dense
convolution with `FactorizedConv.from_conv`. With `init: "random"` and
`n_iter_max > 0`, the random value is the tensor-decomposition initializer, not
a claim that the neural layer is an unfitted random layer.

Rank-energy diagnostics are necessary structural audits for CP compression.
They do not prove that a given CP-rank model will preserve downstream mIoU or
improve hardware latency.

The accepted paper-facing experiment is CP-only: five ranks, three seeds, and
200 fitting iterations, with weight, isolated-layer activation, zero-shot
segmentation, parameter, and MAC diagnostics. Matrix-SVD attempts are retained
only as development provenance and are not paper evidence.
