# Results Directory

This directory contains the regenerated experiment outputs used for the paper
tables and figures.

## Canonical Experiment Summaries

- Dense VGG16-FCN32s evaluation:
  `results/camvid_vgg_cp/dense_eval_summary.csv`
- Zero-shot post-training CP evaluation:
  `results/camvid_vgg_cp/cp_zero_shot_summary.csv`
- Existing fine-tuned CP checkpoint evaluation:
  `results/camvid_vgg_cp/existing_finetuned_summary.csv`
- Profiling:
  `results/camvid_vgg_cp/profile_summary.csv`
- Rank-energy diagnostics:
  `results/camvid_vgg_cp/rank_diagnostics_summary.csv`

The optional `results/camvid_vgg_cp/cp_finetune_summary.csv` file is absent in
the final Colab artifact set because fine-tuning was not rerun in that pass.
The paper uses `existing_finetuned_summary.csv`, which evaluates the locally
available fine-tuned checkpoints without retraining.

## Paper Artifacts

Paper-ready tables, figures, and the artifact manifest are under:

```text
results/paper/
  MANIFEST.json
  tables/
  figures/
```

`results/paper/MANIFEST.json` records generated outputs and missing optional
inputs. A missing `cp_finetune_summary.csv` is expected for the final artifact
set and should not be interpreted as a failed run.

## Large Files

Datasets and model checkpoints are intentionally not stored here. Keep CamVid
data under `data/CamVid/` and checkpoints under `model/`; both are ignored by
git.
