# User Interaction Rules
- Whenever you invoke the `request_user_input` tool to ask the user a question, you must never supply the `autoResolutionMs` parameter.
- You are forbidden from setting automated timeouts or auto-resolving your own clarification questions. 
- Wait until the user submits an answer, no matter how long it takes.

# smallnet Project Context

This repository studies model compression for semantic segmentation through tensor decomposition. The current working direction is a theory-first applied letter/journal article on rank-energy criteria for tensorized convolutional bottlenecks, not a simple "CP compression improves latency" paper.

## Current System

- Model: `src/model.py` defines `VGG16_FCN32s`, a VGG16 feature extractor with FCN32s-style convolutional classifier and bilinear upsampling.
- Dataset: `src/dataset.py` loads CamVid from `data/CamVid/{train,val,test}` and color masks from matching `*_labels` directories. Images/masks are evaluated at `352x480`.
- Main compression target: `classifier.0`, shape `[4096, 512, 7, 7]`, with 102,760,448 weights plus 4,096 bias terms. It dominates the dense checkpoint.
- Existing checkpoints:
  - `model/best_model.pth`: dense model, 134,391,648 parameters, notebook val mIoU about 0.3305.
  - `model/finetuned_rank_256.pth`: CP-factorized `classifier.0`, 32,814,688 parameters.
  - `model/finetuned_rank_128.pth`: CP-factorized `classifier.0`, 32,222,944 parameters.
  - `model/finetuned_rank_64.pth`: CP-factorized `classifier.0`, 31,927,072 parameters.
- Existing result files in `res/` are useful as historical notes, but new paper tables should be regenerated from scripts.

## Important Methodological Notes

- The old draft in `doc/bare_jrnl.pdf` / `doc/bare_jrnl.tex` is not submission-ready. Treat it as disposable source material.
- Do not claim random CP zero-shot initialization preserves learned geometry. With random factors, zero-shot performance is not evidence of structural preservation.
- Do not claim Rank-64 proves regularization from the existing single run. Reframe as a hypothesis or failure-boundary observation until multi-seed evidence exists.
- Do not rely on the old `thop` benchmark for factorized models. It reported identical MACs/params across ranks, which is inconsistent with checkpoint parameter counts.
- CamVid validation is class-imbalanced: Road, Building, Sky, Tree, and Sidewalk are about 83.1% of validation pixels. Pixel accuracy is weak evidence; report mIoU, per-class IoU, present-class mIoU, and frequency-weighted IoU.
- `Void` is a CamVid class in `class_dict.csv`, not automatically ignored by the old code. Be explicit when ignoring it.

## Current Publication Direction

- Preferred thesis: spectral rank-energy criteria predict when tensorizing large convolutional bottlenecks is structurally plausible, but rank choice, initialization, and hardware realization determine whether compression becomes useful.
- Preferred venue family: applied letter/journal first, with IEEE Signal Processing Letters as the nearest format target. A CV conference/workshop path requires stronger modern-model validation.
- Preferred validation scope: two-model validation. Keep VGG16-FCN32s/CamVid as the controlled bottleneck case and add one modern segmentation track when compute/data access is available.
- Artifact policy: cleaned scripts, result tables, and reproducibility material should be releasable.

## Implemented Experiment Interfaces

- `scripts/evaluate.py`: evaluates dense or CP-factorized checkpoints on CamVid and writes JSON/CSV summaries plus per-class IoU.
- `scripts/profile.py`: manually accounts for parameters and MACs for dense convs and CP `tltorch.FactorizedConv`, with optional latency timing.
- `scripts/benchmark.py`: compatibility wrapper around `scripts/profile.py`; do not restore the old `thop` implementation.
- `scripts/rank_analysis.py`: computes unfolding singular spectra, cumulative energy, threshold ranks, and tail-energy diagnostics for dense convolution tensors.
- `src/smallnet/`: reusable SPL-readiness framework for configs, tensor diagnostics, dotted module replacement, factorization baselines, profiling, datasets, result manifests, and paper assets.
- `configs/spl/`: JSON configs for CamVid/VGG, Pascal VOC/DeepLabV3, cloud profiling, and paper asset generation.
- `scripts/run_spl_camvid.py`: config-driven CamVid/VGG runner for evaluation, profiling, and rank-energy manifests under `res/spl_ready/camvid_vgg/`.
- `scripts/run_spl_voc.py`: config-driven Pascal VOC/DeepLabV3 validation runner. It defaults to no VOC download in config; use `--download` on the cloud GPU.
- `scripts/make_paper_assets.py` and `scripts/make_camvid_qualitative.py`: generate paper tables/figures from manifests and checkpoints.
- `doc/spl_readiness_runbook.md`: command runbook for local smoke checks and cloud GPU result generation.

## Canonical SPL-Ready Results as of 2026-06-30

- Cloud GPU used for current regenerated results: Google Colab Tesla T4, PyTorch `2.11.0+cu128`, CUDA available. Treat latency numbers as T4-specific.
- Important Colab workflow note: sessions are ephemeral and may be reclaimed. Run one checkpoint/phase at a time and download manifests immediately. Large uploads must be chunked at about 25 MB. Use clean tar archives with `COPYFILE_DISABLE=1` and exclude `.DS_Store` / `._*` files.
- CamVid accuracy manifests were regenerated one checkpoint at a time:
  - `res/spl_ready/camvid_vgg_dense/`
  - `res/spl_ready/camvid_vgg_cp_rank_256/`
  - `res/spl_ready/camvid_vgg_cp_rank_128/`
  - `res/spl_ready/camvid_vgg_cp_rank_64/`
  - Combined CSV: `res/spl_ready/camvid_vgg_eval_summary_combined.csv`
  - Combined manifest: `res/spl_ready/camvid_vgg_eval_manifest_combined.json`
- CamVid primary Void-ignored mIoU/FWIoU:
  - Dense val: mIoU `0.3426`, FWIoU `0.7954`; test: mIoU `0.3012`, FWIoU `0.7664`.
  - CP rank-256 val: mIoU `0.2501`, FWIoU `0.7491`; test: mIoU `0.2197`, FWIoU `0.7153`.
  - CP rank-128 val: mIoU `0.2709`, FWIoU `0.7661`; test: mIoU `0.2506`, FWIoU `0.7390`.
  - CP rank-64 val: mIoU `0.2730`, FWIoU `0.7690`; test: mIoU `0.2561`, FWIoU `0.7437`.
- CamVid profile combined CSV: `res/spl_ready/camvid_vgg_profile_summary_combined.csv`.
  - Dense: `134,391,648` params, `71.42G` MACs, `109.58 ms` T4 latency.
  - CP rank-256: `32,814,688` params, `54.66G` MACs, `120.90 ms`.
  - CP rank-128: `32,222,944` params, `54.56G` MACs, `116.07 ms`.
  - CP rank-64: `31,927,072` params, `54.52G` MACs, `113.35 ms`.
  - Interpretation: parameter savings are large and manual MACs decrease, but T4 latency worsens for CP factorized convs.
- CamVid rank-energy diagnostics: `res/spl_ready/camvid_vgg_rank/`.
  - For `classifier.0.weight`, threshold ranks are `R_0.90=2821`, `R_0.95=3348`, `R_0.99=3911`.
  - Necessary CP tail-energy proxies are high at tested ranks: rank-64 `0.7817`, rank-128 `0.7322`, rank-256 `0.6601`.
  - Interpretation: tested ranks are far below the rank-energy diagnostic, consistent with incomplete mIoU recovery.
- Paper assets generated from canonical manifests:
  - `res/spl_ready/paper_assets/table_1_rank_diagnostics.csv`
  - `res/spl_ready/paper_assets/table_2_pareto.csv`
  - `res/spl_ready/paper_assets/figure_1_rank_spectrum.png`
  - `res/spl_ready/paper_assets/figure_2_pareto.png`
  - `res/spl_ready/paper_assets/figure_3_qualitative.png`
- Pascal VOC / DeepLabV3 dense validation:
  - Dense outputs: `res/spl_ready/voc_deeplab_dense/`.
  - Dense pretrained `deeplabv3_resnet50`: mIoU `0.7637`, FWIoU `0.9008`, pixel accuracy `0.9439`, `42,004,074` params, `178.72G` MACs, `137.07 ms` T4 latency.
  - Top ASPP conv rank diagnostics require high ranks: `R_0.90` about `796`, `898`, `949` for the three selected layers.
- Pascal VOC / DeepLabV3 targeted CP stress test:
  - Outputs: `res/spl_ready/voc_deeplab_factorized_one/`.
  - Random CP rank-128 on `classifier.0.convs.1.0` collapses mIoU to `0.0349` while reducing params to `37,581,290`, MACs to `160.04G`, and latency slightly worsens to `137.40 ms`.
  - Interpretation: random CP replacement is not a viable zero-shot modern-model compression method; this supports conservative claims about initialization and recovery training.
- `res/spl_ready/camvid_vgg/camvid_eval_manifest.json` and `res/spl_ready/camvid_vgg_dense_smoke/` are smoke artifacts from zero-batch/local checks. Do not use them as paper evidence.

## Recommended Next Experiments

- For SPL, the current evidence is enough to draft a rigorous negative/diagnostic letter around necessary rank-energy criteria, mIoU recovery failure below diagnostic ranks, and hardware latency mismatch.
- Add CP decomposed initialization or short recovery fine-tuning only if time permits; it would strengthen the paper by separating rank insufficiency from random-initialization damage.
- Add at least 3 seeds for rank 64/128/256 fine-tuning if compute permits. Without this, phrase rank-64/rank-128 comparisons as observed checkpoint behavior, not a robust ordering.
- If targeting WACV instead of SPL, expand the VOC/DeepLabV3 validation into more layers/ranks and add qualitative modern-model failure examples.
