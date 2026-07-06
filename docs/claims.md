# Claims Guide

Use this note to keep the paper framing diagnostic and mathematically careful.

## Supported Claims

- CP factorization gives extreme target-layer parameter reduction for the
  dominant VGG16-FCN32s classifier convolution.
- The existing fine-tuned rank-64 CP checkpoint keeps a meaningful fraction of
  dense test mIoU while using about `0.292%` of the target-layer parameters.
  In the final paper table, dense test mIoU is about `0.3012`, while existing
  fine-tuned rank-64 test mIoU is about `0.2561`.
- On the final Colab Tesla T4 batch-1 profile, latency improves for the
  evaluated existing fine-tuned CP models relative to the dense model.
- Rank-energy diagnostics show substantial channel-mode tail energy at low
  ranks, especially in the output-channel and input-channel unfoldings.

## Qualified Claims

- Rank 64 performed best among the existing fine-tuned checkpoints, but this is
  single-seed/checkpoint evidence rather than a robust ordering across repeated
  training runs.
- Zero-shot post-training CP performance improves with rank for ranks `64`,
  `128`, and `256` in the available results.
- Fine-tuning behavior is not monotone in rank in the available checkpoint set.
  Treat this as observed checkpoint behavior, not a general law.

## Claims To Avoid

- Do not claim state-of-the-art semantic segmentation.
- Do not claim CP compression is generally optimal.
- Do not claim rank-energy predicts exact mIoU.
- Do not claim the results generalize beyond VGG16-FCN32s/CamVid without more
  experiments.
