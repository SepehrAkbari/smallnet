# Final CP-200 scientific audit

Scope: post-training CP factorization of `classifier.0` in VGG16-FCN32s on CamVid, using ranks 32, 64, 128, 256, and 512; seeds 0, 1, and 2; and a common 200-iteration fitting budget.

## Protocol and completeness

1. **All expected rows complete:** Yes. Exactly 15 completed `cp_200_iterations` rows are present and each rank/seed key occurs once.
2. **Three seeds per rank:** Yes. Seeds 0, 1, and 2 are present at every rank.
3. **Iteration budget:** Every scientific row records exactly 200 iterations.
4. **Fitted-factor reuse:** Every row records that one fitted layer was reused for reconstruction, isolated-layer activation distortion, and zero-shot evaluation.
5. **Identity checks:** Checkpoint, dataset-validation report, and target-tensor SHA-256 identities are each constant across all 15 rows.
6. **Residual checks:** All required numerical results and factor diagnostics are finite; every row records a passed residual verification; factor hashes agree after reconstruction, activation distortion, and zero-shot evaluation.
7. **Factor degeneracy diagnostic:** Threshold-based degeneracy is reported for 3 rows: rank 512 seed 0, rank 512 seed 1, rank 512 seed 2. These are descriptive factor-scaling flags, not certified CP degeneracy.

## Scientific results

8. **Rank trends:** Mean normalized squared weight residual decreases from `0.944835` at rank 32 to `0.780846` at rank 512. Mean isolated-layer normalized squared activation error decreases from `0.332907` to `0.174798`. Validation present-class mIoU rises from `0.220626` to `0.309271`, and test present-class mIoU rises from `0.202023` to `0.287181`. The dense references are `0.366259` validation and `0.322016` test mIoU.
9. **Seed variability:** Maximum population standard deviations across ranks are `0.000279` for squared weight residual, `0.001182` for squared activation error, `0.002815` for validation mIoU, and `0.002380` for test mIoU. These are small relative to the differences between rank 32 and rank 512, although three seeds do not characterize every possible initialization.
10. **Target-layer compression:** rank 32: 675.94x, rank 64: 342.59x, rank 128: 172.47x, rank 256: 86.53x, rank 512: 43.34x.
    **Full-model compression:** rank 32: 4.23x, rank 64: 4.21x, rank 128: 4.17x, rank 256: 4.10x, rank 512: 3.95x. Target-layer and full-model compression are distinct quantities.
11. **Claims supported:** Higher tested CP rank is associated with lower weight and isolated-layer activation error, higher zero-shot mIoU, and lower compression. The five-point descriptive associations are consistent with activation distortion tracking zero-shot mIoU more closely than weight error in this case study.
12. **Claims not supported:** These results do not establish causality, statistical generality, certified CP convergence, state-of-the-art segmentation, or a hardware speedup. They do not imply that target-layer compression equals full-model compression.

## Scope exclusions and provenance

- Matrix SVD is outside the final paper scope. Failed matrix-SVD development rows are development provenance, not scientific paper results.
- Older fine-tuned checkpoints are historical artifacts and are not directly comparable with the final 200-iteration post-training CP decompositions.
- No causal or statistically definitive correlation claim is made from five rank means.
- The paper-facing completeness decision is CP-only and is unaffected by the abandoned matrix-SVD development branch.
