# Final-structural matrix-SVD tail verification

The final-structural matrix-SVD row is verified against the singular values from
the exact output-mode SVD used to construct that row. The scientific equality
check is

```text
direct residual of replacement.composed_kernel()
    == tail energy from the current SVD singular values
```

Both sides are accumulated on the CPU in float64. The composed kernel is also
compared in float64 with the direct truncated matrix
`U[:, :r] @ diag(S[:r]) @ Vh[:r]`, after reshaping.

The output-mode tail in the earlier reconstruction diagnostic remains in the
row as provenance. It is not the acceptance target for a replacement produced
by a separate SVD call.

## Rank-32 legacy discrepancy

For `classifier.0.weight`, with output unfolding shape `(4096, 25088)`, the
current independent float64 calculation gives:

```text
current same-SVD tail squared                 0.8221586712571629
direct truncated-SVD residual squared         0.8221586712571632
absolute current-tail/direct-residual error   3.3306690738754696e-16
stored reconstruction diagnostic tail squared 0.8222565439437998
absolute current/stored-tail difference       0.0000978726866369
```

The stored artifact was produced on a Tesla T4 with PyTorch 2.12.0 and records
the output-mode matrix SVD device as CUDA. Its cumulative energy was accumulated
after converting the singular values to float64, so the discrepancy is not
caused by rounding a cumulative-energy CSV. The stored rank index is also
correct: using the next singular value would give a tail near
`0.8204609306242667`, not the current value. The stored checkpoint hash, target
layer, tensor shape, and rank join agree with the current experiment.

The evidence therefore localizes the discrepancy to singular values produced
by a separate older float32 CUDA SVD call/backend path, rather than to the
truncated reconstruction, residual formula, rank indexing, a stale rank join,
or a different tensor shape. The old reconstruction artifact is retained
unchanged.

## Recorded fields

Each completed matrix-SVD row records the current and stored tails, both
absolute comparisons, verification tolerance and flags, SVD device and dtype,
float64 residual-evaluation path, and the maximum-absolute and relative
Frobenius differences between the composed kernel and direct truncated SVD.
A stored-tail disagreement is visible in the table and metadata but does not
reject a row whose current same-SVD checks pass.
