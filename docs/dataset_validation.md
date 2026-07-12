# CamVid Dataset Validation

Run dataset validation before regenerating any paper result. Dense evaluation, CP evaluation, fine-tuning, profiling, and paper assets should not be regenerated unless this stage passes on the real CamVid files.

## Pairing Rule

The CamVid loader builds explicit image and mask maps instead of pairing sorted filenames by position. The default normalized key is the filename stem. The paper config uses the observed CamVid label suffix rule:

```json
"pairing": {
  "image_key": "stem",
  "mask_key": "stem",
  "mask_suffix_to_remove": "_L"
}
```

With this rule, `0001TP_009540.png` matches `0001TP_009540_L.png`. There is no positional fallback. Validation fails on missing masks, unmatched masks, duplicate normalized image keys, or duplicate normalized mask keys.

## Strict RGB Masks

`class_dict.csv` is parsed into class definitions containing class index, class name, and RGB value. Mask conversion is strict by default: every RGB triplet in every mask must appear in `class_dict.csv`.

Unknown RGB values are never mapped to class zero. Under the canonical paper config, an unknown color fails validation and experiment loaders raise an error. The report records the RGB value, pixel count, mask filename, and a bounded set of example coordinates.

Masks are still resized with nearest-neighbor interpolation after RGB-to-index conversion.

## Validation Command

Run this CPU-compatible stage in Colab before any paper experiment:

```bash
uv run python scripts/run_experiment.py \
  --config configs/camvid_vgg_cp_paper.json \
  --stage validate-data \
  --device cpu
```

The command inspects every configured split in `dataset.splits`, normally `train`, `val`, and `test`.

## Generated Files

The validation stage writes:

```text
results/camvid_vgg_cp/dataset_validation_report.json
results/camvid_vgg_cp/dataset_validation_summary.csv
results/camvid_vgg_cp/dataset_class_counts.csv
results/camvid_vgg_cp/dataset_validation_config_used.json
```

The JSON report includes source metadata, split convention, class dictionary path and SHA-256 hash, ignored class metadata, pairing rule, per-split counts, unmatched files, duplicate keys, image and mask dimensions, unknown RGB values, per-class pixel counts, inspected pixel totals, and environment metadata.

The summary CSV has one row per split. The class-count CSV has one row per class per split, including pixel count, pixel proportion, RGB value, and whether the class is excluded from evaluation.

## Failed Reports

A failed validation command returns a nonzero exit status after writing the report whenever practical. Inspect:

- `class_dictionary_errors` for class-count, duplicate RGB, ignore-index, or `Void` metadata failures.
- `splits[*].missing_masks` and `splits[*].unmatched_masks` for pairing failures.
- `splits[*].duplicate_or_ambiguous_keys` for duplicate normalized keys.
- `splits[*].unknown_rgb_values` for strict RGB failures.

The paper configuration currently records `source_url: null` because no canonical source URL is encoded in the repository. Authors must fill in the exact dataset source URL before submission.
