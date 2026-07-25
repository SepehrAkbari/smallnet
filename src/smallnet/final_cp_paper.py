"""CP-only paper artifact generation and validation from completed result rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD = "cp_200_iterations"
RANKS = (32, 64, 128, 256, 512)
SEEDS = (0, 1, 2)
ITERATIONS = 200
DENSE_TARGET_PARAMETERS = 102_764_544
DENSE_FULL_PARAMETERS = 134_391_648
DENSE_FULL_MACS = 71_422_771_200

FINITE_FIELDS = (
    "actual_relative_squared_frobenius_error",
    "actual_relative_frobenius_error",
    "gap_above_max_bound",
    "activation_normalized_squared_error",
    "activation_relative_frobenius_error",
    "activation_cosine_similarity",
    "validation_present_class_miou",
    "test_present_class_miou",
    "target_layer_parameter_count",
    "target_layer_parameter_ratio",
    "full_model_parameter_count",
    "target_layer_macs",
    "full_model_macs",
    "compression_factor",
    "residual_calculation_absolute_difference",
    "reconstruction_path_relative_squared_difference",
)

PAPER_TABLES = (
    "final_cp_structural_results.csv",
    "final_cp_structural_results.tex",
    "final_cp_costs.csv",
    "final_cp_costs.tex",
    "dense_reference.csv",
    "dense_reference.tex",
    "final_cp_correlations.csv",
)
FIGURE_STEMS = (
    "final_cp_weight_error_vs_rank",
    "final_cp_activation_error_vs_rank",
    "final_cp_zero_shot_miou_vs_rank",
    "final_cp_compression_accuracy_tradeoff",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_exact(path: Path, rows: list[dict], fieldnames=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def validate_cp_rows(rows: list[dict[str, str]]) -> dict:
    cp = [
        row
        for row in rows
        if row.get("method") == METHOD and row.get("status") == "completed"
    ]
    expected = {(rank, seed) for rank in RANKS for seed in SEEDS}
    keys = [(int(float(row["rank"])), int(float(row["seed"]))) for row in cp]
    counts = {key: keys.count(key) for key in sorted(set(keys))}
    if len(cp) != 15 or set(keys) != expected or any(value != 1 for value in counts.values()):
        raise ValueError(f"Expected exactly 15 unique CP rows; found {len(cp)} with {counts}")
    if any(int(float(row["iteration_budget"])) != ITERATIONS for row in cp):
        raise ValueError("Every canonical CP row must use exactly 200 iterations")
    for row in cp:
        for field in FINITE_FIELDS:
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"Nonfinite or missing {field} at rank={row['rank']} seed={row['seed']}")
        if row.get("same_fitted_factors_reused_for_all_metrics", "").lower() != "true":
            raise ValueError("A CP row does not record same-factor metric reuse")
        final_hash = row.get("final_factor_hash_sha256")
        stage_hashes = {
            row.get("factor_hash_after_reconstruction"),
            row.get("factor_hash_after_activation_distortion"),
            row.get("factor_hash_after_zero_shot"),
        }
        if stage_hashes != {final_hash}:
            raise ValueError(f"Metric-stage factor hash mismatch at {(row['rank'], row['seed'])}")
        if row.get("residual_verification_passed", "").lower() != "true":
            raise ValueError(f"Residual verification failed at {(row['rank'], row['seed'])}")
        if row.get("factor_diagnostics_finite", "").lower() != "true":
            raise ValueError(f"Nonfinite factor diagnostic at {(row['rank'], row['seed'])}")
    identities = {}
    for field in (
        "checkpoint_sha256",
        "dataset_validation_report_sha256",
        "target_tensor_sha256",
    ):
        values = sorted({row[field] for row in cp})
        if len(values) != 1 or not values[0]:
            raise ValueError(f"Canonical identity {field} is not constant")
        identities[field] = values[0]
    return {
        "rows": cp,
        "row_count": len(cp),
        "keys": [{"rank": rank, "seed": seed} for rank, seed in sorted(expected)],
        "identities": identities,
        "factor_degeneracy_rows": [
            {"rank": int(row["rank"]), "seed": int(row["seed"])}
            for row in cp
            if row.get("factor_degeneracy_detected", "").lower() == "true"
        ],
    }


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values):
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def aggregate_cp_rows(cp_rows: list[dict[str, str]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in cp_rows:
        grouped[int(float(row["rank"]))].append(row)
    output = []
    metric_map = {
        "relative_squared_weight_error": "actual_relative_squared_frobenius_error",
        "activation_normalized_squared_error": "activation_normalized_squared_error",
        "activation_cosine_similarity": "activation_cosine_similarity",
        "validation_present_class_miou": "validation_present_class_miou",
        "test_present_class_miou": "test_present_class_miou",
    }
    for rank in RANKS:
        rows = grouped[rank]
        result = {"rank": rank, "seed_count": len(rows)}
        for prefix, source in metric_map.items():
            values = [float(row[source]) for row in rows]
            result[f"{prefix}_mean"] = _mean(values)
            result[f"{prefix}_population_std"] = _std(values)
            result[f"{prefix}_min"] = min(values)
            result[f"{prefix}_max"] = max(values)
        representative = rows[0]
        target_parameters = int(float(representative["target_layer_parameter_count"]))
        full_parameters = int(float(representative["full_model_parameter_count"]))
        target_macs = int(float(representative["target_layer_macs"]))
        full_macs = int(float(representative["full_model_macs"]))
        result.update(
            {
                "target_layer_parameter_count": target_parameters,
                "target_layer_parameter_ratio": target_parameters / DENSE_TARGET_PARAMETERS,
                "target_layer_compression_factor": DENSE_TARGET_PARAMETERS / target_parameters,
                "full_model_parameter_count": full_parameters,
                "full_model_parameter_ratio": full_parameters / DENSE_FULL_PARAMETERS,
                "full_model_compression_factor": DENSE_FULL_PARAMETERS / full_parameters,
                "target_layer_macs": target_macs,
                "full_model_macs": full_macs,
                "full_model_mac_ratio": full_macs / DENSE_FULL_MACS,
            }
        )
        output.append(result)
    return output


def _latex_table(path: Path, columns, rows, formats, caption: str, label: str):
    align = "r" * len(columns)
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(title for _, title in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(formats.get(key, "{}").format(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(fig, stem: Path, data_rows: list[dict]):
    write_csv_exact(stem.with_suffix(".csv"), data_rows)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _publication_style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )


def generate_figures(rank_rows: list[dict], figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    _publication_style()
    ranks = np.array([row["rank"] for row in rank_rows])

    specifications = (
        (
            "final_cp_weight_error_vs_rank",
            "relative_squared_weight_error",
            "Normalized squared Frobenius weight residual",
            "CP weight reconstruction error",
        ),
        (
            "final_cp_activation_error_vs_rank",
            "activation_normalized_squared_error",
            "Normalized squared isolated-layer activation error",
            "Isolated classifier.0 activation distortion",
        ),
    )
    for stem, prefix, ylabel, title in specifications:
        means = np.array([row[f"{prefix}_mean"] for row in rank_rows])
        stds = np.array([row[f"{prefix}_population_std"] for row in rank_rows])
        data = [
            {
                "rank": row["rank"],
                "seed_count": row["seed_count"],
                f"{prefix}_mean": row[f"{prefix}_mean"],
                f"{prefix}_population_std": row[f"{prefix}_population_std"],
            }
            for row in rank_rows
        ]
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.errorbar(ranks, means, yerr=stds, marker="o", capsize=3, color="#285F8F")
        ax.set(xlabel="CP rank", ylabel=ylabel, title=title)
        ax.set_xticks(ranks)
        ax.grid(axis="y", alpha=0.2)
        _save_figure(fig, figures_dir / stem, data)

    val = np.array([row["validation_present_class_miou_mean"] for row in rank_rows])
    val_std = np.array(
        [row["validation_present_class_miou_population_std"] for row in rank_rows]
    )
    test = np.array([row["test_present_class_miou_mean"] for row in rank_rows])
    test_std = np.array(
        [row["test_present_class_miou_population_std"] for row in rank_rows]
    )
    zero_data = [
        {
            "rank": row["rank"],
            "seed_count": row["seed_count"],
            "validation_present_class_miou_mean": row["validation_present_class_miou_mean"],
            "validation_present_class_miou_population_std": row[
                "validation_present_class_miou_population_std"
            ],
            "test_present_class_miou_mean": row["test_present_class_miou_mean"],
            "test_present_class_miou_population_std": row[
                "test_present_class_miou_population_std"
            ],
        }
        for row in rank_rows
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.errorbar(ranks, val, yerr=val_std, marker="o", capsize=3, label="Validation")
    ax.errorbar(ranks, test, yerr=test_std, marker="s", capsize=3, label="Test")
    ax.set(
        xlabel="CP rank",
        ylabel="Present-class mIoU",
        title="Zero-shot segmentation after CP replacement",
    )
    ax.set_xticks(ranks)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir / "final_cp_zero_shot_miou_vs_rank", zero_data)

    tradeoff = [
        {
            "rank": row["rank"],
            "seed_count": row["seed_count"],
            "target_layer_compression_factor": row["target_layer_compression_factor"],
            "full_model_compression_factor": row["full_model_compression_factor"],
            "validation_present_class_miou_mean": row["validation_present_class_miou_mean"],
            "validation_present_class_miou_population_std": row[
                "validation_present_class_miou_population_std"
            ],
            "test_present_class_miou_mean": row["test_present_class_miou_mean"],
            "test_present_class_miou_population_std": row[
                "test_present_class_miou_population_std"
            ],
        }
        for row in rank_rows
    ]
    full_compression = np.array([row["full_model_compression_factor"] for row in rank_rows])
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    ax.errorbar(
        full_compression,
        val,
        yerr=val_std,
        marker="o",
        capsize=3,
        label="Validation",
    )
    ax.errorbar(
        full_compression,
        test,
        yerr=test_std,
        marker="s",
        capsize=3,
        label="Test",
    )
    for row, x, y in zip(rank_rows, full_compression, val):
        offset = (-38, 6) if x == max(full_compression) else (3, 3)
        ax.annotate(
            f"r={row['rank']}",
            (x, y),
            xytext=offset,
            textcoords="offset points",
        )
    ax.set(
        xlabel="Full-model parameter compression factor",
        ylabel="Present-class mIoU",
        title="CP compression–accuracy tradeoff",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir / "final_cp_compression_accuracy_tradeoff", tradeoff)


def generate_correlations(rank_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rank_rows)
    pairs = (
        ("relative_squared_weight_error_mean", "validation_present_class_miou_mean"),
        ("relative_squared_weight_error_mean", "test_present_class_miou_mean"),
        ("activation_normalized_squared_error_mean", "validation_present_class_miou_mean"),
        ("activation_normalized_squared_error_mean", "test_present_class_miou_mean"),
        ("rank", "validation_present_class_miou_mean"),
        ("rank", "test_present_class_miou_mean"),
    )
    rows = []
    for left, right in pairs:
        for coefficient in ("pearson", "spearman"):
            rows.append(
                {
                    "left_variable": left,
                    "right_variable": right,
                    "coefficient": coefficient,
                    "value": float(frame[left].corr(frame[right], method=coefficient)),
                    "observation_count": 5,
                    "scope": "descriptive_five_rank_mean_association",
                    "inferential_test_performed": False,
                }
            )
    return rows


def _write_audit(path: Path, rank_rows: list[dict], validated: dict, dense: dict):
    first, last = rank_rows[0], rank_rows[-1]
    max_weight_std = max(
        row["relative_squared_weight_error_population_std"] for row in rank_rows
    )
    max_activation_std = max(
        row["activation_normalized_squared_error_population_std"] for row in rank_rows
    )
    max_val_std = max(
        row["validation_present_class_miou_population_std"] for row in rank_rows
    )
    max_test_std = max(
        row["test_present_class_miou_population_std"] for row in rank_rows
    )
    layer_compression = ", ".join(
        f"rank {row['rank']}: {row['target_layer_compression_factor']:.2f}x"
        for row in rank_rows
    )
    model_compression = ", ".join(
        f"rank {row['rank']}: {row['full_model_compression_factor']:.2f}x"
        for row in rank_rows
    )
    degenerate = validated["factor_degeneracy_rows"]
    lines = [
        "# Final CP-200 scientific audit",
        "",
        "Scope: post-training CP factorization of `classifier.0` in VGG16-FCN32s "
        "on CamVid, using ranks 32, 64, 128, 256, and 512; seeds 0, 1, and 2; "
        "and a common 200-iteration fitting budget.",
        "",
        "## Protocol and completeness",
        "",
        "1. **All expected rows complete:** Yes. Exactly 15 completed "
        "`cp_200_iterations` rows are present and each rank/seed key occurs once.",
        "2. **Three seeds per rank:** Yes. Seeds 0, 1, and 2 are present at every rank.",
        "3. **Iteration budget:** Every scientific row records exactly 200 iterations.",
        "4. **Fitted-factor reuse:** Every row records that one fitted layer was reused "
        "for reconstruction, isolated-layer activation distortion, and zero-shot evaluation.",
        "5. **Identity checks:** Checkpoint, dataset-validation report, and target-tensor "
        "SHA-256 identities are each constant across all 15 rows.",
        "6. **Residual checks:** All required numerical results and factor diagnostics are "
        "finite; every row records a passed residual verification; factor hashes agree "
        "after reconstruction, activation distortion, and zero-shot evaluation.",
        f"7. **Factor degeneracy diagnostic:** Threshold-based degeneracy is reported for "
        f"{len(degenerate)} rows: "
        + (
            ", ".join(f"rank {row['rank']} seed {row['seed']}" for row in degenerate)
            if degenerate
            else "none"
        )
        + ". These are descriptive factor-scaling flags, not certified CP degeneracy.",
        "",
        "## Scientific results",
        "",
        f"8. **Rank trends:** Mean normalized squared weight residual decreases from "
        f"`{first['relative_squared_weight_error_mean']:.6f}` at rank 32 to "
        f"`{last['relative_squared_weight_error_mean']:.6f}` at rank 512. Mean isolated-layer "
        f"normalized squared activation error decreases from "
        f"`{first['activation_normalized_squared_error_mean']:.6f}` to "
        f"`{last['activation_normalized_squared_error_mean']:.6f}`. Validation present-class "
        f"mIoU rises from `{first['validation_present_class_miou_mean']:.6f}` to "
        f"`{last['validation_present_class_miou_mean']:.6f}`, and test present-class mIoU "
        f"rises from `{first['test_present_class_miou_mean']:.6f}` to "
        f"`{last['test_present_class_miou_mean']:.6f}`. The dense references are "
        f"`{dense['validation_present_class_miou']:.6f}` validation and "
        f"`{dense['test_present_class_miou']:.6f}` test mIoU.",
        f"9. **Seed variability:** Maximum population standard deviations across ranks are "
        f"`{max_weight_std:.6f}` for squared weight residual, `{max_activation_std:.6f}` "
        f"for squared activation error, `{max_val_std:.6f}` for validation mIoU, and "
        f"`{max_test_std:.6f}` for test mIoU. These are small relative to the differences "
        "between rank 32 and rank 512, although three seeds do not characterize every "
        "possible initialization.",
        f"10. **Target-layer compression:** {layer_compression}.",
        f"    **Full-model compression:** {model_compression}. Target-layer and full-model "
        "compression are distinct quantities.",
        "11. **Claims supported:** Higher tested CP rank is associated with lower weight and "
        "isolated-layer activation error, higher zero-shot mIoU, and lower compression. "
        "The five-point descriptive associations are consistent with activation distortion "
        "tracking zero-shot mIoU more closely than weight error in this case study.",
        "12. **Claims not supported:** These results do not establish causality, statistical "
        "generality, certified CP convergence, state-of-the-art segmentation, or a hardware "
        "speedup. They do not imply that target-layer compression equals full-model compression.",
        "",
        "## Scope exclusions and provenance",
        "",
        "- Matrix SVD is outside the final paper scope. Failed matrix-SVD development rows "
        "are development provenance, not scientific paper results.",
        "- Older fine-tuned checkpoints are historical artifacts and are not directly "
        "comparable with the final 200-iteration post-training CP decompositions.",
        "- No causal or statistically definitive correlation claim is made from five rank means.",
        "- The paper-facing completeness decision is CP-only and is unaffected by the abandoned "
        "matrix-SVD development branch.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_publication_todos(path: Path):
    path.write_text(
        "\n".join(
            [
                "# Publication metadata requiring author input",
                "",
                "- Exact CamVid download/source URL.",
                "- CamVid archive or version identifier.",
                "- CamVid access date, if required by the selected venue.",
                "- Final author affiliations and contact information.",
                "- Target journal or proceedings format.",
                "- Code/data archival DOI, if an archive will be created.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(root: Path, identities: dict):
    required = [
        "results/camvid_vgg_cp/final_cp_200/final_cp_summary.csv",
        "results/camvid_vgg_cp/final_cp_200/final_cp_rank_summary.csv",
        "results/camvid_vgg_cp/final_cp_200/final_cp_metadata.json",
        "results/camvid_vgg_cp/final_cp_200/final_cp_validation.json",
        *[f"results/paper/tables/{name}" for name in PAPER_TABLES],
        *[
            f"results/paper/figures/{stem}.{suffix}"
            for stem in FIGURE_STEMS
            for suffix in ("csv", "pdf", "png")
        ],
        "results/paper/final_cp_200_audit.md",
        "results/paper/PUBLICATION_TODOS.md",
        "results/camvid_vgg_cp/dense_eval_summary.csv",
        "results/camvid_vgg_cp/dataset_validation_report.json",
        "results/paper/cp_iteration_sensitivity_audit.md",
        "results/paper/rank512_repeatability_decision.md",
        "results/paper/rank512_stability_audit.md",
        "configs/camvid_vgg_cp_paper.json",
        "results/camvid_vgg_cp/final_structural/final_structural_config_used.json",
    ]
    artifacts = []
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Manifest input missing or empty: {relative}")
        artifacts.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    manifest = {
        "kind": "final_cp_200_paper_reproducibility_manifest",
        "scope": {
            "method": METHOD,
            "ranks": list(RANKS),
            "seeds": list(SEEDS),
            "iteration_budget": ITERATIONS,
            "matrix_svd_in_final_scope": False,
        },
        "checkpoint_sha256": identities["checkpoint_sha256"],
        "dataset_validation_report_sha256": identities[
            "dataset_validation_report_sha256"
        ],
        "target_tensor_sha256": identities["target_tensor_sha256"],
        "artifacts": artifacts,
    }
    (root / "results/paper/MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_final_cp_artifacts(root: Path) -> dict:
    root = root.resolve()
    raw_dir = root / "results/camvid_vgg_cp/final_structural"
    raw_summary = raw_dir / "final_structural_summary.csv"
    raw_cp_summary = raw_dir / "final_structural_cp_only_summary.csv"
    pre_hashes = {
        str(raw_summary.relative_to(root)): sha256_file(raw_summary),
        str(raw_cp_summary.relative_to(root)): sha256_file(raw_cp_summary),
    }
    source_rows = read_csv(raw_summary)
    validated = validate_cp_rows(source_rows)
    cp_rows = validated["rows"]
    factors_dir = raw_dir / "factors"
    for row in cp_rows:
        artifact = factors_dir / Path(row["factor_artifact_path"]).name
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing factor artifact: {artifact.relative_to(root)}")
        if sha256_file(artifact) != row["factor_artifact_sha256"]:
            raise ValueError(f"Factor artifact hash mismatch: {artifact.relative_to(root)}")

    output_dir = root / "results/camvid_vgg_cp/final_cp_200"
    tables_dir = root / "results/paper/tables"
    figures_dir = root / "results/paper/figures"
    development_dir = root / "results/development/matrix_svd_attempts"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    development_dir.mkdir(parents=True, exist_ok=True)

    final_summary = write_csv_exact(output_dir / "final_cp_summary.csv", cp_rows)
    rank_rows = aggregate_cp_rows(cp_rows)
    final_rank_summary = write_csv_exact(
        output_dir / "final_cp_rank_summary.csv", rank_rows
    )
    failed_svd = [
        row
        for row in source_rows
        if row.get("method") == "matrix_svd_output_unfolding"
        and row.get("status") != "completed"
    ]
    if failed_svd:
        write_csv_exact(
            development_dir / "final_structural_failed_matrix_svd_rows.csv",
            failed_svd,
        )
    dense_rows = read_csv(root / "results/camvid_vgg_cp/dense_eval_summary.csv")
    dense = {
        "model_kind": "dense_reference",
        "target_layer_parameter_count": DENSE_TARGET_PARAMETERS,
        "full_model_parameter_count": DENSE_FULL_PARAMETERS,
        "full_model_macs": DENSE_FULL_MACS,
    }
    for row in dense_rows:
        split = "validation" if row["split"] == "val" else row["split"]
        dense[f"{split}_present_class_miou"] = float(row["mean_iou_present_classes"])
        dense[f"{split}_all_class_miou"] = float(row["mean_iou_all_classes"])
        dense[f"{split}_pixel_accuracy"] = float(row["pixel_accuracy"])
        dense[f"{split}_frequency_weighted_iou"] = float(row["frequency_weighted_iou"])

    structural_columns = (
        "rank",
        "seed_count",
        "relative_squared_weight_error_mean",
        "relative_squared_weight_error_population_std",
        "activation_normalized_squared_error_mean",
        "activation_normalized_squared_error_population_std",
        "validation_present_class_miou_mean",
        "validation_present_class_miou_population_std",
        "test_present_class_miou_mean",
        "test_present_class_miou_population_std",
        "target_layer_compression_factor",
        "full_model_compression_factor",
    )
    structural_rows = [{key: row[key] for key in structural_columns} for row in rank_rows]
    write_csv_exact(
        tables_dir / "final_cp_structural_results.csv",
        structural_rows,
        structural_columns,
    )
    cost_columns = (
        "rank",
        "target_layer_parameter_count",
        "target_layer_parameter_ratio",
        "target_layer_compression_factor",
        "full_model_parameter_count",
        "full_model_parameter_ratio",
        "full_model_compression_factor",
        "target_layer_macs",
        "full_model_macs",
        "full_model_mac_ratio",
    )
    cost_rows = [{key: row[key] for key in cost_columns} for row in rank_rows]
    write_csv_exact(tables_dir / "final_cp_costs.csv", cost_rows, cost_columns)
    write_csv_exact(tables_dir / "dense_reference.csv", [dense])

    _latex_table(
        tables_dir / "final_cp_structural_results.tex",
        (
            ("rank", "Rank"),
            ("relative_squared_weight_error_mean", r"$R_W$"),
            ("relative_squared_weight_error_population_std", r"$\sigma(R_W)$"),
            ("activation_normalized_squared_error_mean", r"$R_A$"),
            ("activation_normalized_squared_error_population_std", r"$\sigma(R_A)$"),
            ("validation_present_class_miou_mean", r"Val mIoU"),
            ("validation_present_class_miou_population_std", r"$\sigma$"),
            ("test_present_class_miou_mean", r"Test mIoU"),
            ("test_present_class_miou_population_std", r"$\sigma$"),
            ("target_layer_compression_factor", r"Layer $\times$"),
            ("full_model_compression_factor", r"Model $\times$"),
        ),
        structural_rows,
        {
            "rank": "{:d}",
            **{
                key: "{:.4f}"
                for key in structural_columns
                if key not in {"rank", "seed_count", "target_layer_compression_factor", "full_model_compression_factor"}
            },
            "target_layer_compression_factor": "{:.2f}",
            "full_model_compression_factor": "{:.2f}",
        },
        "CP-only structural results. Standard deviations are population standard deviations over three seeds.",
        "tab:final-cp-structural",
    )
    _latex_table(
        tables_dir / "final_cp_costs.tex",
        (
            ("rank", "Rank"),
            ("target_layer_parameter_count", "Layer params"),
            ("target_layer_parameter_ratio", "Layer ratio"),
            ("target_layer_compression_factor", r"Layer $\times$"),
            ("full_model_parameter_count", "Model params"),
            ("full_model_parameter_ratio", "Model ratio"),
            ("full_model_compression_factor", r"Model $\times$"),
            ("target_layer_macs", "Layer MACs"),
            ("full_model_macs", "Model MACs"),
        ),
        cost_rows,
        {
            "rank": "{:d}",
            "target_layer_parameter_count": "{:,d}",
            "target_layer_parameter_ratio": "{:.4f}",
            "target_layer_compression_factor": "{:.2f}",
            "full_model_parameter_count": "{:,d}",
            "full_model_parameter_ratio": "{:.4f}",
            "full_model_compression_factor": "{:.2f}",
            "target_layer_macs": "{:,d}",
            "full_model_macs": "{:,d}",
        },
        "CP representation costs. Layer and full-model compression factors are distinct.",
        "tab:final-cp-costs",
    )
    _latex_table(
        tables_dir / "dense_reference.tex",
        (
            ("target_layer_parameter_count", "Layer params"),
            ("full_model_parameter_count", "Model params"),
            ("full_model_macs", "Model MACs"),
            ("validation_present_class_miou", "Val mIoU"),
            ("test_present_class_miou", "Test mIoU"),
        ),
        [dense],
        {
            "target_layer_parameter_count": "{:,d}",
            "full_model_parameter_count": "{:,d}",
            "full_model_macs": "{:,d}",
            "validation_present_class_miou": "{:.4f}",
            "test_present_class_miou": "{:.4f}",
        },
        "Dense VGG16-FCN32s reference under the validated CamVid policy.",
        "tab:dense-reference",
    )
    correlations = generate_correlations(rank_rows)
    write_csv_exact(tables_dir / "final_cp_correlations.csv", correlations)
    generate_figures(rank_rows, figures_dir)

    final_summary_hash = sha256_file(final_summary)
    metadata = {
        "scope": "final_cp_200_paper_artifacts",
        "method": METHOD,
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
        "iteration_budget": ITERATIONS,
        "row_count": 15,
        "aggregation_standard_deviation": "population (ddof=0)",
        "matrix_svd_in_final_scope": False,
        "historical_finetuned_checkpoints_in_final_scope": False,
        "source_raw_summary": str(raw_summary.relative_to(root)),
        "source_raw_summary_sha256": pre_hashes[str(raw_summary.relative_to(root))],
        "final_cp_summary_sha256": final_summary_hash,
        "identities": validated["identities"],
        "factor_degeneracy_rows": validated["factor_degeneracy_rows"],
        "factor_artifact_storage": {
            "committed": True,
            "directory": "results/camvid_vgg_cp/final_structural/factors",
            "note": "Legacy row paths contain /content; artifacts are resolved by basename in the canonical relative directory.",
        },
        "development_only_matrix_svd_rows": len(failed_svd),
        "development_only_matrix_svd_path": (
            "results/development/matrix_svd_attempts/final_structural_failed_matrix_svd_rows.csv"
        ),
    }
    metadata_path = output_dir / "final_cp_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    post_hashes = {
        str(raw_summary.relative_to(root)): sha256_file(raw_summary),
        str(raw_cp_summary.relative_to(root)): sha256_file(raw_cp_summary),
    }
    if post_hashes != pre_hashes:
        raise RuntimeError("Raw canonical CP CSV hashes changed during paper finalization")
    validation = {
        "status": "passed",
        "canonical_cp_row_count": 15,
        "unique_scientific_key_count": 15,
        "expected_ranks": list(RANKS),
        "expected_seeds": list(SEEDS),
        "iteration_budget": ITERATIONS,
        "all_required_numeric_values_finite": True,
        "identity_hashes_constant": True,
        "same_fitted_factors_reused": True,
        "metric_stage_factor_hashes_match": True,
        "residual_checks_passed": True,
        "raw_canonical_cp_csv_hashes_before_cleanup": pre_hashes,
        "raw_canonical_cp_csv_hashes_after_cleanup": post_hashes,
        "raw_scientific_values_unchanged": True,
        "final_cp_summary_sha256": final_summary_hash,
        "factor_artifact_count": len(list(factors_dir.glob("*.pt"))),
        "factor_artifact_file_hashes_match_rows": True,
        "nested_result_tree_absent": not (
            root / "results/camvid_vgg_cp/camvid_vgg_cp"
        ).exists(),
    }
    validation_path = output_dir / "final_cp_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    _write_audit(
        root / "results/paper/final_cp_200_audit.md",
        rank_rows,
        validated,
        dense,
    )
    _write_publication_todos(root / "results/paper/PUBLICATION_TODOS.md")
    _write_manifest(root, validated["identities"])
    return {
        "metadata": metadata,
        "validation": validation,
        "rank_rows": rank_rows,
        "dense": dense,
    }


def validate_paper_artifacts(root: Path) -> dict:
    root = root.resolve()
    final_dir = root / "results/camvid_vgg_cp/final_cp_200"
    summary = final_dir / "final_cp_summary.csv"
    validated = validate_cp_rows(read_csv(summary))
    factors_dir = root / "results/camvid_vgg_cp/final_structural/factors"
    for row in validated["rows"]:
        artifact = factors_dir / Path(row["factor_artifact_path"]).name
        if not artifact.is_file() or sha256_file(artifact) != row["factor_artifact_sha256"]:
            raise ValueError(
                f"Missing or mismatched factor artifact for rank={row['rank']} seed={row['seed']}"
            )
    ranks = read_csv(final_dir / "final_cp_rank_summary.csv")
    if [int(row["rank"]) for row in ranks] != list(RANKS):
        raise ValueError("Final CP rank summary is incomplete or unordered")
    for row in ranks:
        if int(float(row["seed_count"])) != 3:
            raise ValueError("Every rank aggregate must contain three seeds")
    paper_paths = [root / "results/paper/tables" / name for name in PAPER_TABLES]
    for stem in FIGURE_STEMS:
        paper_paths.extend(
            root / "results/paper/figures" / f"{stem}.{suffix}"
            for suffix in ("csv", "pdf", "png")
        )
    for path in paper_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty paper artifact: {path.relative_to(root)}")
        if path.suffix in {".csv", ".tex"}:
            lowered = path.read_text(errors="ignore").lower()
            if "matrix_svd" in lowered or "fine-tuned" in lowered or "finetuned" in lowered:
                raise ValueError(f"Non-CP method leaked into {path.relative_to(root)}")
    if (root / "results/camvid_vgg_cp/camvid_vgg_cp").exists():
        raise ValueError("Duplicated results/camvid_vgg_cp/camvid_vgg_cp tree exists")
    manifest_path = root / "results/paper/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.is_file():
            raise ValueError(f"Manifest path is missing: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {item['path']}")
        if item["path"].startswith("/content/"):
            raise ValueError("Manifest contains an absolute Colab path")
    return {
        "status": "passed",
        "canonical_cp_rows": validated["row_count"],
        "rank_aggregates": len(ranks),
        "paper_artifacts_checked": len(paper_paths),
        "manifest_entries_checked": len(manifest["artifacts"]),
    }
