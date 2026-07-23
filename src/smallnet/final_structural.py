'''Final 200-iteration structural comparison utilities.'''

import hashlib
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.smallnet.cp_iteration_sensitivity import _cp_components_hash
from src.smallnet.factorization import factorized_conv_from_conv
from src.smallnet.modules import get_module
from src.smallnet.results import write_csv
from src.smallnet.structural import _normalize_nonnegative_integer


FINAL_CP_METHOD = "cp_200_iterations"
FINAL_SVD_METHOD = "matrix_svd_output_unfolding"
FINAL_ITERATION_BUDGET = 200


def final_structural_key(row):
    method = str(row.get("method", "")).strip()
    if method not in {FINAL_CP_METHOD, FINAL_SVD_METHOD}:
        raise ValueError(f"unsupported final structural method {method!r}")
    rank = _normalize_nonnegative_integer(row.get("rank"), "rank", positive=True)
    if method == FINAL_CP_METHOD:
        seed = _normalize_nonnegative_integer(row.get("seed"), "seed")
        budget = _normalize_nonnegative_integer(
            row.get("iteration_budget"), "iteration_budget", positive=True
        )
        if budget != FINAL_ITERATION_BUDGET:
            raise ValueError(
                f"final CP rows must use {FINAL_ITERATION_BUDGET} iterations, got {budget}"
            )
    else:
        seed_value = row.get("seed", "")
        if seed_value not in ("", None):
            if isinstance(seed_value, float) and math.isnan(seed_value):
                seed_value = ""
            else:
                raise ValueError("matrix-SVD rows must not have a CP seed")
        seed = ""
        budget_value = row.get("iteration_budget", "")
        if budget_value not in ("", None):
            if not (isinstance(budget_value, float) and math.isnan(budget_value)):
                raise ValueError("matrix-SVD rows must not have an iteration budget")
        budget = ""
    return method, rank, seed, budget


def normalize_final_structural_rows(rows, context="final structural rows", *, warn=True):
    normalized_by_key = {}
    rejected = []
    diagnostics = []
    for index, original in enumerate(rows):
        row = dict(original)
        try:
            method, rank, seed, budget = final_structural_key(row)
            row.update(
                {
                    "method": method,
                    "rank": rank,
                    "seed": seed,
                    "iteration_budget": budget,
                }
            )
        except (TypeError, ValueError) as exc:
            rejected.append(row)
            diagnostics.append(
                {
                    "context": context,
                    "row_index": index,
                    "reason": str(exc),
                    "method": row.get("method", ""),
                    "rank": row.get("rank", ""),
                    "seed": row.get("seed", ""),
                    "iteration_budget": row.get("iteration_budget", ""),
                }
            )
            continue
        key = final_structural_key(row)
        previous = normalized_by_key.get(key)
        if previous is not None:
            diagnostics.append(
                {
                    "context": context,
                    "row_index": index,
                    "reason": f"duplicate scientific row for key {key!r}; retained one row",
                }
            )
            if previous.get("status") == "completed" and row.get("status") != "completed":
                continue
        normalized_by_key[key] = row
    if diagnostics and warn:
        warnings.warn(
            f"{context}: rejected or deduplicated {len(diagnostics)} row(s)",
            RuntimeWarning,
            stacklevel=2,
        )
    normalized = sorted(
        normalized_by_key.values(),
        key=lambda row: (
            int(row["rank"]),
            row["method"],
            -1 if row["seed"] == "" else int(row["seed"]),
        ),
    )
    return normalized, rejected, diagnostics


def fitted_factor_hash(module):
    return _cp_components_hash(
        module.weight.weights.detach().cpu(),
        [factor.detach().cpu() for factor in module.weight.factors],
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_factor_artifact(
    path,
    fitted,
    *,
    rank,
    seed,
    iteration_budget,
    initializer,
    initialization_hash,
    checkpoint_hash,
    dataset_validation_hash,
    target_tensor_hash,
    decomposition_runtime_seconds,
    mttkrp_implementation="",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    factor_hash = fitted_factor_hash(fitted)
    payload = {
        "kind": "smallnet_cp_fitted_layer",
        "rank": int(rank),
        "seed": int(seed),
        "iteration_budget": int(iteration_budget),
        "initializer": initializer,
        "initialization_hash_sha256": initialization_hash,
        "final_factor_hash_sha256": factor_hash,
        "checkpoint_sha256": checkpoint_hash,
        "dataset_validation_report_sha256": dataset_validation_hash,
        "target_tensor_sha256": target_tensor_hash,
        "decomposition_runtime_seconds": float(decomposition_runtime_seconds),
        "mttkrp_implementation": mttkrp_implementation,
        "state_dict": {
            key: value.detach().cpu() for key, value in fitted.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "factor_artifact_path": str(path),
        "factor_artifact_sha256": sha256_file(path),
        "final_factor_hash_sha256": factor_hash,
    }


def load_factor_artifact(
    path,
    dense_conv,
    *,
    expected_rank,
    expected_seed,
    expected_iteration_budget,
    expected_initializer,
    expected_checkpoint_hash,
    expected_dataset_validation_hash,
    expected_target_tensor_hash,
):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "rank": int(expected_rank),
        "seed": int(expected_seed),
        "iteration_budget": int(expected_iteration_budget),
        "initializer": expected_initializer,
        "checkpoint_sha256": expected_checkpoint_hash,
        "dataset_validation_report_sha256": expected_dataset_validation_hash,
        "target_tensor_sha256": expected_target_tensor_hash,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Factor artifact identity mismatch for {field}: "
                f"{payload.get(field)!r} != {value!r}"
            )
    fitted = factorized_conv_from_conv(
        dense_conv,
        rank=expected_rank,
        factorization="cp",
        init=expected_initializer,
        n_iter_max=0,
    )
    fitted.load_state_dict(payload["state_dict"])
    actual_factor_hash = fitted_factor_hash(fitted)
    if actual_factor_hash != payload["final_factor_hash_sha256"]:
        raise RuntimeError("Loaded factor values do not match the saved factor hash")
    return fitted, payload, {
        "factor_artifact_path": str(path),
        "factor_artifact_sha256": sha256_file(path),
        "final_factor_hash_sha256": actual_factor_hash,
    }


class ActivationDistortionAccumulator:
    '''Accumulate global and per-example activation statistics without batch-ratio bias.'''

    def __init__(self):
        self.dense_squared = 0.0
        self.compressed_squared = 0.0
        self.residual_squared = 0.0
        self.dot = 0.0
        self.element_count = 0
        self.example_count = 0
        self.example_ratio_sum = 0.0
        self.example_ratio_squared_sum = 0.0
        self.example_ratio_min = float("inf")
        self.example_ratio_max = float("-inf")

    def update(self, dense, compressed):
        if dense.shape != compressed.shape:
            raise ValueError(
                f"Activation shapes differ: {tuple(dense.shape)} != {tuple(compressed.shape)}"
            )
        dense = dense.detach().to(torch.float64)
        compressed = compressed.detach().to(torch.float64)
        if not torch.isfinite(dense).all() or not torch.isfinite(compressed).all():
            raise ValueError("Activation distortion received NaN or Inf")
        residual = dense - compressed
        batch = int(dense.shape[0])
        dense_flat = dense.reshape(batch, -1)
        compressed_flat = compressed.reshape(batch, -1)
        residual_flat = residual.reshape(batch, -1)
        dense_sq = torch.sum(dense_flat.square(), dim=1)
        residual_sq = torch.sum(residual_flat.square(), dim=1)
        ratios = residual_sq / torch.clamp(
            dense_sq, min=torch.finfo(torch.float64).tiny
        )
        self.dense_squared += float(torch.sum(dense_sq))
        self.compressed_squared += float(torch.sum(compressed_flat.square()))
        self.residual_squared += float(torch.sum(residual_sq))
        self.dot += float(torch.sum(dense_flat * compressed_flat))
        self.element_count += dense.numel()
        self.example_count += batch
        self.example_ratio_sum += float(torch.sum(ratios))
        self.example_ratio_squared_sum += float(torch.sum(ratios.square()))
        self.example_ratio_min = min(self.example_ratio_min, float(torch.min(ratios)))
        self.example_ratio_max = max(self.example_ratio_max, float(torch.max(ratios)))

    def finalize(self):
        if self.example_count <= 0 or self.dense_squared <= 0:
            raise ValueError("No nonzero dense activation examples were accumulated")
        normalized_squared = self.residual_squared / self.dense_squared
        mean = self.example_ratio_sum / self.example_count
        variance = max(
            self.example_ratio_squared_sum / self.example_count - mean * mean, 0.0
        )
        denominator = math.sqrt(self.dense_squared * self.compressed_squared)
        return {
            "activation_normalized_squared_error": normalized_squared,
            "activation_relative_frobenius_error": math.sqrt(
                max(normalized_squared, 0.0)
            ),
            "activation_cosine_similarity": self.dot / denominator
            if denominator > 0
            else float("nan"),
            "dense_activation_norm": math.sqrt(self.dense_squared),
            "compressed_activation_norm": math.sqrt(self.compressed_squared),
            "activation_absolute_mse": self.residual_squared / self.element_count,
            "activation_example_normalized_squared_error_mean": mean,
            "activation_example_normalized_squared_error_std_population": math.sqrt(
                variance
            ),
            "activation_example_normalized_squared_error_min": self.example_ratio_min,
            "activation_example_normalized_squared_error_max": self.example_ratio_max,
            "activation_example_count": self.example_count,
            "activation_element_count": self.element_count,
            "activation_input_protocol": (
                "dense classifier.0 input reused for dense and compressed target layers"
            ),
            "activation_scope": "isolated_target_layer_output",
        }


@torch.no_grad()
def measure_layer_activation_distortion(
    dense_model,
    target_layer,
    compressed_layer,
    loader,
    device,
    *,
    max_batches=None,
):
    dense_model = dense_model.to(device).eval()
    compressed_layer = compressed_layer.to(device).eval()
    target = get_module(dense_model, target_layer)
    accumulator = ActivationDistortionAccumulator()

    def hook(module, inputs, output):
        common_input = inputs[0]
        compressed_output = compressed_layer(common_input)
        accumulator.update(output, compressed_output)

    handle = target.register_forward_hook(hook)
    try:
        for batch_index, (images, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            dense_model(images.to(device))
    finally:
        handle.remove()
    return accumulator.finalize()


def required_metric_fields(method):
    common = {
        "actual_relative_squared_frobenius_error",
        "actual_relative_frobenius_error",
        "gap_above_max_bound",
        "activation_normalized_squared_error",
        "activation_relative_frobenius_error",
        "activation_cosine_similarity",
        "dense_activation_norm",
        "compressed_activation_norm",
        "activation_absolute_mse",
        "activation_example_normalized_squared_error_mean",
        "activation_example_normalized_squared_error_std_population",
        "activation_example_normalized_squared_error_min",
        "activation_example_normalized_squared_error_max",
        "validation_present_class_miou",
        "test_present_class_miou",
        "target_layer_parameter_count",
        "target_layer_parameter_ratio",
        "full_model_parameter_count",
        "target_layer_macs",
        "full_model_macs",
        "compression_factor",
    }
    if method == FINAL_CP_METHOD:
        common |= {
            "initialization_hash_sha256",
            "final_factor_hash_sha256",
            "factor_artifact_sha256",
            "decomposition_runtime_seconds",
        }
    return common


def verify_completed_row_schema(row):
    key = final_structural_key(row)
    missing = [
        field
        for field in required_metric_fields(key[0])
        if row.get(field, "") in ("", None)
    ]
    if missing:
        raise ValueError(f"Completed row {key} lacks fields: {missing}")
    numeric_fields = [
        field
        for field in required_metric_fields(key[0])
        if field
        not in {
            "initialization_hash_sha256",
            "final_factor_hash_sha256",
            "factor_artifact_sha256",
        }
    ]
    nonfinite = [
        field for field in numeric_fields if not math.isfinite(float(row[field]))
    ]
    if nonfinite:
        raise ValueError(f"Completed row {key} contains nonfinite fields: {nonfinite}")
    return True


def aggregate_final_structural_rows(rows):
    normalized, _, diagnostics = normalize_final_structural_rows(
        rows, context="final structural rank aggregation", warn=False
    )
    completed = [row for row in normalized if row.get("status") == "completed"]
    output = []
    grouped = defaultdict(list)
    for row in completed:
        grouped[(row["method"], int(row["rank"]))].append(row)
    metric_fields = [
        "actual_relative_squared_frobenius_error",
        "activation_normalized_squared_error",
        "validation_present_class_miou",
        "test_present_class_miou",
        "target_layer_parameter_ratio",
        "compression_factor",
    ]
    for (method, rank), group in sorted(grouped.items()):
        aggregate = {
            "method": method,
            "rank": rank,
            "completed_scientific_row_count": len(group),
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_std_population"] = float(values.std(ddof=0))
            aggregate[f"{field}_min"] = float(values.min())
            aggregate[f"{field}_max"] = float(values.max())
        output.append(aggregate)
    return output, diagnostics


def _write_metric_figure(rows, figures_dir, stem, y_field, ylabel, title):
    normalized, _, diagnostics = normalize_final_structural_rows(
        rows, context=f"{stem} figure", warn=False
    )
    completed = [row for row in normalized if row.get("status") == "completed"]
    if not completed:
        raise ValueError(f"No completed rows for {stem}")
    frame = pd.DataFrame(completed)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame[y_field] = pd.to_numeric(frame[y_field], errors="coerce")
    tradeoff = stem == "final_structural_zero_shot_tradeoff"
    x_field = "target_layer_parameter_ratio" if tradeoff else "rank"
    frame[x_field] = pd.to_numeric(frame[x_field], errors="coerce")
    frame = frame.dropna(subset=["rank", x_field, y_field])
    if frame.empty:
        raise ValueError(f"No numeric rows for {stem}")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_rows = []
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    cp = frame[frame["method"] == FINAL_CP_METHOD]
    if not cp.empty:
        for seed, group in cp.groupby("seed"):
            values = group.sort_values("rank")
            ax.scatter(
                values[x_field],
                values[y_field],
                s=30,
                alpha=0.75,
                label=f"CP seed {int(seed)}",
            )
            for _, row in values.iterrows():
                figure_rows.append(
                    {
                        "series": "cp_individual",
                        "method": FINAL_CP_METHOD,
                        "rank": int(row["rank"]),
                        "seed": int(row["seed"]),
                        x_field: float(row[x_field]),
                        y_field: float(row[y_field]),
                    }
                )
        if tradeoff:
            means = cp.groupby("rank", as_index=False)[[x_field, y_field]].mean()
        else:
            means = cp.groupby("rank", as_index=False)[y_field].mean()
        ax.plot(
            means[x_field],
            means[y_field],
            color="black",
            linewidth=2.0,
            marker="o",
            label="CP mean",
        )
        for _, row in means.iterrows():
            figure_rows.append(
                {
                    "series": "cp_mean",
                    "method": FINAL_CP_METHOD,
                    "rank": int(row["rank"]),
                    "seed": "",
                    x_field: float(row[x_field]),
                    y_field: float(row[y_field]),
                }
            )
    svd = frame[frame["method"] == FINAL_SVD_METHOD].sort_values("rank")
    if not svd.empty:
        ax.plot(
            svd[x_field],
            svd[y_field],
            color="#D55E00",
            linewidth=1.7,
            marker="s",
            label="Matrix SVD",
        )
        for _, row in svd.iterrows():
            figure_rows.append(
                {
                    "series": "matrix_svd",
                    "method": FINAL_SVD_METHOD,
                    "rank": int(row["rank"]),
                    "seed": "",
                    x_field: float(row[x_field]),
                    y_field: float(row[y_field]),
                }
            )
    if tradeoff:
        dense_field = f"dense_{y_field}"
        if dense_field in frame and frame[dense_field].notna().any():
            dense_value = float(frame[dense_field].dropna().iloc[0])
            ax.scatter([1.0], [dense_value], marker="*", s=80, color="#666666", label="Dense")
            figure_rows.append(
                {
                    "series": "dense_reference",
                    "method": "dense",
                    "rank": "",
                    "seed": "",
                    x_field: 1.0,
                    y_field: dense_value,
                }
            )
        ax.set_xscale("log")
        ax.set_xlabel("Target-layer parameter ratio")
    else:
        ax.set_xscale("log", base=2)
        ranks = sorted(int(value) for value in frame["rank"].unique())
        ax.set_xticks(ranks, [str(value) for value in ranks])
        ax.set_xlabel("Nominal rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    if tradeoff:
        ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    data_path = figures_dir / f"{stem}.csv"
    write_csv(data_path, figure_rows)
    paths = [str(data_path)]
    for suffix in ("pdf", "png"):
        path = figures_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths, diagnostics


def write_final_structural_figures(rows, figures_dir):
    outputs = []
    diagnostics = []
    specifications = (
        (
            "final_structural_weight_error",
            "actual_relative_squared_frobenius_error",
            "Normalized squared Frobenius residual",
            "Weight reconstruction at the final structural protocol",
        ),
        (
            "final_structural_activation_distortion",
            "activation_normalized_squared_error",
            "Normalized squared activation error",
            "Target-layer activation distortion",
        ),
        (
            "final_structural_zero_shot_tradeoff",
            "validation_present_class_miou",
            "Validation present-class mIoU",
            "Zero-shot segmentation at the final structural protocol",
        ),
    )
    for stem, field, ylabel, title in specifications:
        paths, current = _write_metric_figure(
            rows, figures_dir, stem, field, ylabel, title
        )
        outputs.extend(paths)
        diagnostics.extend(current)
    return outputs, diagnostics


def _spearman(frame, left, right):
    subset = frame[[left, right]].dropna()
    if len(subset) < 3 or subset[left].nunique() < 2 or subset[right].nunique() < 2:
        return None
    return float(subset[left].corr(subset[right], method="spearman"))


def write_final_structural_audit(
    rows,
    rank_summary,
    path,
    *,
    expected_ranks,
    expected_seeds,
):
    normalized, _, _ = normalize_final_structural_rows(rows, warn=False)
    expected_keys = {
        (FINAL_CP_METHOD, int(rank), int(seed), FINAL_ITERATION_BUDGET)
        for rank in expected_ranks
        for seed in expected_seeds
    } | {
        (FINAL_SVD_METHOD, int(rank), "", "") for rank in expected_ranks
    }
    scientific = {final_structural_key(row): row for row in normalized}
    complete = all(
        key in scientific and scientific[key].get("status") == "completed"
        for key in expected_keys
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final structural evaluation audit",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        "The CP rows use one fitted 200-iteration layer artifact for weight reconstruction, "
        "isolated target-layer activation distortion, and validation/test zero-shot evaluation. "
        "Matrix SVD denotes the deterministic output-unfolding matrix control.",
        "",
    ]
    if not complete:
        missing = [
            key
            for key in sorted(expected_keys, key=str)
            if key not in scientific or scientific[key].get("status") != "completed"
        ]
        lines.extend(
            [
                "The final scientific decisions are deferred until all requested rows complete.",
                "",
                f"Missing or failed keys: `{missing}`.",
            ]
        )
        path.write_text("\n".join(lines) + "\n")
        return str(path), False

    frame = pd.DataFrame(
        [row for row in normalized if row.get("status") == "completed"]
    )
    cp = frame[frame["method"] == FINAL_CP_METHOD].copy()
    summaries = pd.DataFrame(rank_summary)
    cp_summary = summaries[summaries["method"] == FINAL_CP_METHOD].sort_values("rank")
    svd_summary = summaries[summaries["method"] == FINAL_SVD_METHOD].sort_values("rank")
    weight_order = cp_summary[
        ["rank", "actual_relative_squared_frobenius_error_mean"]
    ].to_dict("records")
    activation_order = cp_summary[
        ["rank", "activation_normalized_squared_error_mean"]
    ].to_dict("records")
    zero_shot_order = cp_summary[
        ["rank", "validation_present_class_miou_mean", "test_present_class_miou_mean"]
    ].to_dict("records")
    weight_corr = _spearman(
        cp,
        "actual_relative_squared_frobenius_error",
        "validation_present_class_miou",
    )
    activation_corr = _spearman(
        cp, "activation_normalized_squared_error", "validation_present_class_miou"
    )
    if activation_corr is None and weight_corr is None:
        correlation_decision = (
            "Both exploratory associations are undefined because an input is constant "
            "or the completed subset is undersized."
        )
    elif activation_corr is not None and (
        weight_corr is None or abs(activation_corr) > abs(weight_corr)
    ):
        correlation_decision = (
            "Activation distortion has the stronger absolute exploratory Spearman association."
        )
    else:
        correlation_decision = (
            "Weight error has the stronger or equal absolute exploratory Spearman association."
        )
    cp_rank_means = {
        int(row["rank"]): row for row in cp_summary.to_dict("records")
    }
    frontier = []
    for rank in sorted(cp_rank_means):
        candidate = cp_rank_means[rank]
        ratio = float(candidate["target_layer_parameter_ratio_mean"])
        score = float(candidate["validation_present_class_miou_mean"])
        dominated = any(
            float(other["target_layer_parameter_ratio_mean"]) <= ratio
            and float(other["validation_present_class_miou_mean"]) >= score
            and (
                float(other["target_layer_parameter_ratio_mean"]) < ratio
                or float(other["validation_present_class_miou_mean"]) > score
            )
            for other_rank, other in cp_rank_means.items()
            if other_rank != rank
        )
        if not dominated:
            frontier.append(rank)
    fine_tune = [
        int(row["rank"])
        for row in sorted(
            cp_summary.to_dict("records"),
            key=lambda row: (
                float(row["validation_present_class_miou_mean"]),
                -float(row["target_layer_parameter_ratio_mean"]),
            ),
            reverse=True,
        )[:3]
    ]
    evaluated_high_rank = 512 if 512 in set(cp_summary["rank"]) else int(
        cp_summary["rank"].max()
    )
    high_rank_row = cp_summary[cp_summary["rank"] == evaluated_high_rank].iloc[0]
    rank512_usable = (
        math.isfinite(
            float(high_rank_row["activation_normalized_squared_error_mean"])
        )
        and math.isfinite(
            float(high_rank_row["validation_present_class_miou_mean"])
        )
    )
    lines.extend(
        [
            "## Scientific decisions",
            "",
            f"1. **Weight reconstruction versus rank.** `{weight_order}`.",
            f"2. **Activation distortion versus rank.** `{activation_order}`.",
            f"3. **Zero-shot segmentation versus rank.** `{zero_shot_order}`.",
            "4. **Does activation distortion track zero-shot performance more closely than "
            f"weight error?** {correlation_decision} Weight-error Spearman=`{weight_corr}`; "
            f"activation-distortion Spearman=`{activation_corr}`. These are exploratory "
            "descriptions without significance tests or causal interpretation.",
            "5. **CP versus matrix SVD at matched nominal rank.** See the paired rank rows in "
            f"`{svd_summary.to_dict('records')}`. Equal nominal rank does not imply equal "
            "parameter count.",
            f"6. **Compression-accuracy Pareto frontier among CP ranks.** `{frontier}`.",
            f"7. **Three ranks selected for controlled fine-tuning.** `{fine_tune}` based on "
            "validation present-class mIoU with parameter ratio as a secondary ordering.",
            f"8. **Is rank 512 numerically and functionally usable at 200 iterations?** "
            f"`{rank512_usable}` for evaluated high rank `{evaluated_high_rank}` in this "
            "completed structural protocol; this does not establish "
            "fine-tuned performance.",
            "9. **Are further structural experiments necessary?** No additional structural "
            "experiment is required if all identity, finiteness, and reuse checks in metadata pass. "
            "Proceed only to the separately controlled fine-tuning stage selected above.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return str(path), True
