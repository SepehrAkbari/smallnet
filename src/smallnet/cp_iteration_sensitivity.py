'''Targeted CP iteration-budget sensitivity diagnostics.'''

import gc
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.factorization import factorized_conv_from_conv
from src.smallnet.reproducibility import set_seed
from src.smallnet.results import write_csv
from src.smallnet.structural import (
    _normalize_nonnegative_integer,
    configure_memory_efficient_mttkrp,
    cp_conv_parameter_count,
    dense_conv_parameter_count,
    diagnostic_bounds,
    explicit_khatri_rao_peak_bytes,
    fit_cp_approximation,
    normalized_frobenius_residual,
    output_mode_svd,
)


SENSITIVITY_METHOD = "cp"
DEFAULT_ITERATION_BUDGETS = (10, 25, 50, 100)


def scientific_cp_iteration_key(row):
    method = str(row.get("method", "")).strip()
    if method != SENSITIVITY_METHOD:
        raise ValueError(f"unsupported sensitivity method label {method!r}")
    rank = _normalize_nonnegative_integer(row.get("rank"), "rank", positive=True)
    seed = _normalize_nonnegative_integer(row.get("seed"), "seed")
    budget = _normalize_nonnegative_integer(
        row.get("iteration_budget"), "iteration_budget", positive=True
    )
    return method, rank, seed, budget


def normalize_cp_iteration_rows(
    rows, context="CP iteration-sensitivity rows", *, warn=True
):
    '''Normalize key fields, retain one scientific row per rank/seed/budget.'''
    normalized_by_key = {}
    rejected = []
    diagnostics = []
    for index, original in enumerate(rows):
        row = dict(original)
        try:
            key = scientific_cp_iteration_key(row)
            row["method"], row["rank"], row["seed"], row["iteration_budget"] = key
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
        previous = normalized_by_key.get(key)
        if previous is not None:
            diagnostics.append(
                {
                    "context": context,
                    "row_index": index,
                    "reason": f"duplicate scientific row for key {key!r}; retained one row",
                    "method": key[0],
                    "rank": key[1],
                    "seed": key[2],
                    "iteration_budget": key[3],
                }
            )
            if previous.get("status") == "completed" and row.get("status") != "completed":
                continue
        normalized_by_key[key] = row
    if diagnostics and warn:
        warnings.warn(
            f"{context}: excluded or deduplicated {len(diagnostics)} row(s); "
            "details are recorded in metadata",
            RuntimeWarning,
            stacklevel=2,
        )
    normalized = sorted(
        normalized_by_key.values(),
        key=lambda row: (int(row["rank"]), int(row["seed"]), int(row["iteration_budget"])),
    )
    return normalized, rejected, diagnostics


def _hash_tensor(hasher, name, tensor):
    value = tensor.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes())


def _cp_components_hash(weights, factors):
    hasher = hashlib.sha256()
    if weights is not None:
        _hash_tensor(hasher, "weights", weights)
    for index, factor in enumerate(factors):
        _hash_tensor(hasher, f"factor_{index}", factor)
    return hasher.hexdigest()


def cp_initialization_hash(
    conv,
    rank,
    seed,
    init,
    device,
    *,
    memory_efficient_mttkrp=True,
    mttkrp_rank_chunk_size=64,
    mttkrp_max_explicit_bytes=512 * 1024**2,
):
    '''Hash the exact zero-iteration CP factors produced by the fitted-layer path.'''
    set_seed(seed, deterministic=True)
    configure_memory_efficient_mttkrp(
        memory_efficient_mttkrp,
        rank_chunk_size=mttkrp_rank_chunk_size,
        max_explicit_bytes=mttkrp_max_explicit_bytes,
    )
    probe = factorized_conv_from_conv(
        conv.to(device),
        rank=rank,
        factorization="cp",
        init=init,
        n_iter_max=0,
    ).cpu()
    digest = _cp_components_hash(probe.weight.weights, probe.weight.factors)
    del probe
    gc.collect()
    return digest


def fit_cp_approximation_with_initialization_capture(*args, **kwargs):
    '''Fit CP while hashing the factors returned by the actual ALS initializer.'''
    from tltorch.factorized_tensors.factorized_tensors import CPTensor

    parafac = CPTensor.init_from_tensor.__globals__["parafac"]
    parafac_globals = parafac.__globals__
    initialize_cp = parafac_globals["initialize_cp"]
    captured = []

    def recording_initialize_cp(*initialize_args, **initialize_kwargs):
        initialized = initialize_cp(*initialize_args, **initialize_kwargs)
        weights, factors = initialized
        captured.append(_cp_components_hash(weights, factors))
        return initialized

    parafac_globals["initialize_cp"] = recording_initialize_cp
    try:
        result = fit_cp_approximation(*args, **kwargs)
    finally:
        parafac_globals["initialize_cp"] = initialize_cp
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected exactly one captured CP initialization, observed {len(captured)}"
        )
    return (*result, captured[0])


def apply_residual_reduction_comparisons(rows):
    '''Attach the four requested within-rank/seed residual comparisons.'''
    normalized, rejected, diagnostics = normalize_cp_iteration_rows(
        rows, context="iteration-sensitivity residual comparisons", warn=False
    )
    completed = defaultdict(dict)
    for row in normalized:
        if row.get("status") == "completed":
            completed[(int(row["rank"]), int(row["seed"]))][int(row["iteration_budget"])] = float(
                row["actual_relative_squared_frobenius_error"]
            )
    names = {
        (10, 25): "absolute_squared_residual_reduction_10_to_25",
        (25, 50): "absolute_squared_residual_reduction_25_to_50",
        (50, 100): "absolute_squared_residual_reduction_50_to_100",
    }
    for row in normalized:
        values = completed.get((int(row["rank"]), int(row["seed"])), {})
        for (start, stop), name in names.items():
            row[name] = values[start] - values[stop] if start in values and stop in values else ""
        if 10 in values and 100 in values:
            row["relative_squared_residual_reduction_10_to_100"] = (
                (values[10] - values[100]) / values[10] if values[10] else ""
            )
        else:
            row["relative_squared_residual_reduction_10_to_100"] = ""
    return [*normalized, *rejected], diagnostics


def aggregate_cp_iteration_rows(rows, expected_seed_count=None):
    normalized, _, diagnostics = normalize_cp_iteration_rows(
        rows, context="CP iteration-sensitivity aggregation", warn=False
    )
    grouped = defaultdict(list)
    reference = {}
    for row in normalized:
        if row.get("status") != "completed":
            continue
        key = (int(row["rank"]), int(row["iteration_budget"]))
        grouped[key].append(float(row["actual_relative_squared_frobenius_error"]))
        reference[key] = row
    output = []
    by_rank = defaultdict(dict)
    for (rank, budget), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        ref = reference[(rank, budget)]
        record = {
            "method": SENSITIVITY_METHOD,
            "rank": rank,
            "iteration_budget": budget,
            "completed_seed_count": len(array),
            "expected_seed_count": expected_seed_count if expected_seed_count is not None else "",
            "actual_relative_squared_frobenius_error_mean": float(array.mean()),
            "actual_relative_squared_frobenius_error_std_population": float(array.std(ddof=0)),
            "actual_relative_squared_frobenius_error_min": float(array.min()),
            "actual_relative_squared_frobenius_error_max": float(array.max()),
            "actual_relative_squared_frobenius_error_seed_range": float(array.max() - array.min()),
            "max_unfolding_tail_bound_squared": float(ref["max_unfolding_tail_bound_squared"]),
            "output_mode_svd_residual_squared": float(ref["output_mode_svd_residual_squared"]),
            "gap_above_max_bound_mean": float(
                array.mean() - float(ref["max_unfolding_tail_bound_squared"])
            ),
        }
        output.append(record)
        by_rank[rank][budget] = record
    for rank, budgets in by_rank.items():
        if 50 in budgets and 100 in budgets:
            change = budgets[50]["actual_relative_squared_frobenius_error_mean"] - budgets[100][
                "actual_relative_squared_frobenius_error_mean"
            ]
            for record in budgets.values():
                record["mean_absolute_squared_residual_reduction_50_to_100"] = change
                record["mean_change_50_to_100_less_than_1e_minus_3"] = abs(change) < 1e-3
        if 10 in budgets and 100 in budgets:
            initial = budgets[10]["actual_relative_squared_frobenius_error_mean"]
            relative = (initial - budgets[100]["actual_relative_squared_frobenius_error_mean"]) / initial
            range_10 = budgets[10]["actual_relative_squared_frobenius_error_seed_range"]
            range_100 = budgets[100]["actual_relative_squared_frobenius_error_seed_range"]
            trend = "unchanged"
            if range_100 > range_10:
                trend = "increased"
            elif range_100 < range_10:
                trend = "decreased"
            for record in budgets.values():
                record["mean_relative_squared_residual_reduction_10_to_100"] = relative
                record["mean_relative_reduction_10_to_100_less_than_1_percent"] = abs(relative) < 0.01
                record["seed_range_change_10_to_100"] = range_100 - range_10
                record["seed_variability_trend_10_to_100"] = trend
    return output, diagnostics


def sensitivity_rank_ordering_changes(aggregates):
    frame = pd.DataFrame(aggregates)
    if frame.empty:
        return {"defined": False, "reason": "no completed aggregate rows"}
    orders = {}
    for budget, group in frame.groupby("iteration_budget"):
        orders[int(budget)] = [
            int(value)
            for value in group.sort_values("actual_relative_squared_frobenius_error_mean")["rank"]
        ]
    complete_orders = [tuple(order) for order in orders.values() if len(order) == frame["rank"].nunique()]
    return {
        "defined": bool(complete_orders),
        "rank_order_by_iteration_budget_best_to_worst": orders,
        "rank_order_changed": len(set(complete_orders)) > 1 if complete_orders else None,
    }


def reference_diagnostic_for_conv(weight, ranks, device):
    '''Compute spectral references when canonical reconstruction metadata is unavailable.'''
    u, singular_values, vh, runtime, svd_device = output_mode_svd(weight, device=device)
    diagnostic = rank_energy_diagnostic(
        weight.detach().cpu(),
        modes=(0, 1, 2, 3),
        fixed_ranks=ranks,
        precomputed_singular_values={0: singular_values.numpy()},
    )
    diagnostic["output_mode_matrix_svd_device"] = svd_device
    diagnostic["output_mode_matrix_svd_runtime_seconds"] = runtime
    del u, singular_values, vh
    gc.collect()
    return diagnostic


def cp_iteration_rows_for_conv(
    conv,
    ranks,
    seeds,
    iteration_budgets,
    init,
    device,
    diagnostic,
    checkpoint_hash,
    *,
    tolerance=1e-5,
    memory_efficient_mttkrp=True,
    mttkrp_rank_chunk_size=64,
    mttkrp_max_explicit_bytes=512 * 1024**2,
    skip_keys=None,
    expected_initialization_hashes=None,
    on_update=None,
):
    '''Fit independent CP approximations from verified identical initial factors.'''
    weight = conv.weight.detach().cpu()
    shape = tuple(weight.shape)
    dense_parameters = dense_conv_parameter_count(shape, conv.bias is not None)
    skip_keys = set(skip_keys or ())
    expected_initialization_hashes = dict(expected_initialization_hashes or {})
    rows = []
    failures = []
    initialization_diagnostics = []
    for rank in ranks:
        rank = int(rank)
        max_bound, output_bound, _ = diagnostic_bounds(diagnostic, rank)
        parameters = cp_conv_parameter_count(shape, rank, conv.bias is not None)
        for seed in seeds:
            seed = int(seed)
            for budget in iteration_budgets:
                budget = int(budget)
                key = (SENSITIVITY_METHOD, rank, seed, budget)
                if key in skip_keys:
                    continue
                initialization_hash = ""
                try:
                    initialization_hash = cp_initialization_hash(
                        conv,
                        rank,
                        seed,
                        init,
                        device,
                        memory_efficient_mttkrp=memory_efficient_mttkrp,
                        mttkrp_rank_chunk_size=mttkrp_rank_chunk_size,
                        mttkrp_max_explicit_bytes=mttkrp_max_explicit_bytes,
                    )
                    hash_key = (rank, seed)
                    expected = expected_initialization_hashes.setdefault(hash_key, initialization_hash)
                    if initialization_hash != expected:
                        raise AssertionError(
                            f"initial CP factors differ across budgets for rank={rank}, seed={seed}: "
                            f"{initialization_hash} != {expected}"
                        )
                    seed_status = set_seed(seed, deterministic=True)
                    (
                        fitted,
                        approximation,
                        runtime,
                        mttkrp_implementation,
                        actual_fit_initialization_hash,
                    ) = fit_cp_approximation_with_initialization_capture(
                        conv,
                        rank,
                        seed,
                        init,
                        budget,
                        device,
                        memory_efficient_mttkrp,
                        mttkrp_rank_chunk_size,
                        mttkrp_max_explicit_bytes,
                    )
                    if actual_fit_initialization_hash != initialization_hash:
                        raise AssertionError(
                            "The factors captured from the actual fitted call do not match the "
                            f"zero-iteration probe: {actual_fit_initialization_hash} != "
                            f"{initialization_hash}"
                        )
                    squared, ordinary = normalized_frobenius_residual(weight, approximation)
                    actual_parameter_count = sum(parameter.numel() for parameter in fitted.parameters())
                    if actual_parameter_count != parameters:
                        raise AssertionError(
                            f"CP parameter count {actual_parameter_count} != formula {parameters}"
                        )
                    gap = squared - max_bound
                    if gap < -tolerance:
                        raise AssertionError(
                            f"CP squared residual {squared} violates unfolding lower bound {max_bound}"
                        )
                    row = {
                        "method": SENSITIVITY_METHOD,
                        "rank": rank,
                        "seed": seed,
                        "iteration_budget": budget,
                        "initializer": init,
                        "initialization_identifier": f"sha256:{initialization_hash}",
                        "initialization_hash_sha256": initialization_hash,
                        "actual_fit_initialization_hash_sha256": actual_fit_initialization_hash,
                        "initialization_verification": (
                            "independent_zero_iteration_probe_matches_factors_captured_from_actual_fit"
                        ),
                        "initialization_matches_other_budgets": True,
                        "actual_relative_squared_frobenius_error": squared,
                        "actual_relative_frobenius_error": ordinary,
                        "max_unfolding_tail_bound_squared": max_bound,
                        "gap_above_max_bound": gap,
                        "output_mode_svd_residual_squared": output_bound,
                        "decomposition_runtime_seconds": runtime,
                        "status": "completed",
                        "completed_requested_budget": True,
                        "convergence_status": "completed_requested_budget_iteration_history_unavailable",
                        "failure_exception": "",
                        "device": torch.device(device).type,
                        "deterministic_settings": json.dumps(seed_status, sort_keys=True),
                        "checkpoint_sha256": checkpoint_hash,
                        "dense_tensor_shape": "x".join(map(str, shape)),
                        "numerical_precision": str(weight.dtype),
                        "parameter_count": parameters,
                        "parameter_ratio": parameters / dense_parameters,
                        "compression_factor": dense_parameters / parameters,
                        "bound_tolerance": tolerance,
                        "mttkrp_implementation": mttkrp_implementation,
                        "mttkrp_rank_chunk_size": int(mttkrp_rank_chunk_size),
                        "mttkrp_max_explicit_bytes": int(mttkrp_max_explicit_bytes),
                        "default_explicit_khatri_rao_peak_bytes_avoided": explicit_khatri_rao_peak_bytes(
                            shape, rank
                        ),
                    }
                    rows.append(row)
                    del fitted, approximation
                except Exception as exc:
                    failure = {
                        "method": SENSITIVITY_METHOD,
                        "rank": rank,
                        "seed": seed,
                        "iteration_budget": budget,
                        "exception": repr(exc),
                    }
                    failures.append(failure)
                    rows.append(
                        {
                            "method": SENSITIVITY_METHOD,
                            "rank": rank,
                            "seed": seed,
                            "iteration_budget": budget,
                            "initializer": init,
                            "initialization_hash_sha256": initialization_hash,
                            "status": "failed",
                            "completed_requested_budget": False,
                            "failure_exception": repr(exc),
                            "device": torch.device(device).type,
                            "checkpoint_sha256": checkpoint_hash,
                            "dense_tensor_shape": "x".join(map(str, shape)),
                        }
                    )
                gc.collect()
                if on_update:
                    on_update(rows)
    for (rank, seed), expected in sorted(expected_initialization_hashes.items()):
        observed = {
            row.get("initialization_hash_sha256")
            for row in rows
            if row.get("status") == "completed"
            and int(row["rank"]) == rank
            and int(row["seed"]) == seed
        }
        if observed:
            initialization_diagnostics.append(
                {
                    "rank": rank,
                    "seed": seed,
                    "expected_initialization_hash_sha256": expected,
                    "observed_hashes": sorted(observed),
                    "identical_across_executed_budgets": observed == {expected},
                }
            )
    return rows, failures, initialization_diagnostics


def write_cp_iteration_sensitivity_figure(rows, figures_dir):
    '''Write a partial-safe three-panel sensitivity figure and companion CSV.'''
    normalized, _, diagnostics = normalize_cp_iteration_rows(
        rows, context="CP iteration-sensitivity figure"
    )
    completed = [row for row in normalized if row.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed CP iteration-sensitivity rows are available")
    frame = pd.DataFrame(completed)
    for field in (
        "rank",
        "seed",
        "iteration_budget",
        "actual_relative_squared_frobenius_error",
        "max_unfolding_tail_bound_squared",
        "output_mode_svd_residual_squared",
    ):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    malformed = frame[
        ["rank", "seed", "iteration_budget", "actual_relative_squared_frobenius_error"]
    ].isna().any(axis=1)
    if malformed.any():
        warnings.warn(
            f"CP iteration-sensitivity figure excluded {int(malformed.sum())} malformed row(s)",
            RuntimeWarning,
            stacklevel=2,
        )
        frame = frame.loc[~malformed].copy()
    ranks = sorted(int(value) for value in frame["rank"].unique())
    fig, axes = plt.subplots(1, len(ranks), figsize=(4.1 * len(ranks), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    figure_rows = []
    colors = plt.get_cmap("tab10")
    for panel, (ax, rank) in enumerate(zip(axes, ranks)):
        subset = frame[frame["rank"] == rank].sort_values(["iteration_budget", "seed"])
        for seed_index, (seed, seed_rows) in enumerate(subset.groupby("seed")):
            x = seed_rows["iteration_budget"].to_numpy(dtype=float)
            y = seed_rows["actual_relative_squared_frobenius_error"].to_numpy(dtype=float)
            plot_x = x * (1.0 + 0.018 * (seed_index - (subset["seed"].nunique() - 1) / 2))
            ax.scatter(
                plot_x,
                y,
                facecolors="none",
                edgecolors=colors(panel),
                linewidths=1.1,
                s=28,
                alpha=0.9,
                label="CP individual seeds" if seed_index == 0 else None,
            )
            for budget, plotted_budget, residual in zip(x, plot_x, y):
                figure_rows.append(
                    {
                        "series": "cp_individual_seed",
                        "rank": rank,
                        "seed": int(seed),
                        "iteration_budget": int(budget),
                        "plotted_iteration_budget": float(plotted_budget),
                        "normalized_squared_frobenius_residual": float(residual),
                    }
                )
        means = subset.groupby("iteration_budget", as_index=False)[
            "actual_relative_squared_frobenius_error"
        ].mean()
        ax.plot(
            means["iteration_budget"],
            means["actual_relative_squared_frobenius_error"],
            color=colors(panel),
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="CP mean",
        )
        for _, item in means.iterrows():
            figure_rows.append(
                {
                    "series": "cp_mean",
                    "rank": rank,
                    "seed": "",
                    "iteration_budget": int(item["iteration_budget"]),
                    "plotted_iteration_budget": float(item["iteration_budget"]),
                    "normalized_squared_frobenius_residual": float(
                        item["actual_relative_squared_frobenius_error"]
                    ),
                }
            )
        bound = float(subset["max_unfolding_tail_bound_squared"].iloc[0])
        svd = float(subset["output_mode_svd_residual_squared"].iloc[0])
        ax.axhline(svd, color="#D55E00", linestyle="--", linewidth=1.5, label="Matrix-SVD reference")
        ax.axhline(bound, color="#333333", linestyle=":", linewidth=1.4, label="Strongest unfolding bound")
        for series, value in (("matrix_svd_reference", svd), ("strongest_unfolding_bound", bound)):
            figure_rows.append(
                {
                    "series": series,
                    "rank": rank,
                    "seed": "",
                    "iteration_budget": "",
                    "plotted_iteration_budget": "",
                    "normalized_squared_frobenius_residual": value,
                }
            )
        ax.set_xscale("log")
        budgets = sorted(int(value) for value in subset["iteration_budget"].unique())
        ax.set_xticks(budgets, [str(value) for value in budgets])
        ax.set_xlabel("CP iteration budget")
        ax.set_title(f"CP rank {rank}")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Normalized squared Frobenius residual")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("CP reconstruction sensitivity to the requested iteration budget", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = figures_dir / "cp_iteration_sensitivity.csv"
    write_csv(data_path, figure_rows)
    paths = [str(data_path)]
    for suffix in ("pdf", "png"):
        path = figures_dir / f"cp_iteration_sensitivity.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths, diagnostics


def write_cp_iteration_sensitivity_audit(
    aggregates,
    path,
    *,
    canonical_ranks,
    canonical_seeds,
    canonical_iteration_budgets,
    failures,
    rank_ordering,
    canonical_ten_iteration_reproduction=None,
):
    '''Write a restrained decision record; incomplete runs are labeled explicitly.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(aggregates)
    expected = len(canonical_ranks) * len(canonical_iteration_budgets)
    reproduction = list(canonical_ten_iteration_reproduction or [])
    reproduction_passed = not reproduction or all(
        item.get("within_tolerance") for item in reproduction
    )
    complete = (
        len(frame) == expected
        and not frame.empty
        and bool((frame["completed_seed_count"] == len(canonical_seeds)).all())
        and not failures
        and reproduction_passed
    )
    decision_ready = complete and set(DEFAULT_ITERATION_BUDGETS).issubset(
        set(canonical_iteration_budgets)
    )
    lines = [
        "# CP iteration-budget sensitivity audit",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        "This diagnostic reports completed requested budgets and residual stabilization. "
        "It does not claim certified convergence because no per-iteration convergence history is available.",
        "",
        "## Rank-level diagnostics",
        "",
        "| Rank | Budget | Mean squared residual | Population SD | Seed range | Mean gap above bound |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    if not frame.empty:
        for row in frame.sort_values(["rank", "iteration_budget"]).to_dict("records"):
            lines.append(
                f"| {int(row['rank'])} | {int(row['iteration_budget'])} | "
                f"{row['actual_relative_squared_frobenius_error_mean']:.9f} | "
                f"{row['actual_relative_squared_frobenius_error_std_population']:.9f} | "
                f"{row['actual_relative_squared_frobenius_error_seed_range']:.9f} | "
                f"{row['gap_above_max_bound_mean']:.9f} |"
            )
    lines.extend(["", "## Required descriptive thresholds", ""])
    if decision_ready:
        for rank, group in frame.groupby("rank"):
            row = group.iloc[0]
            lines.append(
                f"- Rank {int(rank)}: mean 50-to-100 reduction "
                f"`{row['mean_absolute_squared_residual_reduction_50_to_100']:.9f}`; "
                f"less than `1e-3`: `{bool(row['mean_change_50_to_100_less_than_1e_minus_3'])}`. "
                f"Mean relative 10-to-100 reduction "
                f"`{100 * row['mean_relative_squared_residual_reduction_10_to_100']:.4f}%`; "
                f"less than 1%: `{bool(row['mean_relative_reduction_10_to_100_less_than_1_percent'])}`. "
                f"Seed variability `{row['seed_variability_trend_10_to_100']}`."
            )
        lines.append(
            f"- Rank ordering changed across complete budgets: `{rank_ordering.get('rank_order_changed')}`."
        )
    else:
        lines.append(
            "- Threshold decisions are deferred until the complete 10/25/50/100 canonical grid is available."
        )
    if reproduction:
        lines.extend(["", "## Canonical ten-iteration reproduction", ""])
        for item in reproduction:
            if item.get("available"):
                lines.append(
                    f"- rank={item['rank']}, seed={item['seed']}: absolute residual difference "
                    f"`{item['absolute_squared_residual_difference']:.3e}`, within tolerance="
                    f"`{item['within_tolerance']}`."
                )
            else:
                lines.append(
                    f"- rank={item['rank']}, seed={item['seed']}: unavailable; "
                    f"{item.get('reason', 'no matching canonical row')}."
                )
    lines.extend(
        [
            "",
            "## Scientific decisions",
            "",
        ]
    )
    if decision_ready:
        plateau = bool(frame.groupby("rank").first()["mean_change_50_to_100_less_than_1e_minus_3"].all())
        under_one_percent = bool(
            frame.groupby("rank").first()[
                "mean_relative_reduction_10_to_100_less_than_1_percent"
            ].all()
        )
        gap_100 = frame[frame["iteration_budget"] == 100].set_index("rank")[
            "gap_above_max_bound_mean"
        ]
        per_rank = frame.groupby("rank").first()
        stable_by_rank = (
            per_rank["mean_change_50_to_100_less_than_1e_minus_3"]
            & per_rank["mean_relative_reduction_10_to_100_less_than_1_percent"]
        )
        ranks_requiring_more_than_ten = [
            int(rank) for rank, stable in stable_by_rank.items() if not bool(stable)
        ]
        ordered_gap_100 = gap_100.sort_index()
        widening_at_100 = bool(np.all(np.diff(ordered_gap_100.to_numpy()) > 0))
        gap_closure = {}
        for rank, group in frame.groupby("rank"):
            by_budget = group.set_index("iteration_budget")
            gap_10 = float(by_budget.loc[10, "gap_above_max_bound_mean"])
            reduction = float(
                by_budget.loc[10, "actual_relative_squared_frobenius_error_mean"]
                - by_budget.loc[100, "actual_relative_squared_frobenius_error_mean"]
            )
            gap_closure[int(rank)] = reduction / gap_10 if gap_10 else float("nan")
        if not plateau:
            later_budget = (
                "No final common budget is justified yet; extend only ranks whose 50-to-100 change "
                "is at least 1e-3, then select a common stabilized budget."
            )
        elif ranks_requiring_more_than_ten:
            later_budget = (
                "Use the larger common 100-iteration budget. Do not use rank-specific budgets, "
                "because they would confound rank comparisons."
            )
        else:
            later_budget = (
                "Retain the common ten-iteration budget; all ranks satisfy both requested "
                "descriptive stabilization thresholds."
            )
        gap_reduction_answer = (
            "No under the requested less-than-1% relative residual-change diagnostic."
            if under_one_percent
            else "At least one rank changes by 1% or more, so budget sensitivity is materially present under the requested diagnostic."
        )
        lines.extend(
            [
                f"1. **Is ten iterations near a residual plateau?** `{plateau and under_one_percent}` "
                "under the two requested descriptive checks: all-rank 50-to-100 change below 1e-3="
                f"`{plateau}` and all-rank absolute relative 10-to-100 change below 1%=`{under_one_percent}`. "
                "This is residual stabilization, not certified convergence.",
                f"2. **Does a larger budget substantially reduce the CP--SVD gap?** {gap_reduction_answer} "
                "The fractions of the "
                "ten-iteration gap removed by 100 iterations are "
                + ", ".join(
                    f"rank {rank}: `{100 * value:.4f}%`" for rank, value in gap_closure.items()
                )
                + ". These descriptive fractions, rather than a preset conclusion, determine whether the "
                "gap reduction is material.",
                "3. **Is the widening high-rank gap present at 100 iterations?** The 100-iteration mean gaps are "
                + ", ".join(f"rank {int(rank)}: `{value:.6f}`" for rank, value in gap_100.items())
                + f". The gap increases monotonically with rank=`{widening_at_100}`; rank ordering changed="
                f"`{rank_ordering.get('rank_order_changed')}`.",
                "4. **Does any rank require a larger canonical budget?** Ranks failing at least one requested "
                f"stabilization check: `{ranks_requiring_more_than_ten}`. This is a protocol decision, not a "
                "convergence certificate.",
                f"5. **Budget for later experiments:** {later_budget}",
                "6. **Would a budget change require rerunning canonical zero-shot results?** Yes. Any CP factors "
                "used for zero-shot evaluation must be refitted at the selected budget and the corresponding zero-shot "
                "rows regenerated. The existing ten-iteration artifacts must remain as a separately labeled protocol.",
                "7. **Is another budget experiment necessary?** No further budget grid is required if the 50-to-100 "
                "changes are descriptively negligible for all ranks; otherwise extend only the affected ranks with "
                "the same initialization protocol.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Ten-iteration plateau decision: deferred.",
                "2. CP--SVD gap-reduction decision: deferred.",
                "3. High-rank 100-iteration gap decision: deferred.",
                "4. Canonical-budget decision: deferred.",
                "5. Later-experiment budget decision: deferred.",
                "6. If the selected fitting budget changes, canonical zero-shot CP results must be rerun and separately labeled.",
                "7. Further iteration-budget work cannot be assessed until the requested grid completes.",
            ]
        )
    if failures:
        lines.extend(["", "## Recorded failures", ""])
        for failure in failures:
            lines.append(
                f"- rank={failure.get('rank')}, seed={failure.get('seed')}, "
                f"budget={failure.get('iteration_budget')}: `{failure.get('exception')}`"
            )
    path.write_text("\n".join(lines) + "\n")
    return str(path), complete
