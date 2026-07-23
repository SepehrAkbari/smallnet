'''Focused numerical-stability diagnostics for high-rank CP fitting.'''

import copy
import gc
import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.smallnet.cp_iteration_sensitivity import (
    _cp_components_hash,
    fit_cp_approximation_with_initialization_capture,
)
from src.smallnet.factorization import factorized_conv_from_conv
from src.smallnet.reproducibility import set_seed
from src.smallnet.results import write_csv
from src.smallnet.structural import (
    _normalize_nonnegative_integer,
    configure_memory_efficient_mttkrp,
    cp_conv_parameter_count,
    dense_conv_parameter_count,
)


STABILITY_METHOD = "cp_rank512_stability"
PRIMARY_PRECISION = "float32"
FLOAT64_PRECISION = "float64"
DEFAULT_BUDGETS = (150, 200, 250, 300, 350, 400, 500, 600, 800)


def _optional_bool(value):
    if value in (None, ""):
        return value
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def scientific_rank512_stability_key(row):
    method = str(row.get("method", "")).strip()
    if method != STABILITY_METHOD:
        raise ValueError(f"unsupported stability method label {method!r}")
    rank = _normalize_nonnegative_integer(row.get("rank"), "rank", positive=True)
    seed = _normalize_nonnegative_integer(row.get("seed"), "seed")
    budget = _normalize_nonnegative_integer(row.get("iteration_budget"), "iteration_budget", positive=True)
    repetition = _normalize_nonnegative_integer(row.get("repetition"), "repetition")
    precision = str(row.get("optimization_precision", "")).strip().lower()
    if precision not in {PRIMARY_PRECISION, FLOAT64_PRECISION}:
        raise ValueError(f"unsupported optimization precision {precision!r}")
    return method, rank, seed, budget, repetition, precision


def normalize_rank512_stability_rows(
    rows, context="rank-512 stability rows", *, warn=True
):
    normalized_by_key = {}
    rejected = []
    diagnostics = []
    for index, original in enumerate(rows):
        row = dict(original)
        try:
            key = scientific_rank512_stability_key(row)
            (
                row["method"],
                row["rank"],
                row["seed"],
                row["iteration_budget"],
                row["repetition"],
                row["optimization_precision"],
            ) = key
            for field in (
                "completed_requested_budget",
                "initialization_identity_verified",
                "residual_is_fresh_for_scientific_key",
                "reconstruction_all_finite",
                "tltorch_reconstruction_all_finite",
                "manual_reconstruction_all_finite",
                "residual_verification_passed",
                "factor_diagnostics_finite",
                "reconstruction_bounded_relative_to_dense",
                "factor_degeneracy_detected",
                "canonical_sensitivity_residual_reproduced_within_tolerance",
                "shared_initialization_matches_canonical_sensitivity",
            ):
                if field in row:
                    row[field] = _optional_bool(row[field])
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
                    "repetition": row.get("repetition", ""),
                    "optimization_precision": row.get("optimization_precision", ""),
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
                    "rank": key[1],
                    "seed": key[2],
                    "iteration_budget": key[3],
                    "repetition": key[4],
                    "optimization_precision": key[5],
                }
            )
            if previous.get("status") == "completed" and row.get("status") != "completed":
                continue
        normalized_by_key[key] = row
    if diagnostics and warn:
        warnings.warn(
            f"{context}: excluded or deduplicated {len(diagnostics)} row(s); details are in metadata",
            RuntimeWarning,
            stacklevel=2,
        )
    normalized = sorted(
        normalized_by_key.values(),
        key=lambda row: (
            row["optimization_precision"],
            int(row["seed"]),
            int(row["iteration_budget"]),
            int(row["repetition"]),
        ),
    )
    return normalized, rejected, diagnostics


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def factor_scaling_diagnostics(
    fitted,
    *,
    near_zero_threshold=1e-12,
    extreme_norm_threshold=1e6,
    scaling_spread_threshold=1e6,
):
    weights = fitted.weight.weights.detach().cpu().to(torch.float64)
    factors = [factor.detach().cpu().to(torch.float64) for factor in fitted.weight.factors]
    component_norms = [torch.abs(weights)] + [torch.linalg.vector_norm(factor, dim=0) for factor in factors]
    stacked = torch.stack(component_norms)
    safe = torch.clamp(stacked, min=torch.finfo(torch.float64).tiny)
    log_products = torch.sum(torch.log(safe), dim=0)
    contribution_norms = torch.exp(torch.clamp(log_products, max=math.log(torch.finfo(torch.float64).max)))
    minimum = torch.min(stacked, dim=0).values
    maximum = torch.max(stacked, dim=0).values
    spreads = maximum / torch.clamp(minimum, min=torch.finfo(torch.float64).tiny)
    near_zero = torch.any(stacked <= float(near_zero_threshold), dim=0)
    extreme = torch.any(stacked >= float(extreme_norm_threshold), dim=0)
    extreme_spread = spreads >= float(scaling_spread_threshold)
    flattened = {
        "component_log_norm_product_min": float(torch.min(log_products)),
        "component_log_norm_product_max": float(torch.max(log_products)),
        "component_log_norm_product_median": float(torch.median(log_products)),
        "component_contribution_norm_min": float(torch.min(contribution_norms)),
        "component_contribution_norm_max": float(torch.max(contribution_norms)),
        "component_scaling_spread_max": float(torch.max(spreads)),
        "component_scaling_spread_median": float(torch.median(spreads)),
        "near_zero_component_count": int(torch.sum(near_zero)),
        "extremely_large_component_count": int(torch.sum(extreme)),
        "extreme_scaling_spread_component_count": int(torch.sum(extreme_spread)),
        "component_scaling_spreads_json": json.dumps(spreads.tolist()),
        "component_log_norm_products_json": json.dumps(log_products.tolist()),
        "component_contribution_norms_json": json.dumps(contribution_norms.tolist()),
        "factor_diagnostics_finite": bool(
            torch.isfinite(stacked).all()
            and torch.isfinite(log_products).all()
            and torch.isfinite(spreads).all()
        ),
    }
    factor_records = []
    for index, factor in enumerate(factors):
        norms = component_norms[index + 1]
        record = {
            "factor_index": index,
            "shape": list(factor.shape),
            "max_abs_value": float(torch.max(torch.abs(factor))),
            "column_norm_min": float(torch.min(norms)),
            "column_norm_max": float(torch.max(norms)),
        }
        factor_records.append(record)
        flattened[f"factor_{index}_max_abs_value"] = record["max_abs_value"]
        flattened[f"factor_{index}_column_norm_min"] = record["column_norm_min"]
        flattened[f"factor_{index}_column_norm_max"] = record["column_norm_max"]
    flattened["component_weight_max_abs_value"] = float(torch.max(torch.abs(weights)))
    flattened["component_weight_abs_min"] = float(torch.min(torch.abs(weights)))
    flattened["component_weight_abs_max"] = float(torch.max(torch.abs(weights)))
    flattened["factor_diagnostics_json"] = json.dumps(factor_records, sort_keys=True)
    flattened["_component_contribution_norms"] = contribution_norms
    flattened["_component_extreme_mask"] = extreme
    flattened["_component_extreme_spread_mask"] = extreme_spread
    return flattened


def verify_cp_reconstruction(
    reference,
    approximation,
    fitted,
    *,
    tolerance=1e-5,
    output_chunk_size=8,
):
    '''Compare tltorch reconstruction with a manual chunked CP contraction in float64.'''
    reference = reference.detach().cpu()
    approximation = approximation.detach().cpu()
    weights = fitted.weight.weights.detach().cpu().to(torch.float64)
    factors = [factor.detach().cpu().to(torch.float64) for factor in fitted.weight.factors]
    if len(factors) != 4:
        raise ValueError(f"Expected four CP factors, got {len(factors)}")
    sums = defaultdict(float)
    maxima = defaultdict(float)
    all_finite = True
    for start in range(0, int(reference.shape[0]), int(output_chunk_size)):
        stop = min(start + int(output_chunk_size), int(reference.shape[0]))
        ref = reference[start:stop].to(torch.float64)
        first = approximation[start:stop].to(torch.float64)
        second = torch.einsum(
            "or,ir,hr,wr,r->oihw",
            factors[0][start:stop],
            factors[1],
            factors[2],
            factors[3],
            weights,
        )
        finite = bool(torch.isfinite(ref).all() and torch.isfinite(first).all() and torch.isfinite(second).all())
        all_finite = all_finite and finite
        if not finite:
            continue
        residual_first = ref - first
        residual_second = ref - second
        path_difference = first - second
        sums["dense_squared"] += float(torch.sum(ref.square()))
        sums["first_residual_squared"] += float(torch.sum(residual_first.square()))
        sums["second_residual_squared"] += float(torch.sum(residual_second.square()))
        sums["path_difference_squared"] += float(torch.sum(path_difference.square()))
        sums["first_reconstruction_squared"] += float(torch.sum(first.square()))
        sums["second_reconstruction_squared"] += float(torch.sum(second.square()))
        maxima["first_reconstruction"] = max(
            maxima["first_reconstruction"], float(torch.max(torch.abs(first)))
        )
        maxima["second_reconstruction"] = max(
            maxima["second_reconstruction"], float(torch.max(torch.abs(second)))
        )
        maxima["path_difference"] = max(
            maxima["path_difference"], float(torch.max(torch.abs(path_difference)))
        )
    if not all_finite or sums["dense_squared"] <= 0:
        return {
            "reconstruction_all_finite": False,
            "residual_verification_passed": False,
            "actual_relative_squared_frobenius_error": float("nan"),
            "actual_relative_frobenius_error": float("nan"),
        }
    denominator = sums["dense_squared"]
    first_squared = sums["first_residual_squared"] / denominator
    second_squared = sums["second_residual_squared"] / denominator
    difference = abs(first_squared - second_squared)
    return {
        "reconstruction_all_finite": True,
        "tltorch_reconstruction_all_finite": bool(torch.isfinite(approximation).all()),
        "manual_reconstruction_all_finite": True,
        "actual_relative_squared_frobenius_error": first_squared,
        "actual_relative_frobenius_error": math.sqrt(max(first_squared, 0.0)),
        "manual_relative_squared_frobenius_error": second_squared,
        "manual_relative_frobenius_error": math.sqrt(max(second_squared, 0.0)),
        "residual_calculation_absolute_difference": difference,
        "residual_verification_passed": difference <= tolerance,
        "reconstruction_path_relative_squared_difference": sums[
            "path_difference_squared"
        ]
        / denominator,
        "reconstruction_path_max_abs_difference": maxima["path_difference"],
        "dense_tensor_norm": math.sqrt(denominator),
        "reconstructed_tensor_norm": math.sqrt(sums["first_reconstruction_squared"]),
        "manual_reconstructed_tensor_norm": math.sqrt(sums["second_reconstruction_squared"]),
        "residual_tensor_norm": math.sqrt(sums["first_residual_squared"]),
        "maximum_absolute_reconstructed_tensor_entry": maxima["first_reconstruction"],
        "manual_maximum_absolute_reconstructed_tensor_entry": maxima["second_reconstruction"],
        "residual_evaluation_precision": "float64",
        "residual_evaluation_path": "tltorch_dense_and_manual_chunked_cp_contraction",
    }


def finalize_degeneracy_indicators(
    factor_diagnostics,
    reconstruction_diagnostics,
    *,
    bounded_reconstruction_multiple=10.0,
    cancellation_ratio_threshold=1e3,
):
    contribution_norms = factor_diagnostics.pop("_component_contribution_norms")
    extreme = factor_diagnostics.pop("_component_extreme_mask")
    extreme_spread = factor_diagnostics.pop("_component_extreme_spread_mask")
    reconstructed_norm = float(reconstruction_diagnostics.get("reconstructed_tensor_norm", float("nan")))
    dense_norm = float(reconstruction_diagnostics.get("dense_tensor_norm", float("nan")))
    finite_norm = math.isfinite(reconstructed_norm) and reconstructed_norm > 0
    cancellation_ratio = (
        float(torch.sum(contribution_norms)) / reconstructed_norm if finite_norm else float("inf")
    )
    maximum_ratio = (
        float(torch.max(contribution_norms)) / reconstructed_norm if finite_norm else float("inf")
    )
    bounded_reconstruction = (
        finite_norm
        and math.isfinite(dense_norm)
        and reconstructed_norm <= bounded_reconstruction_multiple * dense_norm
    )
    extreme_bounded = int(torch.sum((extreme | extreme_spread) & bounded_reconstruction))
    factor_diagnostics.update(
        {
            "component_contribution_sum_to_reconstructed_norm_ratio": cancellation_ratio,
            "component_contribution_max_to_reconstructed_norm_ratio": maximum_ratio,
            "reconstruction_bounded_relative_to_dense": bounded_reconstruction,
            "extreme_scaling_with_bounded_reconstruction_count": extreme_bounded,
            "factor_degeneracy_detected": bool(
                factor_diagnostics["extremely_large_component_count"] > 0
                or factor_diagnostics["extreme_scaling_spread_component_count"] > 0
                or cancellation_ratio >= cancellation_ratio_threshold
            ),
            "cancellation_ratio_threshold": cancellation_ratio_threshold,
            "bounded_reconstruction_multiple": bounded_reconstruction_multiple,
        }
    )
    return factor_diagnostics


def clone_conv_with_precision(conv, precision):
    dtype = torch.float32 if precision == PRIMARY_PRECISION else torch.float64
    cloned = copy.deepcopy(conv).cpu().to(dtype=dtype)
    return cloned


def captured_shared_initialization(
    conv,
    rank,
    seed,
    init,
    device,
    optimization_precision,
    *,
    memory_efficient_mttkrp,
    mttkrp_rank_chunk_size,
    mttkrp_max_explicit_bytes,
):
    '''Create one float32 random initialization and replay its values at either precision.'''
    set_seed(seed, deterministic=True)
    configure_memory_efficient_mttkrp(
        memory_efficient_mttkrp,
        rank_chunk_size=mttkrp_rank_chunk_size,
        max_explicit_bytes=mttkrp_max_explicit_bytes,
    )
    probe_source = clone_conv_with_precision(conv, PRIMARY_PRECISION).to(device)
    probe = factorized_conv_from_conv(
        probe_source,
        rank=rank,
        factorization="cp",
        init=init,
        n_iter_max=0,
    ).cpu()
    base_weights = probe.weight.weights.detach().cpu().to(torch.float32).clone()
    base_factors = [
        factor.detach().cpu().to(torch.float32).clone()
        for factor in probe.weight.factors
    ]
    shared_hash = _cp_components_hash(base_weights, base_factors)
    dtype = torch.float32 if optimization_precision == PRIMARY_PRECISION else torch.float64
    weights = base_weights.to(device=device, dtype=dtype)
    factors = [factor.to(device=device, dtype=dtype) for factor in base_factors]
    precision_hash = _cp_components_hash(weights, factors)
    del probe_source, probe, base_weights, base_factors
    gc.collect()
    return (weights, factors), shared_hash, precision_hash


def fit_rank512_stability_row(
    conv,
    reference_weight,
    *,
    rank,
    seed,
    budget,
    repetition,
    optimization_precision,
    init,
    device,
    checkpoint_hash,
    target_tensor_hash,
    max_unfolding_tail_bound_squared,
    output_mode_svd_residual_squared,
    tolerance,
    memory_efficient_mttkrp,
    mttkrp_rank_chunk_size,
    mttkrp_max_explicit_bytes,
    expected_initialization_hash=None,
    expected_shared_initialization_hash=None,
    thresholds=None,
):
    thresholds = dict(thresholds or {})
    source = clone_conv_with_precision(conv, optimization_precision)
    explicit_initialization, shared_initialization_hash, initialization_hash = (
        captured_shared_initialization(
        conv,
        rank,
        seed,
        init,
        device,
        optimization_precision,
        memory_efficient_mttkrp=memory_efficient_mttkrp,
        mttkrp_rank_chunk_size=mttkrp_rank_chunk_size,
        mttkrp_max_explicit_bytes=mttkrp_max_explicit_bytes,
        )
    )
    if expected_initialization_hash and initialization_hash != expected_initialization_hash:
        raise AssertionError(
            f"initialization hash mismatch for seed={seed}, precision={optimization_precision}: "
            f"{initialization_hash} != {expected_initialization_hash}"
        )
    if (
        expected_shared_initialization_hash
        and shared_initialization_hash != expected_shared_initialization_hash
    ):
        raise AssertionError(
            f"shared initialization hash mismatch for seed={seed}: "
            f"{shared_initialization_hash} != {expected_shared_initialization_hash}"
        )
    seed_status = set_seed(seed, deterministic=True)
    fitted, approximation, runtime, mttkrp_implementation, actual_hash = (
        fit_cp_approximation_with_initialization_capture(
            source,
            rank,
            seed,
            explicit_initialization,
            budget,
            device,
            memory_efficient_mttkrp,
            mttkrp_rank_chunk_size,
            mttkrp_max_explicit_bytes,
        )
    )
    if actual_hash != initialization_hash:
        raise AssertionError(
            f"actual fitted initialization {actual_hash} != probe {initialization_hash}"
        )
    factor_diagnostics = factor_scaling_diagnostics(
        fitted,
        near_zero_threshold=float(thresholds.get("near_zero_component_norm_threshold", 1e-12)),
        extreme_norm_threshold=float(thresholds.get("extreme_factor_norm_threshold", 1e6)),
        scaling_spread_threshold=float(thresholds.get("scaling_spread_threshold", 1e6)),
    )
    reconstruction_diagnostics = verify_cp_reconstruction(
        reference_weight,
        approximation,
        fitted,
        tolerance=tolerance,
        output_chunk_size=int(thresholds.get("residual_output_chunk_size", 8)),
    )
    factor_diagnostics = finalize_degeneracy_indicators(
        factor_diagnostics,
        reconstruction_diagnostics,
        bounded_reconstruction_multiple=float(
            thresholds.get("bounded_reconstruction_multiple", 10.0)
        ),
        cancellation_ratio_threshold=float(
            thresholds.get("cancellation_ratio_threshold", 1e3)
        ),
    )
    finite = bool(
        reconstruction_diagnostics.get("reconstruction_all_finite")
        and factor_diagnostics.get("factor_diagnostics_finite")
    )
    verification_passed = bool(
        finite and reconstruction_diagnostics.get("residual_verification_passed")
    )
    final_factor_hash = _cp_components_hash(
        fitted.weight.weights.detach().cpu(),
        [factor.detach().cpu() for factor in fitted.weight.factors],
    )
    residual_identifier = hashlib.sha256(
        (
            f"{checkpoint_hash}|{target_tensor_hash}|{rank}|{seed}|{budget}|{repetition}|"
            f"{optimization_precision}|{final_factor_hash}"
        ).encode("utf-8")
    ).hexdigest()
    shape = tuple(reference_weight.shape)
    parameters = cp_conv_parameter_count(shape, rank, conv.bias is not None)
    actual_parameters = sum(parameter.numel() for parameter in fitted.parameters())
    if actual_parameters != parameters:
        raise AssertionError(f"CP parameter count {actual_parameters} != {parameters}")
    row = {
        "method": STABILITY_METHOD,
        "rank": int(rank),
        "seed": int(seed),
        "iteration_budget": int(budget),
        "repetition": int(repetition),
        "optimization_precision": optimization_precision,
        "initializer": init,
        "initialization_replay_protocol": "shared_float32_factors_independently_replayed",
        "shared_float32_initialization_hash_sha256": shared_initialization_hash,
        "initialization_hash_sha256": initialization_hash,
        "actual_fit_initialization_hash_sha256": actual_hash,
        "initialization_identity_verified": actual_hash == initialization_hash,
        "final_factor_hash_sha256": final_factor_hash,
        "residual_evaluation_identifier_sha256": residual_identifier,
        "residual_is_fresh_for_scientific_key": True,
        "decomposition_runtime_seconds": runtime,
        "completed_requested_budget": True,
        "convergence_status": "completed_requested_budget_iteration_history_unavailable",
        "status": "completed" if verification_passed else "failed",
        "failure_exception": "" if verification_passed else "nonfinite or inconsistent residual reconstruction",
        "device": torch.device(device).type,
        "deterministic_settings": json.dumps(seed_status, sort_keys=True),
        "checkpoint_sha256": checkpoint_hash,
        "target_tensor_sha256": target_tensor_hash,
        "max_unfolding_tail_bound_squared": max_unfolding_tail_bound_squared,
        "output_mode_svd_residual_squared": output_mode_svd_residual_squared,
        "dense_tensor_shape": "x".join(map(str, shape)),
        "parameter_count": parameters,
        "optimization_dtype": str(next(fitted.parameters()).dtype),
        "mttkrp_implementation": mttkrp_implementation,
        "mttkrp_rank_chunk_size": int(mttkrp_rank_chunk_size),
        "mttkrp_max_explicit_bytes": int(mttkrp_max_explicit_bytes),
        **reconstruction_diagnostics,
        **factor_diagnostics,
    }
    del source, fitted, approximation
    gc.collect()
    return row


def aggregate_rank512_stability_rows(rows, tolerance=1e-5):
    normalized, _, diagnostics = normalize_rank512_stability_rows(
        rows, context="rank-512 stability aggregation", warn=False
    )
    completed = [row for row in normalized if row.get("status") == "completed"]
    seed_budget = defaultdict(list)
    for row in completed:
        seed_budget[
            (
                row["optimization_precision"],
                int(row["seed"]),
                int(row["iteration_budget"]),
            )
        ].append(row)
    seed_summaries = {}
    for key, group in seed_budget.items():
        residuals = np.asarray(
            [float(row["actual_relative_squared_frobenius_error"]) for row in group],
            dtype=np.float64,
        )
        final_factor_hashes = {
            str(row.get("final_factor_hash_sha256", "")).strip()
            for row in group
            if str(row.get("final_factor_hash_sha256", "")).strip()
        }
        seed_summaries[key] = {
            "mean": float(residuals.mean()),
            "min": float(residuals.min()),
            "max": float(residuals.max()),
            "range": float(residuals.max() - residuals.min()),
            "repetition_count": len(residuals),
            "reproducible_within_tolerance": (
                bool(residuals.max() - residuals.min() <= tolerance)
                if len(residuals) >= 2
                else None
            ),
            "final_factor_hashes_identical": (
                len(final_factor_hashes) == 1
                if len(residuals) >= 2 and final_factor_hashes
                else None
            ),
            "rows": group,
        }
    grouped = defaultdict(list)
    for (precision, seed, budget), summary in seed_summaries.items():
        grouped[(precision, budget)].append((seed, summary))
    output = []
    for (precision, budget), items in sorted(grouped.items()):
        means = np.asarray([item[1]["mean"] for item in items], dtype=np.float64)
        rows_for_budget = [row for _, item in items for row in item["rows"]]
        repeated = [item[1] for item in items if item[1]["repetition_count"] >= 2]
        output.append(
            {
                "method": STABILITY_METHOD,
                "rank": int(rows_for_budget[0]["rank"]),
                "optimization_precision": precision,
                "iteration_budget": int(budget),
                "completed_seed_count": len(items),
                "completed_repetition_row_count": len(rows_for_budget),
                "actual_relative_squared_frobenius_error_mean": float(means.mean()),
                "actual_relative_squared_frobenius_error_std_population": float(means.std(ddof=0)),
                "actual_relative_squared_frobenius_error_min": float(means.min()),
                "actual_relative_squared_frobenius_error_max": float(means.max()),
                "actual_relative_squared_frobenius_error_seed_range": float(means.max() - means.min()),
                "maximum_within_seed_repetition_range": max(
                    (item[1]["range"] for item in items), default=0.0
                ),
                "repeated_seed_count": len(repeated),
                "all_repeated_results_reproducible_within_tolerance": (
                    all(item["reproducible_within_tolerance"] for item in repeated)
                    if repeated
                    else ""
                ),
                "all_repeated_final_factor_hashes_identical": (
                    all(item["final_factor_hashes_identical"] for item in repeated)
                    if repeated
                    and all(
                        item["final_factor_hashes_identical"] is not None
                        for item in repeated
                    )
                    else ""
                ),
                "residual_verification_all_passed": all(
                    bool(row.get("residual_verification_passed")) for row in rows_for_budget
                ),
                "nonfinite_row_count": sum(
                    not bool(row.get("reconstruction_all_finite")) for row in rows_for_budget
                ),
                "factor_degeneracy_row_count": sum(
                    bool(row.get("factor_degeneracy_detected")) for row in rows_for_budget
                ),
                "component_scaling_spread_max": max(
                    float(row.get("component_scaling_spread_max", 0.0)) for row in rows_for_budget
                ),
                "cancellation_ratio_max": max(
                    float(
                        row.get(
                            "component_contribution_sum_to_reconstructed_norm_ratio", 0.0
                        )
                    )
                    for row in rows_for_budget
                ),
                "maximum_absolute_factor_value": max(
                    max(
                        float(row.get(f"factor_{index}_max_abs_value", 0.0))
                        for index in range(4)
                    )
                    for row in rows_for_budget
                ),
            }
        )
    by_precision = defaultdict(list)
    for row in output:
        by_precision[row["optimization_precision"]].append(row)
    for precision, records in by_precision.items():
        records.sort(key=lambda row: int(row["iteration_budget"]))
        previous = None
        for record in records:
            if previous is not None:
                change = (
                    previous["actual_relative_squared_frobenius_error_mean"]
                    - record["actual_relative_squared_frobenius_error_mean"]
                )
                record["previous_iteration_budget"] = previous["iteration_budget"]
                record["mean_squared_residual_reduction_from_previous_budget"] = change
                record["mean_squared_residual_deteriorated_from_previous_budget"] = change < -tolerance
            previous = record
    return output, seed_summaries, diagnostics


def write_rank512_stability_figure(rows, figures_dir):
    normalized, _, diagnostics = normalize_rank512_stability_rows(
        rows, context="rank-512 stability figure"
    )
    completed = [row for row in normalized if row.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed rank-512 stability rows are available")
    frame = pd.DataFrame(completed)
    numeric = [
        "seed",
        "iteration_budget",
        "repetition",
        "actual_relative_squared_frobenius_error",
        "output_mode_svd_residual_squared",
        "max_unfolding_tail_bound_squared",
    ]
    for field in numeric:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(
        subset=["seed", "iteration_budget", "repetition", "actual_relative_squared_frobenius_error"]
    )
    primary = frame[frame["optimization_precision"] == PRIMARY_PRECISION].copy()
    if primary.empty:
        raise ValueError("No completed float32 stability rows are available")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    figure_rows = []
    seed_colors = plt.get_cmap("tab10")
    for seed_index, (seed, group) in enumerate(primary.groupby("seed")):
        for repetition, repetition_rows in group.groupby("repetition"):
            values = repetition_rows.sort_values("iteration_budget")
            x = values["iteration_budget"].to_numpy(dtype=float)
            y = values["actual_relative_squared_frobenius_error"].to_numpy(dtype=float)
            offset = 1.0 + 0.012 * (int(repetition) - 0.5)
            ax.scatter(
                x * offset,
                y,
                facecolors="none" if int(repetition) == 0 else seed_colors(seed_index),
                edgecolors=seed_colors(seed_index),
                s=34,
                linewidths=1.1,
                alpha=0.9,
                label=f"seed {int(seed)}, repetition {int(repetition)}",
            )
            for budget, plotted, residual in zip(x, x * offset, y):
                figure_rows.append(
                    {
                        "series": "float32_individual",
                        "rank": int(values["rank"].iloc[0]),
                        "seed": int(seed),
                        "repetition": int(repetition),
                        "optimization_precision": PRIMARY_PRECISION,
                        "iteration_budget": int(budget),
                        "plotted_iteration_budget": float(plotted),
                        "normalized_squared_frobenius_residual": float(residual),
                    }
                )
    seed_budget_means = primary.groupby(["seed", "iteration_budget"], as_index=False)[
        "actual_relative_squared_frobenius_error"
    ].mean()
    means = seed_budget_means.groupby("iteration_budget", as_index=False)[
        "actual_relative_squared_frobenius_error"
    ].mean()
    ax.plot(
        means["iteration_budget"],
        means["actual_relative_squared_frobenius_error"],
        color="#111111",
        linewidth=2.2,
        marker="o",
        label="float32 mean",
    )
    for _, row in means.iterrows():
        figure_rows.append(
            {
                "series": "float32_mean",
                "rank": int(primary["rank"].iloc[0]),
                "seed": "",
                "repetition": "",
                "optimization_precision": PRIMARY_PRECISION,
                "iteration_budget": int(row["iteration_budget"]),
                "plotted_iteration_budget": float(row["iteration_budget"]),
                "normalized_squared_frobenius_residual": float(
                    row["actual_relative_squared_frobenius_error"]
                ),
            }
        )
    high_precision = frame[frame["optimization_precision"] == FLOAT64_PRECISION]
    if not high_precision.empty:
        ax.scatter(
            high_precision["iteration_budget"],
            high_precision["actual_relative_squared_frobenius_error"],
            marker="D",
            color="#CC79A7",
            s=40,
            label="float64 optimization",
        )
        for _, row in high_precision.iterrows():
            figure_rows.append(
                {
                    "series": "float64_optimization",
                    "rank": int(row["rank"]),
                    "seed": int(row["seed"]),
                    "repetition": int(row["repetition"]),
                    "optimization_precision": FLOAT64_PRECISION,
                    "iteration_budget": int(row["iteration_budget"]),
                    "plotted_iteration_budget": float(row["iteration_budget"]),
                    "normalized_squared_frobenius_residual": float(
                        row["actual_relative_squared_frobenius_error"]
                    ),
                }
            )
    svd = float(primary["output_mode_svd_residual_squared"].iloc[0])
    bound = float(primary["max_unfolding_tail_bound_squared"].iloc[0])
    ax.axhline(svd, color="#D55E00", linestyle="--", linewidth=1.4, label="Matrix-SVD reference")
    ax.axhline(bound, color="#666666", linestyle=":", linewidth=1.4, label="Strongest unfolding bound")
    for series, value in (("matrix_svd_reference", svd), ("strongest_unfolding_bound", bound)):
        figure_rows.append(
            {
                "series": series,
                "rank": int(primary["rank"].iloc[0]),
                "seed": "",
                "repetition": "",
                "optimization_precision": "",
                "iteration_budget": "",
                "plotted_iteration_budget": "",
                "normalized_squared_frobenius_residual": value,
            }
        )
    budgets = sorted(int(value) for value in primary["iteration_budget"].unique())
    ax.set_xscale("log")
    ax.set_xticks(budgets, [str(value) for value in budgets])
    ax.set_xlabel("Requested CP iteration budget")
    ax.set_ylabel("Normalized squared Frobenius residual")
    ax.set_title("Rank-512 CP reconstruction stability")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = figures_dir / "rank512_stability.csv"
    write_csv(data_path, figure_rows)
    paths = [str(data_path)]
    for suffix in ("pdf", "png"):
        path = figures_dir / f"rank512_stability.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths, diagnostics


def write_rank512_stability_audit(
    rows,
    aggregates,
    path,
    *,
    budgets,
    seeds,
    tolerance,
    expected_primary_keys,
    expected_float64_keys,
    failures,
    repeatability_budgets=(200, 400),
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized, _, _ = normalize_rank512_stability_rows(rows, warn=False)
    scientific = {scientific_rank512_stability_key(row): row for row in normalized}
    primary_complete = all(
        key in scientific and scientific[key].get("status") == "completed"
        for key in expected_primary_keys
    )
    float64_attempted = all(key in scientific for key in expected_float64_keys)
    complete = primary_complete and float64_attempted
    frame = pd.DataFrame(aggregates)
    primary = frame[frame["optimization_precision"] == PRIMARY_PRECISION].sort_values(
        "iteration_budget"
    ) if not frame.empty else pd.DataFrame()
    lines = [
        "# Rank-512 CP numerical-stability audit",
        "",
        f"Status: **{'complete' if complete else 'incomplete'}**.",
        "",
        "Completed requested budgets are reported without a convergence claim. Float32 fits are evaluated "
        "in float64 through both the fitted layer reconstruction and an independent chunked CP contraction.",
        "",
        "## Budget summary",
        "",
        "| Precision | Budget | Mean squared residual | Seed range | Max repetition range | Degeneracy rows | Max cancellation ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(["optimization_precision", "iteration_budget"]).to_dict("records"):
        lines.append(
            f"| {row['optimization_precision']} | {int(row['iteration_budget'])} | "
            f"{row['actual_relative_squared_frobenius_error_mean']:.9f} | "
            f"{row['actual_relative_squared_frobenius_error_seed_range']:.9f} | "
            f"{row['maximum_within_seed_repetition_range']:.3e} | "
            f"{int(row['factor_degeneracy_row_count'])} | {row['cancellation_ratio_max']:.3e} |"
        )
    lines.extend(["", "## Scientific decisions", ""])
    repeatability_budgets = tuple(int(value) for value in repeatability_budgets)
    repeatability_label = " and ".join(map(str, repeatability_budgets))
    if not primary_complete or primary.empty:
        lines.extend(
            [
                f"1. Repeatability at {repeatability_label}: deferred.",
                "2. Deterioration onset: deferred.",
                "3. Seed specificity: deferred.",
                "4. Factor-degeneracy association: deferred.",
                "5. Residual verification: deferred.",
                "6. Precision effect: deferred until the labeled float64 subset is attempted.",
                "7. Budget-800 behavior: deferred.",
                "8. Downstream common budget: deferred.",
                "9. Rank-512 downstream inclusion: deferred.",
                "10. Additional fitting work: complete the requested diagnostic grid.",
            ]
        )
    else:
        primary_rows = [
            row for row in normalized
            if row.get("status") == "completed" and row["optimization_precision"] == PRIMARY_PRECISION
        ]
        repeat_groups = defaultdict(list)
        repeat_factor_hashes = defaultdict(set)
        for row in primary_rows:
            if int(row["iteration_budget"]) in set(repeatability_budgets):
                repeat_key = (int(row["seed"]), int(row["iteration_budget"]))
                repeat_groups[repeat_key].append(
                    float(row["actual_relative_squared_frobenius_error"])
                )
                repeat_factor_hashes[repeat_key].add(
                    str(row.get("final_factor_hash_sha256", ""))
                )
        repeatability = {
            key: max(values) - min(values) <= tolerance and len(values) >= 2
            for key, values in repeat_groups.items()
        }
        repeatability_complete = len(repeatability) == len(seeds) * len(
            repeatability_budgets
        )
        repeatable = repeatability_complete and all(repeatability.values())
        repeated_factor_hashes_identical = repeatability_complete and all(
            len(repeat_factor_hashes[key]) == 1 for key in repeatability
        )
        records = primary.set_index("iteration_budget").to_dict("index")
        ordered = [budget for budget in budgets if budget in records]
        deterioration_budget = None
        adjacent_changes = []
        for lower, upper in zip(ordered, ordered[1:]):
            change = (
                records[upper]["actual_relative_squared_frobenius_error_mean"]
                - records[lower]["actual_relative_squared_frobenius_error_mean"]
            )
            adjacent_changes.append((lower, upper, change))
            if change > tolerance and deterioration_budget is None:
                deterioration_budget = upper
        has_improvement = any(change < -tolerance for _, _, change in adjacent_changes)
        has_deterioration = any(change > tolerance for _, _, change in adjacent_changes)
        oscillates = has_improvement and has_deterioration
        largest_increase = max(adjacent_changes, key=lambda item: item[2]) if adjacent_changes else None
        per_seed = defaultdict(dict)
        for row in primary_rows:
            per_seed[int(row["seed"])].setdefault(int(row["iteration_budget"]), []).append(
                float(row["actual_relative_squared_frobenius_error"])
            )
        lower_repeatability_budget = min(repeatability_budgets)
        upper_repeatability_budget = max(repeatability_budgets)
        worsened_seeds = []
        for seed, values in per_seed.items():
            if (
                lower_repeatability_budget in values
                and upper_repeatability_budget in values
                and np.mean(values[upper_repeatability_budget])
                - np.mean(values[lower_repeatability_budget])
                > tolerance
            ):
                worsened_seeds.append(seed)
        later_rows = [
            row for row in primary_rows
            if deterioration_budget is not None and int(row["iteration_budget"]) >= deterioration_budget
        ]
        degeneracy_associated = bool(later_rows) and any(
            bool(row.get("factor_degeneracy_detected")) for row in later_rows
        )
        residual_correct = all(
            bool(row.get("residual_verification_passed"))
            and bool(row.get("reconstruction_all_finite"))
            and bool(row.get("residual_is_fresh_for_scientific_key"))
            for row in primary_rows
        )
        identity_correct = (
            len({row.get("checkpoint_sha256") for row in primary_rows}) == 1
            and len({row.get("target_tensor_sha256") for row in primary_rows}) == 1
            and all(
                row.get("initialization_hash_sha256")
                == row.get("actual_fit_initialization_hash_sha256")
                for row in primary_rows
            )
        )
        residual_correct = residual_correct and identity_correct
        float64_rows = [
            row for row in normalized if row["optimization_precision"] == FLOAT64_PRECISION
        ]
        successful_float64 = [row for row in float64_rows if row.get("status") == "completed"]
        precision_differences = []
        for high_precision_row in successful_float64:
            matches = [
                row
                for row in primary_rows
                if int(row["seed"]) == int(high_precision_row["seed"])
                and int(row["iteration_budget"])
                == int(high_precision_row["iteration_budget"])
                and int(row["repetition"]) == int(high_precision_row["repetition"])
            ]
            if matches:
                precision_differences.append(
                    float(high_precision_row["actual_relative_squared_frobenius_error"])
                    - float(matches[0]["actual_relative_squared_frobenius_error"])
                )
        if precision_differences:
            precision_text = (
                f"{len(successful_float64)}/{len(expected_float64_keys)} float64 optimization rows completed. "
                "Float64-minus-float32 squared-residual differences range from "
                f"{min(precision_differences):.3e} to {max(precision_differences):.3e}."
            )
        elif successful_float64:
            precision_text = (
                f"{len(successful_float64)}/{len(expected_float64_keys)} float64 optimization rows completed, "
                "but matching float32 rows are not yet available."
            )
        else:
            precision_text = "Full float64 optimization was attempted but produced no completed rows; all float32 factors and residuals were still evaluated in float64."
        behavior_800 = "unavailable"
        if 800 in records:
            earlier = max(budget for budget in ordered if budget < 800)
            difference = (
                records[800]["actual_relative_squared_frobenius_error_mean"]
                - records[earlier]["actual_relative_squared_frobenius_error_mean"]
            )
            behavior_800 = "better" if difference < -tolerance else "worse" if difference > tolerance else "stable"
            behavior_800 += f" (mean change {difference:+.3e})"
        trend_text = (
            f"Mean residual is monotone non-increasing through budget {ordered[-1]}."
            if deterioration_budget is None
            else f"The first mean increase occurs at budget {deterioration_budget}; oscillation after improvements={oscillates}."
        )
        if largest_increase and largest_increase[2] > tolerance:
            trend_text += (
                f" The largest adjacent increase is {largest_increase[2]:+.3e} from "
                f"{largest_increase[0]} to {largest_increase[1]}."
            )
        scaling_text = "No deterioration interval was available."
        if deterioration_budget is not None:
            preceding_budget = max(budget for budget in ordered if budget < deterioration_budget)
            before = records[preceding_budget]
            after = records[deterioration_budget]
            scaling_text = (
                f"At the first deterioration, max scaling spread changed from "
                f"{before['component_scaling_spread_max']:.3e} to {after['component_scaling_spread_max']:.3e}, "
                f"max cancellation ratio from {before['cancellation_ratio_max']:.3e} to "
                f"{after['cancellation_ratio_max']:.3e}, and degeneracy-row count from "
                f"{int(before['factor_degeneracy_row_count'])} to {int(after['factor_degeneracy_row_count'])}."
            )
        if not residual_correct or not repeatable or not repeated_factor_hashes_identical:
            common_budget = "No downstream budget yet; resolve residual or repeatability failure first."
            rank512_decision = "Do not proceed; retain rank 512 only as an unresolved instability result."
            additional = "A fitting-implementation or determinism investigation remains necessary."
        elif deterioration_budget is not None and deterioration_budget > 200:
            common_budget = "Use 200 iterations as the common stable budget; do not select a per-seed best residual."
            rank512_decision = "Proceed only with the reproducible 200-iteration protocol, explicitly qualified by the later-budget instability."
            additional = (
                "No further budget extension is necessary after the requested 800 row; test a stabilized or alternate CP implementation only if later work requires budgets above 200."
            )
        else:
            common_budget = "Do not select a downstream budget from this implementation until the instability mechanism is resolved."
            rank512_decision = "Retain rank 512 only as an instability result for now."
            additional = "A stabilized or alternate CP fitting implementation is genuinely necessary."
        lines.extend(
            [
                f"1. **Are {repeatability_label} reproducible?** `{repeatable}`; all "
                f"{len(seeds) * len(repeatability_budgets)} seed/budget repeatability groups present=`{repeatability_complete}`; "
                f"final-factor hashes identical within repetitions=`{repeated_factor_hashes_identical}`.",
                f"2. **At what budget does deterioration begin?** `{deterioration_budget}`. {trend_text}",
                f"3. **Is deterioration seed-specific?** Seeds worse at "
                f"{upper_repeatability_budget} than {lower_repeatability_budget}: `{worsened_seeds}`.",
                f"4. **Is deterioration associated with factor-norm explosion or CP degeneracy?** `{degeneracy_associated}` under the recorded scaling and cancellation indicators. {scaling_text}",
                f"5. **Is residual evaluation correct?** `{residual_correct}` across the two reconstruction paths, finite checks, and direct recalculation.",
                f"6. **Does higher precision alter the result?** {precision_text}",
                f"7. **Is budget 800 stable, better, or worse?** `{behavior_800}` relative to the preceding completed budget.",
                f"8. **What common fitting budget should be used downstream?** {common_budget}",
                f"9. **Should rank 512 proceed to activation distortion and fine-tuning?** {rank512_decision}",
                f"10. **Is another fitting experiment necessary?** {additional}",
                "A best-observed-residual selection rule is not adopted: selecting the best budget or seed after observing outcomes introduces selection bias and requires a separately specified, reproducible stopping protocol.",
            ]
        )
    if failures:
        lines.extend(["", "## Recorded failures", ""])
        for failure in failures:
            lines.append(
                f"- seed={failure.get('seed')}, budget={failure.get('iteration_budget')}, "
                f"repetition={failure.get('repetition')}, precision={failure.get('optimization_precision')}: "
                f"`{failure.get('exception')}`"
            )
    path.write_text("\n".join(lines) + "\n")
    return str(path), complete
