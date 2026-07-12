'''Structural reconstruction diagnostics and paper-ready comparison artifacts.'''

import gc
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tensorly as tl
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao as explicit_mttkrp

from src.smallnet.diagnostics import rank_energy_diagnostic, tail_energy
from src.smallnet.factorization import MatrixLowRankConv2d, factorized_conv_from_conv
from src.smallnet.results import write_csv


MODE_NAMES = {
    0: "output channels",
    1: "input channels",
    2: "kernel height",
    3: "kernel width",
}


def normalized_frobenius_residual(reference, approximation):
    '''Return squared and ordinary normalized Frobenius residuals.'''
    if tuple(reference.shape) != tuple(approximation.shape):
        raise ValueError(f"Shape mismatch: {tuple(reference.shape)} != {tuple(approximation.shape)}")
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    approximation64 = approximation.detach().to(device="cpu", dtype=torch.float64)
    denominator = torch.sum(reference64.square())
    if denominator <= 0:
        raise ValueError("Normalized residual is undefined for a zero reference tensor")
    squared = float(torch.sum((reference64 - approximation64).square()) / denominator)
    return squared, math.sqrt(max(squared, 0.0))


def dense_conv_parameter_count(shape, has_bias=True):
    cout, cin, kh, kw = (int(value) for value in shape)
    return cout * cin * kh * kw + (cout if has_bias else 0)


def cp_conv_parameter_count(shape, rank, has_bias=True):
    '''Exact tltorch CP count, including its rank-length component weights.'''
    cout, cin, kh, kw = (int(value) for value in shape)
    rank = int(rank)
    return rank * (cout + cin + kh + kw + 1) + (cout if has_bias else 0)


def matrix_svd_conv_parameter_count(shape, rank, has_bias=True):
    cout, cin, kh, kw = (int(value) for value in shape)
    rank = min(int(rank), cout, cin * kh * kw)
    return rank * (cin * kh * kw + cout) + (cout if has_bias else 0)


def representation_macs(shape, rank, method, output_spatial_elements):
    cout, cin, kh, kw = (int(value) for value in shape)
    pixels = int(output_spatial_elements)
    if method == "dense":
        return pixels * cout * cin * kh * kw
    if method == "cp":
        return pixels * int(rank) * (cin + kh + kw + cout + 1)
    if method == "matrix_svd_output_unfolding":
        return pixels * int(rank) * (cin * kh * kw + cout)
    raise ValueError(f"Unknown method: {method}")


def diagnostic_bounds(diagnostic, rank):
    tails = {
        int(mode): tail_energy(record["cumulative_energy"], rank)
        for mode, record in diagnostic["modes"].items()
    }
    return max(tails.values()), tails.get(0, 0.0), tails


def _rank_chunked_mttkrp(tensor, cp_tensor, mode, rank_chunk_size):
    '''Memory-bounded torch MTTKRP that preserves the CP rank index in chunks.'''
    weights, factors = cp_tensor
    rank = int(factors[0].shape[1])
    other_modes = [index for index in range(tensor.ndim) if index != mode]
    first_mode = max(other_modes, key=lambda index: int(tensor.shape[index]))
    outputs = []
    for start in range(0, rank, int(rank_chunk_size)):
        stop = min(start + int(rank_chunk_size), rank)
        factor = torch.conj(factors[first_mode][:, start:stop])
        component = torch.tensordot(tensor, factor, dims=([first_mode], [0]))
        remaining_modes = [index for index in range(tensor.ndim) if index != first_mode]
        for contracted_mode in [index for index in other_modes if index != first_mode]:
            axis = remaining_modes.index(contracted_mode)
            factor = torch.conj(factors[contracted_mode][:, start:stop])
            broadcast = [1] * component.ndim
            broadcast[axis] = int(factor.shape[0])
            broadcast[-1] = stop - start
            component = (component * factor.reshape(broadcast)).sum(dim=axis)
            remaining_modes.pop(axis)
        if weights is not None:
            component = component * weights[start:stop].reshape(1, -1)
        outputs.append(component)
    return torch.cat(outputs, dim=1)


def configure_memory_efficient_mttkrp(
    enabled,
    rank_chunk_size=64,
    max_explicit_bytes=512 * 1024**2,
):
    if enabled:
        def hybrid_mttkrp(tensor, cp_tensor, mode):
            _, factors = cp_tensor
            rank = int(factors[0].shape[1])
            explicit_bytes = (
                math.prod(int(tensor.shape[index]) for index in range(tensor.ndim) if index != mode)
                * rank
                * tensor.element_size()
            )
            if explicit_bytes <= int(max_explicit_bytes):
                return explicit_mttkrp(tensor, cp_tensor, mode)
            return _rank_chunked_mttkrp(tensor, cp_tensor, mode, rank_chunk_size)

        tl.tenalg.register_backend_method(
            "unfolding_dot_khatri_rao",
            hybrid_mttkrp,
        )
        tl.tenalg.use_dynamic_dispatch()
        return "smallnet_hybrid_memory_bounded_mttkrp"
    tl.tenalg.register_backend_method("unfolding_dot_khatri_rao", explicit_mttkrp)
    tl.tenalg.use_dynamic_dispatch()
    return "tensorly_default_explicit_khatri_rao"


def explicit_khatri_rao_peak_bytes(shape, rank, element_size=4):
    dimensions = [int(value) for value in shape]
    return max(
        math.prod(dimensions[:mode] + dimensions[mode + 1 :]) * int(rank) * int(element_size)
        for mode in range(len(dimensions))
    )


def fit_cp_approximation(
    conv,
    rank,
    seed,
    init,
    n_iter_max,
    device,
    memory_efficient_mttkrp=True,
    mttkrp_rank_chunk_size=64,
    mttkrp_max_explicit_bytes=512 * 1024**2,
):
    '''Fit one CP convolution and explicitly reconstruct its dense kernel.'''
    source = conv.to(device)
    mttkrp_implementation = configure_memory_efficient_mttkrp(
        memory_efficient_mttkrp,
        rank_chunk_size=mttkrp_rank_chunk_size,
        max_explicit_bytes=mttkrp_max_explicit_bytes,
    )
    start = time.perf_counter()
    fitted = factorized_conv_from_conv(
        source,
        rank=rank,
        factorization="cp",
        init=init,
        n_iter_max=n_iter_max,
    )
    runtime = time.perf_counter() - start
    fitted = fitted.cpu()
    approximation = fitted.weight.to_tensor().detach()
    return fitted, approximation, runtime, mttkrp_implementation


def output_mode_svd(weight, device=None):
    requested = torch.device(device or "cpu")
    start = time.perf_counter()
    actual = requested
    matrix = None
    try:
        matrix = weight.detach().to(requested).reshape(weight.shape[0], -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except (RuntimeError, NotImplementedError):
        if requested.type == "cpu":
            raise
        del matrix
        if requested.type == "cuda":
            torch.cuda.empty_cache()
        elif requested.type == "mps":
            torch.mps.empty_cache()
        actual = torch.device("cpu")
        matrix = weight.detach().cpu().reshape(weight.shape[0], -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u, s, vh = u.cpu(), s.cpu(), vh.cpu()
    del matrix
    return u, s, vh, time.perf_counter() - start, actual.type


def verify_matrix_svd_kernel(module, u, s, vh, rank, *, row_chunk=64):
    '''Compare every composed-kernel entry with U_r diag(s_r) Vh_r in row chunks.'''
    first, second = module
    left = second.weight.detach().cpu().reshape(second.out_channels, rank)
    right = first.weight.detach().cpu().reshape(rank, -1)
    expected_left = u[:, :rank] * s[:rank]
    maximum = 0.0
    for start in range(0, left.shape[0], row_chunk):
        stop = min(start + row_chunk, left.shape[0])
        actual = left[start:stop] @ right
        expected = expected_left[start:stop] @ vh[:rank]
        maximum = max(maximum, float(torch.max(torch.abs(actual - expected))))
        del actual, expected
    return maximum


def aggregate_cp_reconstruction_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("method") == "cp" and row.get("status") == "completed":
            grouped[int(row["rank"])].append(float(row["actual_relative_squared_frobenius_error"]))
    output = []
    for rank, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "method": "cp",
                "rank": rank,
                "completed_seed_count": len(values),
                "actual_relative_squared_frobenius_error_mean": float(array.mean()),
                "actual_relative_squared_frobenius_error_std_population": float(array.std(ddof=0)),
                "actual_relative_squared_frobenius_error_min": float(array.min()),
                "actual_relative_squared_frobenius_error_max": float(array.max()),
            }
        )
    return output


def reconstruction_rows_for_conv(
    conv,
    ranks,
    seeds,
    init,
    n_iter_max,
    device,
    tolerance=1e-5,
    memory_efficient_mttkrp=True,
    mttkrp_rank_chunk_size=64,
    mttkrp_max_explicit_bytes=512 * 1024**2,
    on_update=None,
):
    '''Compute deterministic matrix-SVD and multi-seed CP reconstruction rows.'''
    weight = conv.weight.detach().cpu()
    shape = tuple(weight.shape)
    dense_parameters = dense_conv_parameter_count(shape, conv.bias is not None)
    rows = []
    failures = []

    u, s, vh, svd_runtime, svd_device = output_mode_svd(weight, device=device)
    diagnostic = rank_energy_diagnostic(
        weight,
        modes=(0, 1, 2, 3),
        fixed_ranks=ranks,
        precomputed_singular_values={0: s.numpy()},
    )
    diagnostic["output_mode_matrix_svd_device"] = svd_device
    total_energy = float(torch.sum(s.to(torch.float64).square()))
    for rank in ranks:
        rank = min(int(rank), len(s))
        max_bound, output_bound, mode_bounds = diagnostic_bounds(diagnostic, rank)
        parameters = matrix_svd_conv_parameter_count(shape, rank, conv.bias is not None)
        try:
            module = MatrixLowRankConv2d.from_svd(conv.cpu(), rank, u, s, vh)
            kernel_max_abs_difference = verify_matrix_svd_kernel(module, u, s, vh, rank)
            if kernel_max_abs_difference > tolerance:
                raise AssertionError(
                    f"Composed matrix-SVD kernel differs from U_r S_r Vh_r by {kernel_max_abs_difference}"
                )
            actual_parameter_count = sum(parameter.numel() for parameter in module.parameters())
            if actual_parameter_count != parameters:
                raise AssertionError(f"Matrix-SVD parameter count {actual_parameter_count} != formula {parameters}")
            squared = float(torch.sum(s[rank:].to(torch.float64).square()) / total_energy)
            ordinary = math.sqrt(max(squared, 0.0))
            if abs(squared - output_bound) > tolerance:
                raise AssertionError(
                    f"SVD squared residual {squared} differs from output tail {output_bound}"
                )
            row = {
                "method": "matrix_svd_output_unfolding",
                "rank": rank,
                "seed": "",
                "status": "completed",
                "failure_exception": "",
                "dense_tensor_shape": "x".join(map(str, shape)),
                "parameter_count": parameters,
                "parameter_ratio": parameters / dense_parameters,
                "compression_factor": dense_parameters / parameters,
                "max_unfolding_tail_bound_squared": max_bound,
                "output_mode_tail_bound_squared": output_bound,
                "actual_relative_squared_frobenius_error": squared,
                "actual_relative_frobenius_error": ordinary,
                "gap_above_max_bound": "",
                "decomposition_runtime_seconds": svd_runtime,
                "decomposition_iterations": 1,
                "convergence_status": "exact_truncated_svd",
                "initializer": "deterministic_output_mode_svd",
                "n_iter_max": "",
                "bound_tolerance": tolerance,
                "bound_verification_passed": abs(squared - output_bound) <= tolerance,
                "matrix_kernel_max_abs_difference": kernel_max_abs_difference,
                "mode_tail_bounds_squared": mode_bounds,
            }
            rows.append(row)
            del module
        except Exception as exc:
            failure = {"method": "matrix_svd_output_unfolding", "rank": rank, "seed": "", "exception": repr(exc)}
            failures.append(failure)
            rows.append({"method": failure["method"], "rank": rank, "seed": "", "status": "failed", "failure_exception": repr(exc)})
        if on_update:
            on_update(rows)

    del u, s, vh
    gc.collect()

    for rank in ranks:
        max_bound, output_bound, mode_bounds = diagnostic_bounds(diagnostic, int(rank))
        parameters = cp_conv_parameter_count(shape, rank, conv.bias is not None)
        for seed in seeds:
            try:
                from src.smallnet.reproducibility import set_seed

                seed_status = set_seed(seed, deterministic=True)
                fitted, approximation, runtime, mttkrp_implementation = fit_cp_approximation(
                    conv,
                    rank,
                    seed,
                    init,
                    n_iter_max,
                    device,
                    memory_efficient_mttkrp,
                    mttkrp_rank_chunk_size,
                    mttkrp_max_explicit_bytes,
                )
                squared, ordinary = normalized_frobenius_residual(weight, approximation)
                actual_parameter_count = sum(parameter.numel() for parameter in fitted.parameters())
                if actual_parameter_count != parameters:
                    raise AssertionError(f"CP parameter count {actual_parameter_count} != formula {parameters}")
                gap = squared - max_bound
                passed = gap >= -tolerance
                if not passed:
                    raise AssertionError(
                        f"CP squared residual {squared} violates unfolding lower bound {max_bound}"
                    )
                row = {
                    "method": "cp",
                    "rank": int(rank),
                    "seed": int(seed),
                    "status": "completed",
                    "failure_exception": "",
                    "dense_tensor_shape": "x".join(map(str, shape)),
                    "parameter_count": parameters,
                    "parameter_ratio": parameters / dense_parameters,
                    "compression_factor": dense_parameters / parameters,
                    "max_unfolding_tail_bound_squared": max_bound,
                    "output_mode_tail_bound_squared": output_bound,
                    "actual_relative_squared_frobenius_error": squared,
                    "actual_relative_frobenius_error": ordinary,
                    "gap_above_max_bound": gap,
                    "decomposition_runtime_seconds": runtime,
                    "decomposition_iterations": "",
                    "convergence_status": "completed_requested_budget_iteration_history_unavailable",
                    "initializer": init,
                    "n_iter_max": int(n_iter_max),
                    "mttkrp_implementation": mttkrp_implementation,
                    "mttkrp_rank_chunk_size": int(mttkrp_rank_chunk_size),
                    "mttkrp_max_explicit_bytes": int(mttkrp_max_explicit_bytes),
                    "default_explicit_khatri_rao_peak_bytes_avoided": explicit_khatri_rao_peak_bytes(shape, rank),
                    "bound_tolerance": tolerance,
                    "bound_verification_passed": passed,
                    "matrix_kernel_max_abs_difference": "",
                    "mode_tail_bounds_squared": mode_bounds,
                    "seed_status": seed_status,
                }
                rows.append(row)
                del fitted, approximation
            except Exception as exc:
                failure = {"method": "cp", "rank": int(rank), "seed": int(seed), "exception": repr(exc)}
                failures.append(failure)
                rows.append({"method": "cp", "rank": int(rank), "seed": int(seed), "status": "failed", "failure_exception": repr(exc)})
            gc.collect()
            if on_update:
                on_update(rows)

    return rows, aggregate_cp_reconstruction_rows(rows), diagnostic, failures


def write_unfolding_energy_figure(diagnostic, ranks, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode_text, record in diagnostic["modes"].items():
        mode = int(mode_text)
        for index, energy in enumerate(record["cumulative_energy"], start=1):
            rows.append({"mode": mode, "mode_name": MODE_NAMES.get(mode, f"mode {mode}"), "rank": index, "cumulative_energy": energy})
    data_path = figures_dir / "figure_a_unfolding_cumulative_energy.csv"
    write_csv(data_path, rows)
    frame = pd.DataFrame(rows)
    styles = {0: ("-", 2.0), 1: ("--", 2.0), 2: ("-.", 1.5), 3: (":", 1.8)}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for mode, group in frame.groupby("mode"):
        linestyle, width = styles[int(mode)]
        ax.plot(group["rank"], group["cumulative_energy"], linestyle=linestyle, linewidth=width, label=group["mode_name"].iloc[0])
    for rank in ranks:
        ax.axvline(rank, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xlabel("Unfolding rank")
    ax.set_ylabel("Cumulative spectral energy $E_m(r)$")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Cumulative energy of convolution-tensor unfoldings")
    ax.legend()
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = figures_dir / f"figure_a_unfolding_cumulative_energy.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return [str(data_path), *outputs]


def write_reconstruction_figure(rows, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for row in rows if row.get("status") == "completed"])
    data_path = figures_dir / "figure_b_reconstruction_squared_error.csv"
    frame.to_csv(data_path, index=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bounds = frame.drop_duplicates("rank").sort_values("rank")
    ax.plot(bounds["rank"], bounds["max_unfolding_tail_bound_squared"], "k-o", label="Strongest unfolding lower bound $L(r)$")
    cp = frame[frame["method"] == "cp"]
    if not cp.empty:
        ax.scatter(cp["rank"], cp["actual_relative_squared_frobenius_error"], marker="o", facecolors="none", edgecolors="tab:blue", label="CP individual seeds")
        means = cp.groupby("rank", as_index=False)["actual_relative_squared_frobenius_error"].mean()
        ax.plot(means["rank"], means["actual_relative_squared_frobenius_error"], color="tab:blue", label="CP mean")
    matrix = frame[frame["method"] == "matrix_svd_output_unfolding"].sort_values("rank")
    if not matrix.empty:
        ax.plot(matrix["rank"], matrix["actual_relative_squared_frobenius_error"], "s--", color="tab:orange", label="Output-unfolding truncated SVD")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Normalized squared Frobenius error")
    ax.set_title("Unfolding lower bound and actual weight reconstruction error")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = figures_dir / f"figure_b_reconstruction_squared_error.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return [str(data_path), *outputs]


def write_zero_shot_figure(rows, figures_dir, split="test"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for row in rows if row.get("split") == split and row.get("status", "completed") == "completed"])
    data_path = figures_dir / "figure_c_zero_shot_present_class_miou.csv"
    frame.to_csv(data_path, index=False)
    if frame.empty:
        return [str(data_path)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    dense = frame[frame["method"] == "dense"]
    if not dense.empty:
        ax.axhline(float(dense["present_class_miou"].iloc[0]), color="k", linestyle=":", label="Dense baseline")
    cp = frame[frame["method"] == "cp"]
    if not cp.empty:
        ax.scatter(cp["rank"], cp["present_class_miou"], marker="o", facecolors="none", edgecolors="tab:blue", label="CP individual seeds")
        means = cp.groupby("rank", as_index=False)["present_class_miou"].mean()
        ax.plot(means["rank"], means["present_class_miou"], color="tab:blue", label="CP mean")
    matrix = frame[frame["method"] == "matrix_svd_output_unfolding"].sort_values("rank")
    if not matrix.empty:
        ax.plot(matrix["rank"], matrix["present_class_miou"], "s--", color="tab:orange", label="Output-unfolding truncated SVD")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Present-class mIoU")
    ax.set_title(f"Zero-shot present-class mIoU on the CamVid {split} split")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        path = figures_dir / f"figure_c_zero_shot_present_class_miou.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return [str(data_path), *outputs]


def join_structural_tradeoffs(reconstruction_rows, evaluation_rows):
    '''Join one reconstruction row to its validation and test evaluation rows.'''
    reconstruction = {
        (str(row.get("method")), str(row.get("rank", "")), str(row.get("seed", ""))): row
        for row in reconstruction_rows
        if row.get("status", "completed") == "completed"
    }
    grouped = defaultdict(dict)
    for row in evaluation_rows:
        if row.get("status", "completed") != "completed":
            continue
        key = (str(row.get("method")), str(row.get("rank", "")), str(row.get("seed", "")))
        grouped[key][str(row["split"])] = row

    output = []
    for key, splits in sorted(grouped.items()):
        method, rank, seed = key
        any_row = next(iter(splits.values()))
        reconstruction_row = reconstruction.get(key, {})
        dense_target = float(any_row.get("dense_target_layer_parameter_count", 0) or 0)
        target = float(any_row.get("target_layer_parameter_count", 0) or 0)
        output.append(
            {
                "method": method,
                "rank": rank,
                "seed": seed,
                "max_unfolding_tail_bound_squared": reconstruction_row.get(
                    "max_unfolding_tail_bound_squared",
                    any_row.get("max_unfolding_tail_bound_squared", 0.0 if method == "dense" else ""),
                ),
                "actual_relative_squared_frobenius_error": reconstruction_row.get(
                    "actual_relative_squared_frobenius_error",
                    any_row.get("actual_relative_squared_frobenius_error", 0.0 if method == "dense" else ""),
                ),
                "actual_relative_frobenius_error": reconstruction_row.get(
                    "actual_relative_frobenius_error",
                    any_row.get("actual_relative_frobenius_error", 0.0 if method == "dense" else ""),
                ),
                "zero_shot_validation_present_class_miou": splits.get("val", {}).get("present_class_miou", ""),
                "zero_shot_test_present_class_miou": splits.get("test", {}).get("present_class_miou", ""),
                "target_layer_parameter_ratio": target / dense_target if dense_target else "",
                "target_layer_parameter_count": int(target) if target else "",
                "full_model_parameter_count": any_row.get("full_model_parameter_count", ""),
                "target_layer_macs": any_row.get("target_layer_macs", ""),
                "full_model_macs": any_row.get("full_model_macs", ""),
            }
        )
    return output


def exploratory_correlations(tradeoff_rows):
    '''Return descriptive correlations only; no p-values or significance tests.'''
    frame = pd.DataFrame(tradeoff_rows)
    if frame.empty:
        return {"label": "exploratory_descriptive_only", "correlations": []}
    numeric = [
        "max_unfolding_tail_bound_squared",
        "actual_relative_squared_frobenius_error",
        "zero_shot_validation_present_class_miou",
        "zero_shot_test_present_class_miou",
        "target_layer_parameter_ratio",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    records = []
    for method, group in [("all_nondense", frame[frame["method"] != "dense"]), *frame.groupby("method")]:
        if method == "dense" or len(group) < 2:
            continue
        for x in ("max_unfolding_tail_bound_squared", "actual_relative_squared_frobenius_error", "target_layer_parameter_ratio"):
            for y in ("zero_shot_validation_present_class_miou", "zero_shot_test_present_class_miou"):
                pair = group[[x, y]].dropna()
                if len(pair) < 2:
                    continue
                records.append(
                    {
                        "method_scope": method,
                        "x": x,
                        "y": y,
                        "observation_count": len(pair),
                        "pearson_correlation": float(pair[x].corr(pair[y], method="pearson")),
                        "spearman_correlation": float(pair[x].corr(pair[y], method="spearman")),
                    }
                )
    return {
        "label": "exploratory_descriptive_only",
        "significance_tests_performed": False,
        "causal_or_predictive_claim_supported": False,
        "correlations": records,
    }
