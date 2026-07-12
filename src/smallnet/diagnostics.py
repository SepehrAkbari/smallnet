'''
Rank-energy diagnostics for convolutional tensors.
'''

import numpy as np
import torch


def unfold_tensor(weight, mode):
    if mode < 0 or mode >= weight.ndim:
        raise ValueError(f"Mode {mode} is invalid for tensor order {weight.ndim}")
    weight = weight.detach().cpu()
    order = list(range(weight.ndim))
    order[0], order[mode] = order[mode], order[0]
    return weight.permute(order).reshape(weight.shape[mode], -1)


def singular_values(weight, mode):
    matrix = unfold_tensor(weight, mode)
    return torch.linalg.svdvals(matrix).cpu().numpy()


def cumulative_energy_from_singular_values(values):
    values = np.asarray(values, dtype=np.float64)
    energy = values**2
    total = energy.sum()
    if total <= 0:
        return np.zeros_like(energy)
    return np.cumsum(energy) / total


def spectrum_for_mode(weight, mode):
    values = singular_values(weight, mode)
    return values, cumulative_energy_from_singular_values(values)


def rank_for_energy(cumulative_energy, threshold):
    cumulative_energy = np.asarray(cumulative_energy, dtype=np.float64)
    hits = np.nonzero(cumulative_energy >= threshold)[0]
    if len(hits) == 0:
        return int(len(cumulative_energy))
    return int(hits[0] + 1)


def tail_energy(cumulative_energy, rank):
    cumulative_energy = np.asarray(cumulative_energy, dtype=np.float64)
    if rank <= 0:
        return 1.0
    if rank >= len(cumulative_energy):
        return 0.0
    return float(1.0 - cumulative_energy[rank - 1])


def best_matrix_tail_energy(weight, mode, rank):
    _, cumulative = spectrum_for_mode(weight, mode)
    return tail_energy(cumulative, rank)


def cp_lower_bound_tail(weight, rank, modes=(0, 1, 2, 3)):
    tails = {}
    for mode in modes:
        if mode < weight.ndim:
            tails[str(mode)] = best_matrix_tail_energy(weight, mode, rank)
    return {
        "rank": int(rank),
        "mode_tail_energy": tails,
        "max_tail_energy": max(tails.values()) if tails else 0.0,
    }


def rank_energy_diagnostic(
    weight,
    modes=(0, 1, 2, 3),
    thresholds=(0.9, 0.95, 0.99),
    fixed_ranks=(64, 128, 256),
    precomputed_singular_values=None,
):
    per_mode = {}
    threshold_ranks = {f"{threshold:.3f}": [] for threshold in thresholds}
    fixed_rank_tails = {str(rank): [] for rank in fixed_ranks}

    for mode in modes:
        if mode < 0 or mode >= weight.ndim:
            continue
        precomputed = (precomputed_singular_values or {}).get(mode)
        if precomputed is None:
            values, cumulative = spectrum_for_mode(weight, mode)
        else:
            values = np.asarray(precomputed, dtype=np.float64)
            cumulative = cumulative_energy_from_singular_values(values)
        mode_record = {
            "matrix_shape": list(unfold_tensor(weight, mode).shape),
            "singular_values": values.tolist(),
            "cumulative_energy": cumulative.tolist(),
            "threshold_ranks": {},
            "tail_energy_at_fixed_ranks": {},
        }
        for threshold in thresholds:
            selected_rank = rank_for_energy(cumulative, threshold)
            key = f"{threshold:.3f}"
            mode_record["threshold_ranks"][key] = selected_rank
            threshold_ranks[key].append(selected_rank)
        for rank in fixed_ranks:
            tail = tail_energy(cumulative, rank)
            key = str(rank)
            mode_record["tail_energy_at_fixed_ranks"][key] = tail
            fixed_rank_tails[key].append(tail)
        per_mode[str(mode)] = mode_record

    return {
        "shape": list(weight.shape),
        "params": int(weight.numel()),
        "modes": per_mode,
        "rank_energy_thresholds": {
            key: max(values) if values else 0 for key, values in threshold_ranks.items()
        },
        "cp_necessary_tail_energy": {
            key: max(values) if values else 0.0 for key, values in fixed_rank_tails.items()
        },
    }


def cp_tensor_from_factors(factors, weights=None):
    if weights is None:
        weights = torch.ones(factors[0].shape[1], dtype=factors[0].dtype)
    tensor = None
    rank = factors[0].shape[1]
    for idx in range(rank):
        component = weights[idx]
        for mode, factor in enumerate(factors):
            shape = [1] * len(factors)
            shape[mode] = factor.shape[0]
            component = component * factor[:, idx].reshape(shape)
        tensor = component if tensor is None else tensor + component
    return tensor


def matrix_rank_by_mode(tensor, mode, tol=1e-6):
    values = singular_values(tensor, mode)
    return int((values > tol).sum())
