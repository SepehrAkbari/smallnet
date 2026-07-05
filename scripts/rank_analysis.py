'''
Rank-energy analysis for dense convolutional tensors.
'''

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/smallnet-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/smallnet-matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch


def unfold_tensor(weight, mode):
    weight = weight.detach().cpu()
    order = list(range(weight.ndim))
    order[0], order[mode] = order[mode], order[0]
    unfolded = weight.permute(order).reshape(weight.shape[mode], -1)
    return unfolded


def spectrum_for_mode(weight, mode):
    matrix = unfold_tensor(weight, mode)
    singular_values = torch.linalg.svdvals(matrix).numpy()
    energy = singular_values**2
    total_energy = energy.sum()
    cumulative = energy.cumsum() / total_energy if total_energy > 0 else energy
    return singular_values, cumulative


def rank_for_threshold(cumulative_energy, threshold):
    cumulative_energy = np.asarray(cumulative_energy)
    hits = (cumulative_energy >= threshold).nonzero()[0]
    return int(hits[0] + 1) if len(hits) else int(len(cumulative_energy))


def tail_energy_at_rank(cumulative_energy, rank):
    cumulative_energy = np.asarray(cumulative_energy)
    if rank <= 0:
        return 1.0
    if rank >= len(cumulative_energy):
        return 0.0
    return float(1.0 - cumulative_energy[rank - 1])


def analyze_weight(name, weight, modes, thresholds, fixed_ranks):
    records = []
    spectra = {}
    for mode in modes:
        singular_values, cumulative = spectrum_for_mode(weight, mode)
        spectra[mode] = {
            "singular_values": singular_values.tolist(),
            "cumulative_energy": cumulative.tolist(),
        }
        record = {
            "layer": name,
            "shape": list(weight.shape),
            "mode": mode,
            "params": int(weight.numel()),
            "matrix_shape": list(unfold_tensor(weight, mode).shape),
        }
        for threshold in thresholds:
            record[f"rank_at_{threshold:.3f}"] = rank_for_threshold(cumulative, threshold)
        for rank in fixed_ranks:
            record[f"tail_energy_rank_{rank}"] = tail_energy_at_rank(cumulative, rank)
        records.append(record)
    return records, spectra


def load_dense_conv_weights(checkpoint_path, requested_layers=None, all_conv=False):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    weights = {}
    for name, tensor in state_dict.items():
        if not name.endswith(".weight") or tensor.ndim != 4:
            continue
        if requested_layers and name not in requested_layers:
            continue
        if all_conv or requested_layers:
            weights[name] = tensor
    if not weights and not all_conv:
        default = "classifier.0.weight"
        if default not in state_dict:
            raise ValueError(f"Default layer {default} not found in {checkpoint_path}")
        weights[default] = state_dict[default]
    return weights


def write_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_spectrum(layer_name, mode, spectra, thresholds, out_path):
    data = spectra[mode]
    singular_values = data["singular_values"]
    cumulative = data["cumulative_energy"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(singular_values, color="black", linewidth=2, label="Singular values")
    ax1.set_yscale("log")
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Magnitude (log scale)")

    ax2 = ax1.twinx()
    ax2.plot(cumulative, color="tab:blue", alpha=0.7, label="Cumulative energy")
    ax2.set_ylabel("Cumulative energy")
    ax2.set_ylim(0, 1.01)

    for threshold in thresholds:
        rank = rank_for_threshold(cumulative, threshold)
        ax1.axvline(rank, linestyle="--", alpha=0.5, label=f"{threshold:.0%} energy rank={rank}")

    ax1.set_title(f"Rank-energy spectrum: {layer_name}, mode {mode}")
    ax1.grid(True, which="both", alpha=0.2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model/best_model.pth")
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--all-conv", action="store_true")
    parser.add_argument("--modes", type=int, nargs="*", default=[0])
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.9, 0.95, 0.99])
    parser.add_argument("--fixed-ranks", type=int, nargs="*", default=[64, 128, 256])
    parser.add_argument("--out-csv", default="res/rank_analysis.csv")
    parser.add_argument("--out-json", default="res/rank_analysis.json")
    parser.add_argument("--plot", default="res/rank_energy_classifier0_mode0.png")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = load_dense_conv_weights(args.checkpoint, requested_layers=args.layers, all_conv=args.all_conv)
    records = []
    all_spectra = {}

    for name, weight in weights.items():
        modes = [mode for mode in args.modes if 0 <= mode < weight.ndim]
        layer_records, spectra = analyze_weight(name, weight, modes, args.thresholds, args.fixed_ranks)
        records.extend(layer_records)
        all_spectra[name] = spectra

    write_csv(records, args.out_csv)
    write_json({"checkpoint": args.checkpoint, "records": records, "spectra": all_spectra}, args.out_json)

    first_layer = next(iter(all_spectra))
    first_mode = args.modes[0]
    if args.plot and first_mode in all_spectra[first_layer]:
        plot_spectrum(first_layer, first_mode, all_spectra[first_layer], args.thresholds, args.plot)

    print(f"Analyzed {len(records)} layer/mode combinations")
    print(f"Wrote: {args.out_csv}, {args.out_json}")
    if args.plot:
        print(f"Wrote: {args.plot}")


if __name__ == "__main__":
    main()
