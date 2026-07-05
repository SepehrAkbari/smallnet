'''
Paper asset generation from saved experiment manifests.
'''

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/smallnet-cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/smallnet-matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from src.smallnet.results import load_manifest, write_csv


def rank_diagnostic_rows(rank_manifest):
    rows = []
    for layer_name, diagnostic in rank_manifest.get("diagnostics", {}).items():
        base = {
            "layer": layer_name,
            "params": diagnostic.get("params"),
            "shape": diagnostic.get("shape"),
        }
        for threshold, rank in diagnostic.get("rank_energy_thresholds", {}).items():
            base[f"r_tau_{threshold}"] = rank
        for rank, tail in diagnostic.get("cp_necessary_tail_energy", {}).items():
            base[f"max_tail_r{rank}"] = tail
        rows.append(base)
    return rows


def pareto_rows(eval_manifest, profile_manifest=None):
    profile_by_label = {}
    if profile_manifest:
        for row in profile_manifest.get("profiles", []):
            profile_by_label[row["label"]] = row

    rows = []
    for result in eval_manifest.get("evaluations", []):
        row = {
            "label": result["label"],
            "split": result["split"],
            "rank": result.get("rank", ""),
            "ignore_class": result.get("ignore_class", ""),
            "mIoU": result["summary"].get("mean_iou_all_classes"),
            "present_mIoU": result["summary"].get("mean_iou_present_classes"),
            "FWIoU": result["summary"].get("frequency_weighted_iou"),
            "pixel_accuracy": result["summary"].get("pixel_accuracy"),
        }
        profile = profile_by_label.get(result["label"], {})
        row["parameters"] = profile.get("parameters", result.get("parameters", ""))
        row["macs"] = profile.get("macs", "")
        row["latency_ms"] = profile.get("latency_ms", "")
        rows.append(row)
    return rows


def write_rank_table(rank_manifest_path, out_csv):
    manifest = load_manifest(rank_manifest_path)
    rows = rank_diagnostic_rows(manifest)
    return write_csv(out_csv, rows)


def plot_rank_spectrum(rank_manifest_path, out_png, layer=None, mode="0"):
    manifest = load_manifest(rank_manifest_path)
    diagnostics = manifest.get("diagnostics", {})
    if not diagnostics:
        raise ValueError("Rank manifest does not contain diagnostics")
    layer = layer or next(iter(diagnostics))
    diagnostic = diagnostics[layer]
    mode_record = diagnostic["modes"][str(mode)]
    singular_values = mode_record["singular_values"]
    cumulative = mode_record["cumulative_energy"]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(singular_values, color="black", linewidth=1.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Magnitude (log)")
    ax1.grid(True, which="both", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(cumulative, color="tab:blue", linewidth=1.4, alpha=0.75)
    ax2.set_ylim(0, 1.01)
    ax2.set_ylabel("Cumulative energy")

    for threshold, rank in mode_record.get("threshold_ranks", {}).items():
        ax1.axvline(rank, linestyle="--", alpha=0.5, label=f"{float(threshold):.0%}: r={rank}")

    ax1.set_title(f"Rank-energy spectrum: {layer}, mode {mode}")
    ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def write_pareto_table(eval_manifest_path, profile_manifest_path, out_csv):
    eval_manifest = load_manifest(eval_manifest_path)
    profile_manifest = load_manifest(profile_manifest_path) if profile_manifest_path else None
    rows = pareto_rows(eval_manifest, profile_manifest=profile_manifest)
    fieldnames = [
        "label",
        "split",
        "rank",
        "ignore_class",
        "mIoU",
        "present_mIoU",
        "FWIoU",
        "pixel_accuracy",
        "parameters",
        "macs",
        "latency_ms",
    ]
    return write_csv(out_csv, rows, fieldnames=fieldnames)


def plot_pareto(eval_manifest_path, profile_manifest_path, out_png, metric="mIoU"):
    eval_manifest = load_manifest(eval_manifest_path)
    profile_manifest = load_manifest(profile_manifest_path) if profile_manifest_path else None
    rows = pareto_rows(eval_manifest, profile_manifest=profile_manifest)
    rows = [row for row in rows if row.get("parameters") not in ("", None) and row.get(metric) is not None]
    if not rows:
        raise ValueError("No rows with parameters and metric available for Pareto plot")

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        x = float(row["parameters"]) / 1e6
        y = float(row[metric])
        ax.scatter(x, y, s=60)
        ax.annotate(row["label"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png
