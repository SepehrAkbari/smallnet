'''
Paper table and figure generation helpers.
'''

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.smallnet.config import ensure_dir
from src.smallnet.results import environment_metadata


def resolve_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def paper_paths(config, root):
    paper_cfg = config.get("paper", {})
    tables_dir = ensure_dir(resolve_path(paper_cfg.get("tables_dir", "results/paper/tables"), root))
    figures_dir = ensure_dir(resolve_path(paper_cfg.get("figures_dir", "results/paper/figures"), root))
    manifest_path = resolve_path(paper_cfg.get("manifest_path", "results/paper/MANIFEST.json"), root)
    ensure_dir(manifest_path.parent)
    return tables_dir, figures_dir, manifest_path


def write_latex_table(df, path, caption=None, label=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(
            df.to_latex(
                index=False,
                float_format=lambda value: f"{value:.4f}",
                caption=caption,
                label=label,
                escape=True,
            )
        )
    return path


def rank_energy_rows(diagnostics, ranks):
    summary_rows = []
    singular_rows = []
    for layer, diagnostic in diagnostics.items():
        for mode, mode_record in diagnostic["modes"].items():
            cumulative = mode_record["cumulative_energy"]
            values = mode_record["singular_values"]
            matrix_shape = "x".join(str(value) for value in mode_record["matrix_shape"])
            for rank in ranks:
                if rank <= 0:
                    energy = 0.0
                elif rank >= len(cumulative):
                    energy = 1.0 if len(cumulative) else 0.0
                else:
                    energy = float(cumulative[rank - 1])
                summary_rows.append(
                    {
                        "layer": layer,
                        "mode": int(mode),
                        "matrix_shape": matrix_shape,
                        "rank": int(rank),
                        "energy": energy,
                        "tail_energy": 1.0 - energy,
                    }
                )
            for idx, value in enumerate(values, start=1):
                singular_rows.append(
                    {
                        "layer": layer,
                        "mode": int(mode),
                        "index": idx,
                        "singular_value": float(value),
                        "cumulative_energy": float(cumulative[idx - 1]),
                    }
                )
    return summary_rows, singular_rows


def write_rank_energy_artifacts(diagnostics, config, root):
    tables_dir, figures_dir, _ = paper_paths(config, root)
    ranks = config.get("rank_diagnostics", {}).get(
        "fixed_ranks",
        config.get("cp", {}).get("zero_shot_ranks", config.get("cp", {}).get("ranks", [32, 64, 128, 256, 512])),
    )
    summary_rows, singular_rows = rank_energy_rows(diagnostics, ranks)
    summary = pd.DataFrame(summary_rows)
    singular = pd.DataFrame(singular_rows)

    summary_path = tables_dir / "rank_energy_summary.csv"
    singular_path = tables_dir / "singular_values_by_mode.csv"
    summary.to_csv(summary_path, index=False)
    singular.to_csv(singular_path, index=False)
    write_latex_table(
        summary,
        tables_dir / "rank_energy_summary.tex",
        caption="Unfolding rank-energy and tail-energy diagnostics for the target convolution.",
        label="tab:rank-energy",
    )

    if not singular.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, group in singular.groupby("mode"):
            ax.plot(group["index"], group["singular_value"], label=f"mode {mode}")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value")
        ax.set_title("Singular value decay by unfolding mode")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "singular_value_decay_by_mode.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, group in singular.groupby("mode"):
            ax.plot(group["index"], group["cumulative_energy"], label=f"mode {mode}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Cumulative energy")
        ax.set_ylim(0.0, 1.01)
        ax.set_title("Cumulative energy by unfolding mode")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "cumulative_energy_by_mode.pdf")
        plt.close(fig)

    return {
        "rank_energy_summary_csv": str(summary_path),
        "rank_energy_summary_tex": str(tables_dir / "rank_energy_summary.tex"),
        "singular_values_by_mode_csv": str(singular_path),
    }


def read_optional_csv(path):
    path = Path(path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def sort_results(df):
    if df.empty:
        return df
    out = df.copy()
    if "model_kind" not in out.columns:
        out["model_kind"] = out.get("source", "")
    if "split" not in out.columns:
        out["split"] = ""
    if "rank" not in out.columns:
        out["rank"] = ""
    if "label" not in out.columns:
        out["label"] = ""
    out["_rank_numeric"] = pd.to_numeric(out["rank"], errors="coerce")
    out["_split_order"] = out["split"].map({"train": 0, "val": 1, "test": 2}).fillna(9)
    out = out.sort_values(["_split_order", "model_kind", "_rank_numeric", "label"], na_position="first")
    return out.drop(columns=["_rank_numeric", "_split_order"])


def collect_main_results(output_dir):
    inputs = {
        "dense": output_dir / "dense_eval_summary.csv",
        "zero_shot": output_dir / "cp_zero_shot_summary.csv",
        "existing_finetuned": output_dir / "existing_finetuned_summary.csv",
        "new_finetuned": output_dir / "cp_finetune_summary.csv",
    }
    frames = []
    missing = []
    for source, path in inputs.items():
        frame = read_optional_csv(path)
        if frame is None:
            missing.append(str(path))
            continue
        frame["source"] = source
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), missing
    return sort_results(pd.concat(frames, ignore_index=True)), missing


def collect_profile_results(output_dir):
    path = output_dir / "profile_summary.csv"
    frame = read_optional_csv(path)
    if frame is None:
        return pd.DataFrame(), [str(path)]
    return frame, []


def plot_metric_vs_ratio(main, figures_dir, ratio_col, filename, ylabel):
    if main.empty or ratio_col not in main.columns:
        return None
    metric = "mean_iou_all_classes"
    usable = main[main[metric].notna() & main[ratio_col].notna()].copy()
    if usable.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, marker in [("val", "o"), ("test", "s")]:
        group = usable[usable["split"] == split]
        if not group.empty:
            ax.scatter(group[ratio_col], group[metric], marker=marker, label=split)
            for _, row in group.iterrows():
                ax.annotate(str(row["label"]).replace("cp_rank_", "r"), (row[ratio_col], row[metric]), fontsize=7)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("mIoU over non-excluded classes")
    ax.set_title(f"mIoU vs {ylabel}")
    ax.legend()
    fig.tight_layout()
    out = figures_dir / filename
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_profile(profile, figures_dir, x_col, filename, xlabel):
    if profile.empty or x_col not in profile.columns or "latency_mean_ms" not in profile.columns:
        return None
    usable = profile[profile[x_col].notna() & profile["latency_mean_ms"].notna()].copy()
    if usable.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(usable[x_col], usable["latency_mean_ms"])
    for _, row in usable.iterrows():
        ax.annotate(str(row["label"]).replace("cp_rank_", "r"), (row[x_col], row["latency_mean_ms"]), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(f"Latency vs {xlabel}")
    fig.tight_layout()
    out = figures_dir / filename
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def build_paper_artifacts(config, root):
    tables_dir, figures_dir, manifest_path = paper_paths(config, root)
    output_dir = resolve_path(config.get("output_dir", "results/camvid_vgg_cp"), root)
    main, missing_main = collect_main_results(output_dir)
    profile, missing_profile = collect_profile_results(output_dir)

    outputs = {}
    main_path = tables_dir / "main_results.csv"
    main.to_csv(main_path, index=False)
    outputs["main_results_csv"] = str(main_path)
    outputs["main_results_tex"] = str(
        write_latex_table(
            main,
            tables_dir / "main_results.tex",
            caption="CamVid segmentation accuracy and parameter ratios.",
            label="tab:main-results",
        )
    )

    profile_path = tables_dir / "profile_results.csv"
    profile.to_csv(profile_path, index=False)
    outputs["profile_results_csv"] = str(profile_path)
    outputs["profile_results_tex"] = str(
        write_latex_table(
            profile,
            tables_dir / "profile_results.tex",
            caption="Parameter, MAC, and latency profiling results.",
            label="tab:profile-results",
        )
    )

    for key, path in [
        (
            "miou_vs_target_layer_parameter_ratio_pdf",
            plot_metric_vs_ratio(
                main,
                figures_dir,
                "target_layer_compression_ratio",
                "miou_vs_target_layer_parameter_ratio.pdf",
                "Target-layer parameter ratio",
            ),
        ),
        (
            "miou_vs_total_parameter_ratio_pdf",
            plot_metric_vs_ratio(
                main,
                figures_dir,
                "total_compression_ratio",
                "miou_vs_total_parameter_ratio.pdf",
                "Total parameter ratio",
            ),
        ),
        (
            "latency_vs_macs_pdf",
            plot_profile(profile, figures_dir, "macs", "latency_vs_macs.pdf", "MACs"),
        ),
        (
            "latency_vs_total_parameter_ratio_pdf",
            plot_profile(
                profile,
                figures_dir,
                "total_compression_ratio",
                "latency_vs_total_parameter_ratio.pdf",
                "Total parameter ratio",
            ),
        ),
    ]:
        if path:
            outputs[key] = path

    for key, path in {
        "rank_energy_summary_csv": tables_dir / "rank_energy_summary.csv",
        "rank_energy_summary_tex": tables_dir / "rank_energy_summary.tex",
        "singular_values_by_mode_csv": tables_dir / "singular_values_by_mode.csv",
        "singular_value_decay_by_mode_pdf": figures_dir / "singular_value_decay_by_mode.pdf",
        "cumulative_energy_by_mode_pdf": figures_dir / "cumulative_energy_by_mode.pdf",
    }.items():
        if path.is_file():
            outputs[key] = str(path)

    manifest = {
        "schema": "smallnet.paper_manifest.v1",
        "environment": environment_metadata(),
        "input_directory": str(output_dir),
        "missing_inputs": missing_main + missing_profile,
        "outputs": outputs,
        "interpretation_note": (
            "Unfolding rank-energy is a necessary structural diagnostic for CP compression. "
            "It does not prove that a CP-rank model will preserve downstream segmentation accuracy."
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
