'''
Run the SPL-ready CamVid/VGG controlled bottleneck experiment.
'''

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.smallnet.config import ensure_dir, load_config
from src.smallnet.data import (
    load_camvid_class_names,
    make_camvid_loader,
    pairing_rule_from_config,
    strict_unknown_colors_from_config,
)
from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.evaluation import evaluate_segmentation
from src.smallnet.models import load_vgg16_fcn32s_checkpoint
from src.smallnet.modules import count_parameters
from src.smallnet.profiling import latency_ms, manual_macs
from src.smallnet.results import save_manifest, write_csv
from src.utils import summarize_hist


def auto_device(config_device=None):
    if config_device:
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_evaluations(config, device, max_batches=None, num_workers_override=None):
    output_dir = ensure_dir(config["output_dir"])
    data_cfg = config["dataset"]
    eval_cfg = config["eval"]
    class_names = load_camvid_class_names(Path(data_cfg["root"]) / "class_dict.csv")
    evaluations = []

    for checkpoint in config["checkpoints"]:
        print(f"[camvid:evaluate] loading {checkpoint['label']} from {checkpoint['path']}", flush=True)
        model, rank = load_vgg16_fcn32s_checkpoint(
            checkpoint["path"],
            num_classes=config["model"].get("num_classes", len(class_names)),
        )
        params = count_parameters(model)
        for split in eval_cfg["splits"]:
            print(f"[camvid:evaluate] {checkpoint['label']} split={split}", flush=True)
            loader = make_camvid_loader(
                data_cfg["root"],
                split,
                batch_size=eval_cfg.get("batch_size", 4),
                num_workers=eval_cfg.get("num_workers", 2) if num_workers_override is None else num_workers_override,
                image_size=tuple(eval_cfg.get("image_size", [352, 480])),
                pairing_rule=pairing_rule_from_config(data_cfg),
                strict_unknown_colors=strict_unknown_colors_from_config(data_cfg),
                unknown_color_ignore_index=data_cfg.get("unknown_color_ignore_index"),
                num_classes=data_cfg.get("num_classes"),
                ignore_index=data_cfg.get("ignore_index"),
                ignore_class_name=data_cfg.get("ignore_class_name"),
                allow_non_contiguous_class_indices=bool(data_cfg.get("allow_non_contiguous_class_indices", False)),
                required_class_at_index=data_cfg.get("required_class_at_index"),
            )
            hist_all, _ = evaluate_segmentation(
                model,
                loader,
                len(class_names),
                device,
                ignore_index=None,
                class_names=class_names,
                max_batches=max_batches,
            )
            for policy in eval_cfg["void_policies"]:
                print(
                    f"[camvid:evaluate] {checkpoint['label']} split={split} policy={policy['name']} summary",
                    flush=True,
                )
                ignore_class = policy.get("ignore_class")
                ignore_index = class_names.index(ignore_class) if ignore_class else None
                hist = hist_all.copy()
                if ignore_index is not None:
                    hist[ignore_index, :] = 0
                summary = summarize_hist(
                    hist,
                    class_names=class_names,
                    exclude_indices=[ignore_index] if ignore_index is not None else None,
                )
                summary.update(
                    {
                        "label": checkpoint["label"],
                        "checkpoint": checkpoint["path"],
                        "rank": "dense" if rank is None else rank,
                        "parameters": params,
                        "split": split,
                        "ignore_class": ignore_class or "",
                        "policy": policy["name"],
                    }
                )
                evaluations.append(
                    {
                        "label": checkpoint["label"],
                        "checkpoint": checkpoint["path"],
                        "rank": "dense" if rank is None else rank,
                        "parameters": params,
                        "split": split,
                        "ignore_class": ignore_class or "",
                        "policy": policy["name"],
                        "summary": summary,
                        "confusion_matrix": hist.astype(int).tolist(),
                    }
                )

    rows = []
    for item in evaluations:
        row = {
            "label": item["label"],
            "checkpoint": item["checkpoint"],
            "rank": item["rank"],
            "parameters": item["parameters"],
            "split": item["split"],
            "policy": item["policy"],
            "ignore_class": item["ignore_class"],
            "pixel_accuracy": item["summary"]["pixel_accuracy"],
            "mean_iou_all_classes": item["summary"]["mean_iou_all_classes"],
            "mean_iou_present_classes": item["summary"]["mean_iou_present_classes"],
            "frequency_weighted_iou": item["summary"]["frequency_weighted_iou"],
        }
        rows.append(row)

    write_csv(output_dir / "camvid_eval_summary.csv", rows)
    return save_manifest(
        output_dir / "camvid_eval_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "kind": "camvid_evaluation",
            "config_path": config["_config_path"],
            "device": device.type,
            "max_batches": max_batches,
            "num_workers_override": num_workers_override,
            "evaluations": evaluations,
        },
    )


def run_profiles(config, device, latency_override=None):
    output_dir = ensure_dir(config["output_dir"])
    profile_cfg = config["profile"]
    input_size = tuple(profile_cfg.get("input_size", config["model"]["input_size"]))
    profiles = []

    for checkpoint in config["checkpoints"]:
        print(f"[camvid:profile] loading {checkpoint['label']} from {checkpoint['path']}", flush=True)
        model, rank = load_vgg16_fcn32s_checkpoint(
            checkpoint["path"],
            num_classes=config["model"].get("num_classes", 32),
        )
        model = model.to(device)
        macs, layer_records = manual_macs(model, input_size, device=device)
        latency = None
        run_latency = profile_cfg.get("latency", False) if latency_override is None else latency_override
        if run_latency:
            print(f"[camvid:profile] latency {checkpoint['label']}", flush=True)
            latency = latency_ms(
                model,
                input_size,
                device,
                warmup=profile_cfg.get("warmup", 20),
                iterations=profile_cfg.get("iterations", 100),
            )
        profiles.append(
            {
                "label": checkpoint["label"],
                "checkpoint": checkpoint["path"],
                "rank": "dense" if rank is None else rank,
                "parameters": count_parameters(model),
                "macs": macs,
                "latency_ms": latency,
                "input_size": list(input_size),
                "layer_records": layer_records,
            }
        )

    write_csv(
        output_dir / "camvid_profile_summary.csv",
        [
            {
                "label": row["label"],
                "rank": row["rank"],
                "parameters": row["parameters"],
                "macs": row["macs"],
                "latency_ms": row["latency_ms"],
            }
            for row in profiles
        ],
    )
    return save_manifest(
        output_dir / "camvid_profile_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "kind": "camvid_profile",
            "config_path": config["_config_path"],
            "device": device.type,
            "profiles": profiles,
        },
    )


def run_rank_analysis(config):
    output_dir = ensure_dir(config["output_dir"])
    rank_cfg = config["rank_analysis"]
    checkpoint = rank_cfg.get("checkpoint", config["checkpoints"][0]["path"])
    state_dict = torch.load(checkpoint, map_location="cpu")
    diagnostics = {}
    rows = []

    for layer in rank_cfg["layers"]:
        print(f"[camvid:rank] analyzing {layer}", flush=True)
        weight = state_dict[layer]
        diagnostic = rank_energy_diagnostic(
            weight,
            modes=rank_cfg.get("modes", [0, 1, 2, 3]),
            thresholds=rank_cfg.get("thresholds", [0.9, 0.95, 0.99]),
            fixed_ranks=rank_cfg.get("fixed_ranks", [64, 128, 256]),
        )
        diagnostics[layer] = diagnostic
        row = {"layer": layer, "params": diagnostic["params"], "shape": diagnostic["shape"]}
        for threshold, selected_rank in diagnostic["rank_energy_thresholds"].items():
            row[f"r_tau_{threshold}"] = selected_rank
        for rank, tail in diagnostic["cp_necessary_tail_energy"].items():
            row[f"max_tail_r{rank}"] = tail
        rows.append(row)

    write_csv(output_dir / "camvid_rank_diagnostics.csv", rows)
    return save_manifest(
        output_dir / "camvid_rank_manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "kind": "camvid_rank_analysis",
            "config_path": config["_config_path"],
            "checkpoint": checkpoint,
            "diagnostics": diagnostics,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spl/camvid_vgg.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--labels", nargs="+", default=None, help="Optional checkpoint labels to run.")
    parser.add_argument("--output-dir", default=None, help="Override config output directory.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional CPU smoke-test limit.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override eval DataLoader workers.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--no-latency", action="store_true", help="Override config and skip latency timing.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.labels:
        labels = set(args.labels)
        config["checkpoints"] = [item for item in config["checkpoints"] if item["label"] in labels]
        if not config["checkpoints"]:
            raise ValueError(f"No checkpoints matched labels: {sorted(labels)}")
    device = auto_device(args.device or config.get("device"))
    outputs = []
    if not args.skip_eval:
        outputs.append(
            run_evaluations(
                config,
                device,
                max_batches=args.max_batches,
                num_workers_override=args.num_workers,
            )
        )
    if not args.skip_profile:
        outputs.append(run_profiles(config, device, latency_override=False if args.no_latency else None))
    if not args.skip_rank:
        outputs.append(run_rank_analysis(config))
    print("Wrote manifests:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
