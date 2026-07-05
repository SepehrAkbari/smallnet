'''
Run the SPL-ready Pascal VOC/DeepLabV3 validation experiment.
'''

import argparse
import copy
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
from src.smallnet.data import VOC_CLASS_NAMES, make_voc_loader
from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.evaluation import evaluate_segmentation
from src.smallnet.factorization import replace_conv_layer
from src.smallnet.models import build_deeplabv3_resnet50
from src.smallnet.modules import count_parameters, get_module, select_top_conv_layers
from src.smallnet.profiling import latency_ms, manual_macs
from src.smallnet.results import save_manifest, write_csv


def auto_device(config_device=None):
    if config_device:
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_layers(model, config):
    layers = config.get("layers")
    if layers:
        return layers
    selected = select_top_conv_layers(
        model,
        limit=config.get("top_k_layers", 3),
        min_params=config.get("min_params", 100_000),
    )
    return [row.name for row in selected]


def evaluate_variant(label, model, loader, device, config, max_batches=None):
    hist, summary = evaluate_segmentation(
        model,
        loader,
        num_classes=21,
        device=device,
        ignore_index=config["dataset"].get("ignore_index", 255),
        class_names=VOC_CLASS_NAMES,
        max_batches=max_batches,
    )
    return {
        "label": label,
        "parameters": count_parameters(model),
        "summary": summary,
        "confusion_matrix": hist.astype(int).tolist(),
    }


def profile_variant(label, model, device, profile_cfg):
    input_size = tuple(profile_cfg["input_size"])
    model = model.to(device)
    macs, layer_records = manual_macs(model, input_size, device=device)
    latency = None
    if profile_cfg.get("latency", False):
        latency = latency_ms(
            model,
            input_size,
            device,
            warmup=profile_cfg.get("warmup", 20),
            iterations=profile_cfg.get("iterations", 100),
        )
    return {
        "label": label,
        "parameters": count_parameters(model),
        "macs": macs,
        "latency_ms": latency,
        "input_size": list(input_size),
        "layer_records": layer_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spl/voc_deeplab.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional CPU smoke-test limit.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override eval DataLoader workers.")
    parser.add_argument("--download", action="store_true", help="Override config and download VOC.")
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke-test model construction without weight download.")
    parser.add_argument("--output-dir", default=None, help="Override config output directory.")
    parser.add_argument("--dense-only", action="store_true", help="Skip factorized variants.")
    parser.add_argument("--factor-layers", nargs="+", default=None, help="Optional layer names to factorize.")
    parser.add_argument("--factor-ranks", nargs="+", type=int, default=None, help="Optional CP ranks to evaluate.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-rank", action="store_true")
    parser.add_argument("--no-latency", action="store_true", help="Override config and skip latency timing.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output_dir = ensure_dir(config["output_dir"])
    device = auto_device(args.device or config.get("device"))

    pretrained = config["model"].get("pretrained", True) and not args.no_pretrained
    model = build_deeplabv3_resnet50(pretrained=pretrained)
    target_layers = [] if args.skip_rank else select_layers(model, config["rank_analysis"])

    diagnostics = {}
    rank_rows = []
    for layer_name in target_layers:
        layer = get_module(model, layer_name)
        diagnostic = rank_energy_diagnostic(
            layer.weight.detach(),
            modes=config["rank_analysis"].get("modes", [0, 1, 2, 3]),
            thresholds=config["rank_analysis"].get("thresholds", [0.9, 0.95, 0.99]),
            fixed_ranks=config["rank_analysis"].get("fixed_ranks", [32, 64, 128]),
        )
        diagnostics[layer_name] = diagnostic
        row = {"layer": layer_name, "params": diagnostic["params"], "shape": diagnostic["shape"]}
        for threshold, selected_rank in diagnostic["rank_energy_thresholds"].items():
            row[f"r_tau_{threshold}"] = selected_rank
        for rank, tail in diagnostic["cp_necessary_tail_energy"].items():
            row[f"max_tail_r{rank}"] = tail
        rank_rows.append(row)

    rank_manifest = None
    if not args.skip_rank:
        write_csv(output_dir / "voc_rank_diagnostics.csv", rank_rows)
        rank_manifest = save_manifest(
            output_dir / "voc_rank_manifest.json",
            {
                "experiment_id": config["experiment_id"],
                "kind": "voc_rank_analysis",
                "config_path": config["_config_path"],
                "pretrained": pretrained,
                "target_layers": target_layers,
                "diagnostics": diagnostics,
            },
        )

    evaluations = []
    profiles = []
    loader = None
    if not args.skip_eval:
        dataset_cfg = config["dataset"]
        loader = make_voc_loader(
            dataset_cfg["root"],
            split=dataset_cfg.get("split", "val"),
            batch_size=dataset_cfg.get("batch_size", 4),
            num_workers=dataset_cfg.get("num_workers", 2) if args.num_workers is None else args.num_workers,
            image_size=tuple(dataset_cfg.get("image_size", [520, 520])),
            download=args.download or dataset_cfg.get("download", False),
        )
        evaluations.append(evaluate_variant("deeplabv3_dense", model, loader, device, config, args.max_batches))

    profile_cfg = dict(config["profile"])
    if args.no_latency:
        profile_cfg["latency"] = False

    if not args.skip_profile:
        profiles.append(profile_variant("deeplabv3_dense", model, device, profile_cfg))

    factor_cfg = config.get("factorization", {})
    if factor_cfg.get("enabled", True) and target_layers and not args.dense_only:
        factor_layers = args.factor_layers or target_layers[: factor_cfg.get("max_layers", 3)]
        factor_ranks = args.factor_ranks or factor_cfg.get("ranks", [32, 64, 128])
        for layer_name in factor_layers:
            for rank in factor_ranks:
                variant = copy.deepcopy(model)
                replace_conv_layer(
                    variant,
                    layer_name,
                    rank=rank,
                    factorization=factor_cfg.get("factorization", "cp"),
                    init=factor_cfg.get("init", "random"),
                    n_iter_max=factor_cfg.get("n_iter_max", 0),
                )
                label = f"{layer_name}_rank{rank}"
                if loader is not None:
                    evaluations.append(evaluate_variant(label, variant, loader, device, config, args.max_batches))
                if not args.skip_profile:
                    profiles.append(profile_variant(label, variant, device, profile_cfg))

    eval_manifest = None
    if evaluations:
        write_csv(
            output_dir / "voc_eval_summary.csv",
            [
                {
                    "label": item["label"],
                    "parameters": item["parameters"],
                    "pixel_accuracy": item["summary"]["pixel_accuracy"],
                    "mean_iou_all_classes": item["summary"]["mean_iou_all_classes"],
                    "mean_iou_present_classes": item["summary"]["mean_iou_present_classes"],
                    "frequency_weighted_iou": item["summary"]["frequency_weighted_iou"],
                }
                for item in evaluations
            ],
        )
        eval_manifest = save_manifest(
            output_dir / "voc_eval_manifest.json",
            {
                "experiment_id": config["experiment_id"],
                "kind": "voc_evaluation",
                "config_path": config["_config_path"],
                "device": device.type,
                "max_batches": args.max_batches,
                "num_workers_override": args.num_workers,
                "evaluations": evaluations,
            },
        )

    profile_manifest = None
    if profiles:
        write_csv(
            output_dir / "voc_profile_summary.csv",
            [
                {
                    "label": row["label"],
                    "parameters": row["parameters"],
                    "macs": row["macs"],
                    "latency_ms": row["latency_ms"],
                }
                for row in profiles
            ],
        )
        profile_manifest = save_manifest(
            output_dir / "voc_profile_manifest.json",
            {
                "experiment_id": config["experiment_id"],
                "kind": "voc_profile",
                "config_path": config["_config_path"],
                "device": device.type,
                "profiles": profiles,
            },
        )

    print("Wrote manifests:")
    for path in [rank_manifest, eval_manifest, profile_manifest]:
        if path:
            print(f"  {path}")
    if not any([rank_manifest, eval_manifest, profile_manifest]):
        print("No manifests requested; runner imports and model construction succeeded.")


if __name__ == "__main__":
    main()
