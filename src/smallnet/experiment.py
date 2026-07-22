'''
Canonical CamVid/VGG/CP experiment pipeline.
'''

import csv
import gc
import importlib.metadata
import json
import resource
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from src.smallnet.config import ensure_dir
from src.smallnet.data import (
    camvid_split_available,
    class_validation_options,
    load_camvid_class_names,
    make_camvid_loader,
    parse_camvid_class_dict,
    pairing_rule_from_config,
    strict_unknown_colors_from_config,
    validate_camvid_data,
    sha256_file,
)
from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.evaluation import evaluate_segmentation
from src.smallnet.factorization import MatrixLowRankConv2d, build_factorized_model_from_dense
from src.smallnet.models import build_vgg16_fcn32s, load_vgg16_fcn32s_checkpoint
from src.smallnet.mask_forensics import aggregate_unknown_colors_by_file, inspect_mask_forensics
from src.smallnet.modules import count_parameters, get_module, set_module
from src.smallnet.paper import write_rank_energy_artifacts
from src.smallnet.profiling import latency_stats, manual_macs
from src.smallnet.reproducibility import device_name, set_seed
from src.smallnet.results import load_manifest, save_config_snapshot, save_manifest, write_csv
from src.smallnet.structural import (
    aggregate_cp_reconstruction_rows,
    cp_conv_parameter_count,
    dense_conv_parameter_count,
    exploratory_correlations,
    fit_cp_approximation,
    join_structural_tradeoffs,
    matrix_svd_conv_parameter_count,
    normalize_reconstruction_rows,
    normalized_frobenius_residual,
    output_mode_svd,
    reconstruction_rows_for_conv,
    representation_macs,
    scientific_reconstruction_key,
    write_reconstruction_figure,
    write_unfolding_energy_figure,
    write_zero_shot_figure,
)
from src.smallnet.cp_iteration_sensitivity import (
    aggregate_cp_iteration_rows,
    apply_residual_reduction_comparisons,
    cp_iteration_rows_for_conv,
    normalize_cp_iteration_rows,
    reference_diagnostic_for_conv,
    scientific_cp_iteration_key,
    sensitivity_rank_ordering_changes,
    write_cp_iteration_sensitivity_audit,
    write_cp_iteration_sensitivity_figure,
)


def resolve_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def experiment_output_dir(config, root):
    return ensure_dir(resolve_path(config.get("output_dir", "results/camvid_vgg_cp"), root))


def prepare_run_outputs(config, root, stage):
    output_dir = experiment_output_dir(config, root)
    save_config_snapshot(output_dir / f"{stage}_config_used.json", config)
    return output_dir


def model_config(config):
    return config.get("model", {})


def dataset_config(config):
    return config.get("dataset", {})


def training_config(config):
    return config.get("training", {})


def cp_config(config):
    return config.get("cp", {})


def eval_config(config):
    return config.get("evaluation", {})


def profile_config(config):
    return config.get("profiling", {})


def rank_config(config):
    return config.get("rank_diagnostics", {})


def reconstruction_config(config):
    return config.get("reconstruction", {})


def cp_iteration_sensitivity_config(config):
    return config.get("cp_iteration_sensitivity", {})


def existing_finetuned_config(config):
    return config.get("existing_finetuned_checkpoints", {})


def factorization_method_note(config):
    cp_cfg = cp_config(config)
    return (
        "post-training CP factorization fitted to the dense convolution with "
        f"FactorizedConv.from_conv(init={cp_cfg.get('init', 'random')!r}, "
        f"n_iter_max={cp_cfg.get('n_iter_max', 0)}); this is not a randomly "
        "initialized neural layer unless n_iter_max is zero and no fitting is performed"
    )


def dense_checkpoint_path(config, root):
    checkpoint = model_config(config).get("dense_checkpoint")
    if not checkpoint:
        raise KeyError("model.dense_checkpoint is required")
    return resolve_path(checkpoint, root)


def class_dict_path(config, root):
    data_cfg = dataset_config(config)
    if data_cfg.get("class_dict_path"):
        return resolve_path(data_cfg["class_dict_path"], root)
    return resolve_path(data_cfg["root"], root) / "class_dict.csv"


def data_root(config, root):
    return resolve_path(dataset_config(config)["root"], root)


def image_size(config):
    return tuple(dataset_config(config).get("image_size", [352, 480]))


def configured_splits(config, include_train=True):
    splits = list(eval_config(config).get("splits", ["val", "test"]))
    if include_train and "train" not in splits and "train" in dataset_config(config).get("splits", []):
        splits = ["train", *splits]
    return splits


def ignore_index_and_name(config):
    data_cfg = dataset_config(config)
    ignore_index = data_cfg.get("ignore_index")
    ignore_name = data_cfg.get("ignore_class_name")
    if ignore_index is not None:
        ignore_index = int(ignore_index)
    return ignore_index, ignore_name


def make_model(config):
    model_cfg = model_config(config)
    return build_vgg16_fcn32s(
        num_classes=model_cfg.get("num_classes", dataset_config(config).get("num_classes", 32)),
        pretrained=model_cfg.get("pretrained", False),
    )


def load_dense_model(config, root):
    model = make_model(config)
    state_dict = torch.load(dense_checkpoint_path(config, root), map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def parameter_reference(config, root):
    dense = load_dense_model(config, root)
    target_layer = model_config(config).get("target_layer", "classifier.0")
    return {
        "dense_total_parameters": count_parameters(dense),
        "dense_target_layer_parameters": count_parameters(get_module(dense, target_layer)),
        "target_layer": target_layer,
    }


def parameter_accounting(model, reference):
    total = count_parameters(model)
    target = count_parameters(get_module(model, reference["target_layer"]))
    dense_total = reference["dense_total_parameters"]
    dense_target = reference["dense_target_layer_parameters"]
    return {
        "total_parameters": total,
        "target_layer_parameters": target,
        "dense_target_layer_parameters": dense_target,
        "target_layer_compression_ratio": float(target / dense_target) if dense_target else 0.0,
        "total_compression_ratio": float(total / dense_total) if dense_total else 0.0,
    }


def make_cp_model(config, root, rank, save_path=None):
    model = make_model(config)
    cp_cfg = cp_config(config)
    return build_factorized_model_from_dense(
        model,
        dense_checkpoint_path(config, root),
        layer_name=model_config(config).get("target_layer", "classifier.0"),
        rank=rank,
        factorization=cp_cfg.get("factorization", "cp"),
        init=cp_cfg.get("init", "random"),
        n_iter_max=cp_cfg.get("n_iter_max", 0),
        save_path=save_path,
    )[0]


def load_existing_cp_model(config, checkpoint_path, rank=None):
    model, inferred_rank = load_vgg16_fcn32s_checkpoint(
        checkpoint_path,
        num_classes=model_config(config).get("num_classes", dataset_config(config).get("num_classes", 32)),
        cp_layer=model_config(config).get("target_layer", "classifier.0"),
    )
    return model, inferred_rank if inferred_rank is not None else rank


def loader_for_split(config, root, split, shuffle=False):
    train_cfg = training_config(config)
    eval_cfg = eval_config(config)
    data_cfg = dataset_config(config)
    return make_camvid_loader(
        data_root(config, root),
        split=split,
        batch_size=train_cfg.get("batch_size", eval_cfg.get("batch_size", 4)),
        num_workers=train_cfg.get("num_workers", eval_cfg.get("num_workers", 2)),
        image_size=image_size(config),
        class_dict_path=class_dict_path(config, root),
        shuffle=shuffle,
        pairing_rule=pairing_rule_from_config(data_cfg),
        strict_unknown_colors=strict_unknown_colors_from_config(data_cfg),
        unknown_color_ignore_index=data_cfg.get("unknown_color_ignore_index"),
        num_classes=data_cfg.get("num_classes"),
        ignore_index=data_cfg.get("ignore_index"),
        ignore_class_name=data_cfg.get("ignore_class_name"),
        allow_non_contiguous_class_indices=bool(data_cfg.get("allow_non_contiguous_class_indices", False)),
        required_class_at_index=data_cfg.get("required_class_at_index"),
    )


def run_dataset_validation(config, root, device=None, max_batches=None):
    stage = "dataset_validation"
    output_dir = experiment_output_dir(config, root)
    save_config_snapshot(output_dir / f"{stage}_config_used.json", config)
    report, summary_rows, class_count_rows = validate_camvid_data(config, root)
    definitions = parse_camvid_class_dict(class_dict_path(config, root))
    affected_paths = sorted({item["mask"] for item in report["unknown_rgb_values"]})
    unknown_by_file = aggregate_unknown_colors_by_file(affected_paths, definitions)
    report["unknown_colors_by_file"] = unknown_by_file
    forensic_payload = {
        "schema": "smallnet.dataset_mask_forensics.v1",
        "affected_file_count": len(affected_paths),
        "affected_files": [inspect_mask_forensics(path, class_dict_path(config, root)) for path in affected_paths],
        "evidence_note": "Spatial and RGB intermediacy tests are evidence only; authoritative-source comparison is required to establish provenance.",
    }
    report_path = save_manifest(output_dir / f"{stage}_report.json", report, device=device)
    write_csv(output_dir / f"{stage}_summary.csv", summary_rows)
    write_csv(output_dir / "dataset_class_counts.csv", class_count_rows)
    write_csv(
        output_dir / "dataset_unknown_colors_by_file.csv",
        [
            {
                **{key: value for key, value in row.items() if key not in {"unknown_colors", "connected_region_sizes"}},
                "bounding_box_ymin_xmin_ymax_xmax": ",".join(map(str, row["bounding_box_ymin_xmin_ymax_xmax"])),
                "connected_region_sizes": ",".join(map(str, row["connected_region_sizes"])),
            }
            for row in unknown_by_file
        ],
        fieldnames=["mask_path", "total_unknown_pixels", "distinct_unknown_rgb_values", "bounding_box_ymin_xmin_ymax_xmax", "affected_proportion", "connected_region_count_4_neighbor", "all_unknown_pixels_in_one_connected_region", "connected_region_sizes"],
    )
    save_manifest(output_dir / "dataset_mask_forensics.json", forensic_payload, device=device)
    if report["status"] != "pass":
        raise RuntimeError(f"Dataset validation failed; see {report_path}")
    return report_path


def evaluate_splits(
    model,
    config,
    root,
    device,
    label,
    rank,
    max_batches=None,
    splits=None,
    model_kind="",
    checkpoint="",
    parameter_ref=None,
):
    names = load_camvid_class_names(class_dict_path(config, root))
    ignore_index, ignore_name = ignore_index_and_name(config)
    parameter_ref = parameter_ref or parameter_reference(config, root)
    accounting = parameter_accounting(model, parameter_ref)
    rows = []
    evaluations = []
    skipped = []

    for split in (splits or configured_splits(config, include_train=False)):
        if not camvid_split_available(data_root(config, root), split):
            skipped.append(
                {
                    "split": split,
                    "reason": f"Missing {split} image directory or {split}_labels directory",
                }
            )
            continue
        loader = loader_for_split(config, root, split, shuffle=False)
        hist, summary = evaluate_segmentation(
            model,
            loader,
            dataset_config(config).get("num_classes", len(names)),
            device,
            ignore_index=ignore_index,
            class_names=names,
            max_batches=max_batches,
        )
        summary.update(
            {
                "label": label,
                "rank": rank,
                "split": split,
                "model_kind": model_kind,
                "checkpoint": checkpoint,
                "parameters": accounting["total_parameters"],
                **accounting,
                "ignore_index": "" if ignore_index is None else ignore_index,
                "ignore_class": ignore_name or "",
            }
        )
        rows.append(
            {
                "label": label,
                "rank": rank,
                "split": split,
                "model_kind": model_kind,
                "checkpoint": checkpoint,
                "parameters": accounting["total_parameters"],
                **accounting,
                "pixel_accuracy": summary["pixel_accuracy"],
                "mean_iou_all_classes": summary["mean_iou_all_classes"],
                "mean_iou_present_classes": summary["mean_iou_present_classes"],
                "frequency_weighted_iou": summary["frequency_weighted_iou"],
                "ignore_index": "" if ignore_index is None else ignore_index,
                "ignore_class": ignore_name or "",
            }
        )
        evaluations.append(
            {
                "label": label,
                "rank": rank,
                "split": split,
                "summary": summary,
                "confusion_matrix": hist.astype(int).tolist(),
            }
        )

    return rows, evaluations, skipped


def run_dense_evaluation(config, root, device, max_batches=None):
    stage = "dense_eval"
    output_dir = prepare_run_outputs(config, root, stage)
    model = load_dense_model(config, root)
    parameter_ref = parameter_reference(config, root)
    rows, evaluations, skipped = evaluate_splits(
        model,
        config,
        root,
        device,
        label="dense",
        rank="dense",
        max_batches=max_batches,
        model_kind="dense",
        checkpoint=str(dense_checkpoint_path(config, root)),
        parameter_ref=parameter_ref,
    )
    write_csv(output_dir / f"{stage}_summary.csv", rows)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "dense_baseline_evaluation",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "dense_checkpoint": str(dense_checkpoint_path(config, root)),
            "parameter_reference": parameter_ref,
            "evaluations": evaluations,
            "skipped_splits": skipped,
        },
        device=device,
    )


def iter_rank_seed(config, rank_key="ranks"):
    cp_cfg = cp_config(config)
    ranks = cp_cfg.get(rank_key, cp_cfg.get("ranks", [64, 128, 256]))
    seeds = cp_cfg.get("seeds", training_config(config).get("random_seeds", [0]))
    for rank in ranks:
        for seed in seeds:
            yield int(rank), int(seed)


def run_cp_zero_shot(config, root, device, max_batches=None):
    stage = "cp_zero_shot"
    output_dir = prepare_run_outputs(config, root, stage)
    all_rows = []
    all_evaluations = []
    skipped_splits = []
    parameter_ref = parameter_reference(config, root)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_checkpoints = bool(cp_config(config).get("save_zero_shot_checkpoints", False))

    for rank, seed in iter_rank_seed(config, rank_key="zero_shot_ranks"):
        seed_status = set_seed(seed, deterministic=training_config(config).get("deterministic", True))
        save_path = checkpoint_dir / f"cp_rank_{rank}_seed_{seed}_zero_shot.pth" if save_checkpoints else None
        model = make_cp_model(config, root, rank=rank, save_path=save_path)
        label = f"cp_rank_{rank}_seed_{seed}_zero_shot"
        rows, evaluations, skipped = evaluate_splits(
            model,
            config,
            root,
            device,
            label=label,
            rank=rank,
            max_batches=max_batches,
            model_kind="cp_zero_shot",
            checkpoint=str(save_path or dense_checkpoint_path(config, root)),
            parameter_ref=parameter_ref,
        )
        for row in rows:
            row["seed"] = seed
            row["checkpoint_saved"] = str(save_path or "")
        for item in evaluations:
            item["seed"] = seed
            item["seed_status"] = seed_status
            item["checkpoint_saved"] = str(save_path or "")
        all_rows.extend(rows)
        all_evaluations.extend(evaluations)
        skipped_splits.extend(skipped)

    write_csv(output_dir / f"{stage}_summary.csv", all_rows)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "cp_zero_shot_evaluation",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "factorization_method": factorization_method_note(config),
            "parameter_reference": parameter_ref,
            "evaluations": all_evaluations,
            "skipped_splits": skipped_splits,
        },
        device=device,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batches = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def validation_score(model, config, root, device, max_batches=None):
    rows, _, _ = evaluate_splits(
        model,
        config,
        root,
        device,
        label="validation",
        rank="",
        max_batches=max_batches,
        splits=["val"],
    )
    val_rows = [row for row in rows if row["split"] == "val"]
    if not val_rows:
        return None
    return val_rows[0]["mean_iou_present_classes"]


def freeze_for_finetuning(model, config):
    if training_config(config).get("freeze_features", True) and hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False
    return model


def run_cp_finetune(config, root, device, max_batches=None):
    stage = "cp_finetune"
    output_dir = prepare_run_outputs(config, root, stage)
    if not camvid_split_available(data_root(config, root), "train"):
        raise FileNotFoundError("Fine-tuning requires train and train_labels directories")

    all_rows = []
    all_evaluations = []
    skipped_splits = []
    training_records = []
    parameter_ref = parameter_reference(config, root)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_checkpoints = bool(cp_config(config).get("save_finetuned_checkpoints", False))
    ignore_index, _ = ignore_index_and_name(config)

    train_cfg = training_config(config)
    epochs = int(train_cfg.get("fine_tuning_epochs", 0))
    if epochs <= 0:
        raise ValueError("training.fine_tuning_epochs must be positive for cp_finetune")

    for rank, seed in iter_rank_seed(config, rank_key="fine_tune_ranks"):
        seed_status = set_seed(seed, deterministic=train_cfg.get("deterministic", True))
        model = make_cp_model(config, root, rank=rank).to(device)
        freeze_for_finetuning(model, config)
        train_loader = loader_for_split(config, root, "train", shuffle=True)
        criterion_kwargs = {} if ignore_index is None else {"ignore_index": ignore_index}
        criterion = nn.CrossEntropyLoss(**criterion_kwargs)
        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(train_cfg.get("learning_rate", 1e-4)),
        )

        best_score = None
        best_state = None
        losses = []
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            score = validation_score(model, config, root, device, max_batches=max_batches)
            losses.append({"epoch": epoch, "train_loss": loss, "val_mean_iou_present_classes": score})
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_state = deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
        save_path = checkpoint_dir / f"cp_rank_{rank}_seed_{seed}_finetuned.pth" if save_checkpoints else None
        if save_path:
            torch.save(model.cpu().state_dict(), save_path)
            model = model.to(device)

        label = f"cp_rank_{rank}_seed_{seed}_finetuned"
        rows, evaluations, skipped = evaluate_splits(
            model,
            config,
            root,
            device,
            label=label,
            rank=rank,
            max_batches=max_batches,
            model_kind="cp_finetuned_new",
            checkpoint=str(save_path or ""),
            parameter_ref=parameter_ref,
        )
        for row in rows:
            row["seed"] = seed
            row["epochs"] = epochs
            row["best_val_mean_iou_present_classes"] = "" if best_score is None else best_score
            row["checkpoint_saved"] = str(save_path or "")
        for item in evaluations:
            item["seed"] = seed
            item["seed_status"] = seed_status
            item["epochs"] = epochs
            item["best_val_mean_iou_present_classes"] = best_score
            item["checkpoint_saved"] = str(save_path or "")
        training_records.append(
            {
                "rank": rank,
                "seed": seed,
                "epochs": epochs,
                "losses": losses,
                "best_val_mean_iou_present_classes": best_score,
            }
        )
        all_rows.extend(rows)
        all_evaluations.extend(evaluations)
        skipped_splits.extend(skipped)

    write_csv(output_dir / f"{stage}_summary.csv", all_rows)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "cp_finetune_evaluation",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "factorization_method": factorization_method_note(config),
            "parameter_reference": parameter_ref,
            "training_records": training_records,
            "evaluations": all_evaluations,
            "skipped_splits": skipped_splits,
        },
        device=device,
    )


def run_existing_finetuned_evaluation(config, root, device, max_batches=None):
    stage = "existing_finetuned"
    output_dir = prepare_run_outputs(config, root, stage)
    rows = []
    evaluations = []
    skipped = []
    parameter_ref = parameter_reference(config, root)

    for rank_text, configured_path in existing_finetuned_config(config).items():
        rank = int(rank_text)
        checkpoint = resolve_path(configured_path, root)
        if not checkpoint.is_file():
            skipped.append(
                {
                    "rank": rank,
                    "checkpoint": str(checkpoint),
                    "reason": "checkpoint file is missing",
                }
            )
            continue
        model, actual_rank = load_existing_cp_model(config, checkpoint, rank=rank)
        label = f"cp_rank_{actual_rank}_existing_finetuned"
        rank_rows, rank_evaluations, split_skips = evaluate_splits(
            model,
            config,
            root,
            device,
            label=label,
            rank=actual_rank,
            max_batches=max_batches,
            model_kind="cp_finetuned_existing",
            checkpoint=str(checkpoint),
            parameter_ref=parameter_ref,
        )
        rows.extend(rank_rows)
        evaluations.extend(rank_evaluations)
        skipped.extend(split_skips)

    write_csv(output_dir / "existing_finetuned_summary.csv", rows)
    return save_manifest(
        output_dir / "existing_finetuned_metadata.json",
        {
            "kind": "existing_finetuned_evaluation",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "parameter_reference": parameter_ref,
            "evaluations": evaluations,
            "skipped": skipped,
        },
        device=device,
    )


def run_profiling(config, root, device):
    stage = "profile"
    output_dir = prepare_run_outputs(config, root, stage)
    prof_cfg = profile_config(config)
    h, w = image_size(config)
    batch_size = int(prof_cfg.get("batch_size", 1))
    input_size = tuple(prof_cfg.get("input_size", [batch_size, 3, h, w]))
    run_latency = bool(prof_cfg.get("latency", False))
    rows = []
    profiles = []
    parameter_ref = parameter_reference(config, root)
    actual_device_name = device_name(device)

    models = [("dense", "dense", "dense", str(dense_checkpoint_path(config, root)), load_dense_model(config, root))]
    for rank, seed in iter_rank_seed(config, rank_key="profile_ranks"):
        set_seed(seed, deterministic=training_config(config).get("deterministic", True))
        models.append(
            (
                f"cp_rank_{rank}_seed_{seed}_zero_shot",
                rank,
                "cp_zero_shot",
                str(dense_checkpoint_path(config, root)),
                make_cp_model(config, root, rank),
            )
        )
    if prof_cfg.get("include_existing_finetuned", True):
        for rank_text, configured_path in existing_finetuned_config(config).items():
            checkpoint = resolve_path(configured_path, root)
            if checkpoint.is_file():
                model, actual_rank = load_existing_cp_model(config, checkpoint, rank=int(rank_text))
                models.append(
                    (
                        f"cp_rank_{actual_rank}_existing_finetuned",
                        actual_rank,
                        "cp_finetuned_existing",
                        str(checkpoint),
                        model,
                    )
                )

    for label, rank, model_kind, checkpoint, model in models:
        model = model.to(device)
        macs, records = manual_macs(model, input_size, device=device)
        latency = {
            "latency_mean_ms": None,
            "latency_std_ms": None,
            "latency_median_ms": None,
            "latency_min_ms": None,
            "latency_max_ms": None,
            "latency_warmup_iterations": int(prof_cfg.get("warmup", 10)),
            "latency_iterations": int(prof_cfg.get("iterations", 50)),
            "device_name": actual_device_name,
        }
        if run_latency:
            latency = latency_stats(
                model,
                input_size,
                device,
                warmup=int(prof_cfg.get("warmup", 10)),
                iterations=int(prof_cfg.get("iterations", 50)),
                device_name=actual_device_name,
            )
        accounting = parameter_accounting(model, parameter_ref)
        row = {
            "label": label,
            "rank": rank,
            "model_kind": model_kind,
            "checkpoint": checkpoint,
            "device": device.type,
            "parameters": accounting["total_parameters"],
            **accounting,
            "macs": macs,
            "latency_ms": latency["latency_mean_ms"],
            **latency,
        }
        rows.append(row)
        profiles.append({**row, "input_size": list(input_size), "layer_records": records})

    write_csv(output_dir / f"{stage}_summary.csv", rows)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "profile",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "parameter_reference": parameter_ref,
            "device": device.type,
            "device_name": actual_device_name,
            "profiles": profiles,
        },
        device=device,
    )


def run_rank_diagnostics(config, root, device=None):
    stage = "rank_diagnostics"
    output_dir = prepare_run_outputs(config, root, stage)
    rank_cfg = rank_config(config)
    state_dict = torch.load(dense_checkpoint_path(config, root), map_location="cpu")
    target_layer = model_config(config).get("target_layer", "classifier.0")
    layers = rank_cfg.get("layers", [f"{target_layer}.weight"])
    fixed_ranks = rank_cfg.get("fixed_ranks", cp_config(config).get("ranks", [64, 128, 256]))
    rows = []
    diagnostics = {}

    for layer in layers:
        if layer not in state_dict:
            raise KeyError(f"Checkpoint does not contain tensor {layer}")
        diagnostic = rank_energy_diagnostic(
            state_dict[layer],
            modes=rank_cfg.get("modes", [0, 1, 2, 3]),
            thresholds=rank_cfg.get("thresholds", [0.9, 0.95, 0.99]),
            fixed_ranks=fixed_ranks,
        )
        diagnostics[layer] = diagnostic
        row = {"layer": layer, "params": diagnostic["params"], "shape": diagnostic["shape"]}
        for threshold, selected_rank in diagnostic["rank_energy_thresholds"].items():
            row[f"r_tau_{threshold}"] = selected_rank
        for rank, tail in diagnostic["cp_necessary_tail_energy"].items():
            row[f"max_tail_r{rank}"] = tail
        rows.append(row)

    write_csv(output_dir / f"{stage}_summary.csv", rows)
    paper_outputs = write_rank_energy_artifacts(diagnostics, config, root)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "rank_energy_diagnostics",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "dense_checkpoint": str(dense_checkpoint_path(config, root)),
            "diagnostics": diagnostics,
            "paper_outputs": paper_outputs,
            "interpretation_note": (
                "Unfolding rank-energy is a necessary structural diagnostic for CP compression, "
                "not proof that a CP-rank model will preserve downstream accuracy."
            ),
        },
        device=device,
    )


def _software_versions():
    versions = {}
    for distribution in ("torch", "torchvision", "tensorly", "tensorly-torch", "numpy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _paper_figures_dir(config, root):
    return ensure_dir(resolve_path(config.get("paper", {}).get("figures_dir", "results/paper/figures"), root))


def _reconstruction_conv(config, root):
    recon_cfg = reconstruction_config(config)
    synthetic_shape = recon_cfg.get("synthetic_tensor_shape")
    if synthetic_shape:
        cout, cin, kh, kw = map(int, synthetic_shape)
        set_seed(int(recon_cfg.get("synthetic_seed", 123)), deterministic=True)
        return nn.Conv2d(cin, cout, (kh, kw), bias=bool(recon_cfg.get("synthetic_bias", True)))
    model = load_dense_model(config, root)
    layer = get_module(model, model_config(config).get("target_layer", "classifier.0"))
    if not isinstance(layer, nn.Conv2d):
        raise TypeError(f"Reconstruction target must be nn.Conv2d, got {type(layer)}")
    return layer


def run_reconstruction(config, root, device, max_batches=None):
    '''Run exact spectral, CP, and output-unfolding matrix-SVD diagnostics.'''
    stage = "reconstruction"
    output_dir = prepare_run_outputs(config, root, stage)
    recon_cfg = reconstruction_config(config)
    canonical_ranks = [int(value) for value in recon_cfg.get("ranks", cp_config(config).get("ranks", [32, 64, 128, 256, 512]))]
    canonical_seeds = [int(value) for value in recon_cfg.get("seeds", [0, 1, 2])]
    ranks = [int(value) for value in recon_cfg.get("execution_ranks", canonical_ranks)]
    seeds = [int(value) for value in recon_cfg.get("execution_seeds", canonical_seeds)]
    init = recon_cfg.get("init", cp_config(config).get("init", "random"))
    n_iter_max = int(recon_cfg.get("n_iter_max", cp_config(config).get("n_iter_max", 10)))
    tolerance = float(recon_cfg.get("numerical_tolerance", 1e-5))
    memory_efficient_mttkrp = bool(recon_cfg.get("memory_efficient_mttkrp", True))
    mttkrp_rank_chunk_size = int(recon_cfg.get("mttkrp_rank_chunk_size", 64))
    mttkrp_max_explicit_bytes = int(recon_cfg.get("mttkrp_max_explicit_bytes", 512 * 1024**2))
    summary_path = output_dir / "reconstruction_summary.csv"
    raw_loaded_rows = _read_reconstruction_rows(summary_path) if summary_path.is_file() else []
    normalized_loaded, preserved_raw_rows, row_diagnostics = normalize_reconstruction_rows(
        raw_loaded_rows,
        context="incremental reconstruction CSV load",
    )
    persisted_rows = {
        scientific_reconstruction_key(row): row for row in normalized_loaded
    }
    requested_keys = {
        *(("matrix_svd_output_unfolding", rank) for rank in ranks),
        *(("cp", rank, seed) for rank in ranks for seed in seeds),
    }
    completed_keys = {
        key for key, row in persisted_rows.items() if row.get("status") == "completed"
    }
    if not (requested_keys - completed_keys):
        return run_reconstruction_figures(config, root, device=device)

    def persist(rows):
        normalized_new, rejected_new, new_diagnostics = normalize_reconstruction_rows(
            rows,
            context="incremental reconstruction append",
        )
        row_diagnostics.extend(new_diagnostics)
        preserved_raw_rows.extend(rejected_new)
        for row in normalized_new:
            key = scientific_reconstruction_key(row)
            previous = persisted_rows.get(key)
            if previous and previous.get("status") == "completed" and row.get("status") == "failed":
                continue
            persisted_rows[key] = row
        write_csv(summary_path, [*persisted_rows.values(), *preserved_raw_rows])

    conv = _reconstruction_conv(config, root)
    rows, aggregates, diagnostic, failures = reconstruction_rows_for_conv(
        conv,
        ranks,
        seeds,
        init,
        n_iter_max,
        device,
        tolerance=tolerance,
        memory_efficient_mttkrp=memory_efficient_mttkrp,
        mttkrp_rank_chunk_size=mttkrp_rank_chunk_size,
        mttkrp_max_explicit_bytes=mttkrp_max_explicit_bytes,
        skip_keys=completed_keys,
        on_update=persist,
    )
    rows = [*persisted_rows.values(), *preserved_raw_rows]
    aggregates = aggregate_cp_reconstruction_rows(rows, diagnostics=row_diagnostics)
    write_csv(output_dir / "reconstruction_rank_summary.csv", aggregates)
    figures = []
    figure_generation_failures = []
    for figure_name, writer in (
        (
            "unfolding_cumulative_energy",
            lambda: write_unfolding_energy_figure(
                diagnostic,
                canonical_ranks,
                _paper_figures_dir(config, root),
                diagnostics=row_diagnostics,
            ),
        ),
        (
            "reconstruction_squared_error",
            lambda: write_reconstruction_figure(
                rows,
                _paper_figures_dir(config, root),
                diagnostics=row_diagnostics,
            ),
        ),
    ):
        try:
            figures.extend(writer())
        except Exception as exc:
            figure_generation_failures.append(
                {"figure": figure_name, "exception": repr(exc), "nonfatal": True}
            )
    metadata_path = save_manifest(
        output_dir / "reconstruction_metadata.json",
        {
            "kind": "structural_reconstruction_diagnostics",
            "experiment_id": config.get("experiment_id"),
            "dense_checkpoint": "synthetic" if reconstruction_config(config).get("synthetic_tensor_shape") else str(dense_checkpoint_path(config, root)),
            "dense_checkpoint_sha256": "" if reconstruction_config(config).get("synthetic_tensor_shape") else sha256_file(dense_checkpoint_path(config, root)),
            "target_layer": model_config(config).get("target_layer", "classifier.0"),
            "canonical_ranks": canonical_ranks,
            "canonical_cp_seeds": canonical_seeds,
            "execution_ranks": ranks,
            "execution_cp_seeds": seeds,
            "cp_initializer": init,
            "cp_n_iter_max": n_iter_max,
            "memory_efficient_mttkrp": memory_efficient_mttkrp,
            "mttkrp_rank_chunk_size": mttkrp_rank_chunk_size,
            "mttkrp_max_explicit_bytes": mttkrp_max_explicit_bytes,
            "numerical_tolerance": tolerance,
            "device_requested_and_used_for_cp_fitting": device.type,
            "non_output_mode_spectral_device": "cpu",
            "output_mode_matrix_svd_device": diagnostic.get("output_mode_matrix_svd_device", "cpu"),
            "deterministic_settings": [row.get("seed_status") for row in rows if row.get("method") == "cp" and row.get("status") == "completed"],
            "software_versions": _software_versions(),
            "diagnostic": diagnostic,
            "cp_rank_aggregates": aggregates,
            "failures": failures,
            "row_normalization_diagnostics": row_diagnostics,
            "figure_generation_failures": figure_generation_failures,
            "figure_generation_failure_policy": (
                "Figure failures are nonfatal after computation rows have been saved. "
                "Regenerate them with --stage reconstruction-figures."
            ),
            "convergence_metadata_note": (
                "tensorly-torch 0.5 does not expose per-iteration loss history through FactorizedConv.from_conv; "
                "completed CP rows therefore record the requested iteration budget and explicitly mark actual "
                "iterations and convergence certification as unavailable."
            ),
            "peak_process_resident_memory_raw_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "paper_figure_outputs": figures,
        },
        device=device,
    )
    del conv
    gc.collect()
    return metadata_path


def _read_reconstruction_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run_reconstruction_figures(config, root, device=None, max_batches=None):
    '''Regenerate reconstruction figures from saved rows without decomposition.'''
    stage = "reconstruction_figures"
    output_dir = prepare_run_outputs(config, root, stage)
    summary_path = output_dir / "reconstruction_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Saved reconstruction rows are missing: {summary_path}")

    raw_rows = _read_reconstruction_rows(summary_path)
    normalized_rows, preserved_raw_rows, row_diagnostics = normalize_reconstruction_rows(
        raw_rows,
        context="reconstruction figure-only CSV load",
    )
    rows = [*normalized_rows, *preserved_raw_rows]
    write_csv(summary_path, rows)
    aggregates = aggregate_cp_reconstruction_rows(rows, diagnostics=row_diagnostics)
    write_csv(output_dir / "reconstruction_rank_summary.csv", aggregates)

    reconstruction_metadata_path = output_dir / "reconstruction_metadata.json"
    diagnostic = None
    if reconstruction_metadata_path.is_file():
        diagnostic = load_manifest(reconstruction_metadata_path).get("diagnostic")

    canonical_ranks = [
        int(value)
        for value in reconstruction_config(config).get(
            "ranks", cp_config(config).get("ranks", [32, 64, 128, 256, 512])
        )
    ]
    figures = []
    figure_generation_failures = []
    if diagnostic is None:
        figure_generation_failures.append(
            {
                "figure": "unfolding_cumulative_energy",
                "exception": "saved reconstruction metadata does not contain spectral diagnostics",
                "nonfatal": True,
            }
        )
    else:
        try:
            figures.extend(
                write_unfolding_energy_figure(
                    diagnostic,
                    canonical_ranks,
                    _paper_figures_dir(config, root),
                    diagnostics=row_diagnostics,
                )
            )
        except Exception as exc:
            figure_generation_failures.append(
                {"figure": "unfolding_cumulative_energy", "exception": repr(exc), "nonfatal": True}
            )

    try:
        figures.extend(
            write_reconstruction_figure(
                rows,
                _paper_figures_dir(config, root),
                diagnostics=row_diagnostics,
            )
        )
    except Exception as exc:
        figure_generation_failures.append(
            {"figure": "reconstruction_squared_error", "exception": repr(exc), "nonfatal": True}
        )

    return save_manifest(
        output_dir / "reconstruction_figures_metadata.json",
        {
            "kind": "structural_reconstruction_figure_regeneration",
            "experiment_id": config.get("experiment_id"),
            "source_reconstruction_summary": str(summary_path),
            "source_reconstruction_metadata": str(reconstruction_metadata_path),
            "scientific_row_count": len(normalized_rows),
            "preserved_raw_malformed_row_count": len(preserved_raw_rows),
            "row_normalization_diagnostics": row_diagnostics,
            "figure_outputs": figures,
            "figure_generation_failures": figure_generation_failures,
            "figure_generation_failure_policy": (
                "Figure-only failures are recorded as nonfatal diagnostics."
            ),
        },
        device=device,
    )


def _read_cp_iteration_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _cp_iteration_reference_diagnostic(
    config, root, output_dir, conv, ranks, device, checkpoint_hash
):
    '''Reuse the verified canonical spectrum, or compute it for standalone/synthetic runs.'''
    if not cp_iteration_sensitivity_config(config).get("synthetic_tensor_shape"):
        metadata_path = output_dir / "reconstruction_metadata.json"
        if metadata_path.is_file():
            metadata = load_manifest(metadata_path)
            recorded_hash = metadata.get("dense_checkpoint_sha256", "")
            if recorded_hash and recorded_hash != checkpoint_hash:
                raise RuntimeError(
                    "Canonical reconstruction metadata uses a different dense checkpoint; "
                    "refusing to mix spectral references in the sensitivity experiment"
                )
            diagnostic = metadata.get("diagnostic")
            if diagnostic:
                return diagnostic, str(metadata_path)
    return reference_diagnostic_for_conv(conv.weight, ranks, device), "computed_for_sensitivity_stage"


def run_cp_iteration_sensitivity(config, root, device, max_batches=None):
    '''Run an isolated, resumable CP iteration-budget sensitivity experiment.'''
    stage = "cp_iteration_sensitivity"
    output_dir = prepare_run_outputs(config, root, stage)
    sensitivity_cfg = cp_iteration_sensitivity_config(config)
    recon_cfg = reconstruction_config(config)
    canonical_ranks = [int(value) for value in sensitivity_cfg.get("ranks", [128, 256, 512])]
    canonical_seeds = [int(value) for value in sensitivity_cfg.get("seeds", [0, 1, 2])]
    canonical_budgets = [
        int(value) for value in sensitivity_cfg.get("iteration_budgets", [10, 25, 50, 100])
    ]
    ranks = [int(value) for value in sensitivity_cfg.get("execution_ranks", canonical_ranks)]
    seeds = [int(value) for value in sensitivity_cfg.get("execution_seeds", canonical_seeds)]
    iteration_budgets = [
        int(value)
        for value in sensitivity_cfg.get("execution_iteration_budgets", canonical_budgets)
    ]
    if len(set(ranks)) != len(ranks) or any(value <= 0 for value in ranks):
        raise ValueError("Sensitivity ranks must be unique positive integers")
    if len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("Sensitivity seeds must be unique nonnegative integers")
    if len(set(iteration_budgets)) != len(iteration_budgets) or any(
        value <= 0 for value in iteration_budgets
    ):
        raise ValueError("Iteration budgets must be unique positive integers")
    init = sensitivity_cfg.get("init", recon_cfg.get("init", cp_config(config).get("init", "random")))
    tolerance = float(
        sensitivity_cfg.get("numerical_tolerance", recon_cfg.get("numerical_tolerance", 1e-5))
    )
    memory_efficient_mttkrp = bool(
        sensitivity_cfg.get(
            "memory_efficient_mttkrp", recon_cfg.get("memory_efficient_mttkrp", True)
        )
    )
    mttkrp_rank_chunk_size = int(
        sensitivity_cfg.get(
            "mttkrp_rank_chunk_size", recon_cfg.get("mttkrp_rank_chunk_size", 64)
        )
    )
    mttkrp_max_explicit_bytes = int(
        sensitivity_cfg.get(
            "mttkrp_max_explicit_bytes",
            recon_cfg.get("mttkrp_max_explicit_bytes", 512 * 1024**2),
        )
    )
    summary_path = output_dir / "cp_iteration_sensitivity_summary.csv"
    raw_loaded = _read_cp_iteration_rows(summary_path) if summary_path.is_file() else []
    normalized_loaded, preserved_raw_rows, row_diagnostics = normalize_cp_iteration_rows(
        raw_loaded, context="incremental CP iteration-sensitivity CSV load"
    )
    persisted_rows = {scientific_cp_iteration_key(row): row for row in normalized_loaded}
    requested_keys = {
        ("cp", rank, seed, budget)
        for rank in ranks
        for seed in seeds
        for budget in iteration_budgets
    }
    completed_keys = {
        key for key, row in persisted_rows.items() if row.get("status") == "completed"
    }
    expected_initialization_hashes = {}
    for row in normalized_loaded:
        if row.get("status") != "completed":
            continue
        hash_value = str(row.get("initialization_hash_sha256", "")).strip()
        if not hash_value:
            raise RuntimeError(
                "A completed sensitivity row lacks its initialization hash; refusing an unverifiable resume"
            )
        hash_key = (int(row["rank"]), int(row["seed"]))
        previous = expected_initialization_hashes.setdefault(hash_key, hash_value)
        if previous != hash_value:
            raise RuntimeError(
                f"Saved sensitivity rows contain inconsistent initial factors for {hash_key}: "
                f"{previous} != {hash_value}"
            )

    def persist(new_rows):
        normalized_new, rejected_new, new_diagnostics = normalize_cp_iteration_rows(
            new_rows, context="incremental CP iteration-sensitivity append"
        )
        row_diagnostics.extend(new_diagnostics)
        preserved_raw_rows.extend(rejected_new)
        for row in normalized_new:
            key = scientific_cp_iteration_key(row)
            previous = persisted_rows.get(key)
            if previous and previous.get("status") == "completed" and row.get("status") != "completed":
                continue
            persisted_rows[key] = row
        compared_rows, comparison_diagnostics = apply_residual_reduction_comparisons(
            [*persisted_rows.values(), *preserved_raw_rows]
        )
        row_diagnostics.extend(comparison_diagnostics)
        normalized_compared, rejected_compared, _ = normalize_cp_iteration_rows(
            compared_rows, context="incremental comparison persistence", warn=False
        )
        persisted_rows.clear()
        persisted_rows.update(
            {scientific_cp_iteration_key(row): row for row in normalized_compared}
        )
        preserved_raw_rows[:] = rejected_compared
        write_csv(summary_path, [*persisted_rows.values(), *preserved_raw_rows])

    conv = None
    diagnostic = None
    reference_source = ""
    checkpoint_hash = ""
    new_failures = []
    initialization_diagnostics = []
    if requested_keys - completed_keys:
        synthetic_shape = sensitivity_cfg.get("synthetic_tensor_shape")
        if synthetic_shape:
            cout, cin, kh, kw = map(int, synthetic_shape)
            set_seed(int(sensitivity_cfg.get("synthetic_seed", 123)), deterministic=True)
            conv = nn.Conv2d(
                cin,
                cout,
                (kh, kw),
                bias=bool(sensitivity_cfg.get("synthetic_bias", True)),
            )
            checkpoint_hash = "synthetic"
        else:
            conv = _reconstruction_conv(config, root)
            checkpoint_hash = sha256_file(dense_checkpoint_path(config, root))
        diagnostic, reference_source = _cp_iteration_reference_diagnostic(
            config, root, output_dir, conv, canonical_ranks, device, checkpoint_hash
        )
        new_rows, new_failures, initialization_diagnostics = cp_iteration_rows_for_conv(
            conv,
            ranks,
            seeds,
            iteration_budgets,
            init,
            device,
            diagnostic,
            checkpoint_hash,
            tolerance=tolerance,
            memory_efficient_mttkrp=memory_efficient_mttkrp,
            mttkrp_rank_chunk_size=mttkrp_rank_chunk_size,
            mttkrp_max_explicit_bytes=mttkrp_max_explicit_bytes,
            skip_keys=completed_keys,
            expected_initialization_hashes=expected_initialization_hashes,
            on_update=persist,
        )
        persist(new_rows)
    else:
        if sensitivity_cfg.get("synthetic_tensor_shape"):
            checkpoint_hash = "synthetic"
            metadata_path = output_dir / "cp_iteration_sensitivity_metadata.json"
            if metadata_path.is_file():
                previous_metadata = load_manifest(metadata_path)
                diagnostic = previous_metadata.get("diagnostic")
                reference_source = previous_metadata.get("spectral_reference_source", "saved sensitivity metadata")
        else:
            checkpoint_hash = sha256_file(dense_checkpoint_path(config, root))
            reconstruction_metadata_path = output_dir / "reconstruction_metadata.json"
            if reconstruction_metadata_path.is_file():
                reconstruction_metadata = load_manifest(reconstruction_metadata_path)
                if reconstruction_metadata.get("dense_checkpoint_sha256") != checkpoint_hash:
                    raise RuntimeError("Dense checkpoint changed since canonical reconstruction")
                diagnostic = reconstruction_metadata.get("diagnostic")
                reference_source = str(reconstruction_metadata_path)

    all_rows = [*persisted_rows.values(), *preserved_raw_rows]
    compared_rows, comparison_diagnostics = apply_residual_reduction_comparisons(all_rows)
    row_diagnostics.extend(comparison_diagnostics)
    normalized_final, rejected_final, final_diagnostics = normalize_cp_iteration_rows(
        compared_rows, context="final CP iteration-sensitivity persistence", warn=False
    )
    row_diagnostics.extend(final_diagnostics)
    all_rows = [*normalized_final, *rejected_final]
    write_csv(summary_path, all_rows)
    aggregates, aggregate_diagnostics = aggregate_cp_iteration_rows(
        all_rows, expected_seed_count=len(canonical_seeds)
    )
    row_diagnostics.extend(aggregate_diagnostics)
    rank_summary_path = write_csv(
        output_dir / "cp_iteration_sensitivity_rank_summary.csv", aggregates
    )
    rank_ordering = sensitivity_rank_ordering_changes(aggregates)

    failures = [
        {
            "method": row.get("method"),
            "rank": row.get("rank"),
            "seed": row.get("seed"),
            "iteration_budget": row.get("iteration_budget"),
            "exception": row.get("failure_exception", ""),
        }
        for row in normalized_final
        if row.get("status") == "failed"
    ]
    canonical_ten_iteration_reproduction = []
    if not sensitivity_cfg.get("synthetic_tensor_shape"):
        canonical_summary_path = output_dir / "reconstruction_summary.csv"
        if canonical_summary_path.is_file():
            canonical_rows, _, canonical_diagnostics = normalize_reconstruction_rows(
                _read_reconstruction_rows(canonical_summary_path),
                context="ten-iteration sensitivity reproduction check",
                warn=False,
            )
            row_diagnostics.extend(canonical_diagnostics)
            canonical_lookup = {
                (int(row["rank"]), int(row["seed"])): row
                for row in canonical_rows
                if row.get("method") == "cp" and row.get("status") == "completed"
            }
            reproduction_tolerance = float(
                recon_cfg.get("residual_reproduction_tolerance", tolerance)
            )
            for row in normalized_final:
                if row.get("status") != "completed" or int(row["iteration_budget"]) != 10:
                    continue
                key = (int(row["rank"]), int(row["seed"]))
                canonical = canonical_lookup.get(key)
                if canonical is None:
                    canonical_ten_iteration_reproduction.append(
                        {
                            "rank": key[0],
                            "seed": key[1],
                            "available": False,
                            "within_tolerance": False,
                            "reason": "matching completed canonical reconstruction row is missing",
                        }
                    )
                    continue
                difference = abs(
                    float(row["actual_relative_squared_frobenius_error"])
                    - float(canonical["actual_relative_squared_frobenius_error"])
                )
                canonical_ten_iteration_reproduction.append(
                    {
                        "rank": key[0],
                        "seed": key[1],
                        "available": True,
                        "sensitivity_squared_residual": float(
                            row["actual_relative_squared_frobenius_error"]
                        ),
                        "canonical_squared_residual": float(
                            canonical["actual_relative_squared_frobenius_error"]
                        ),
                        "absolute_squared_residual_difference": difference,
                        "tolerance": reproduction_tolerance,
                        "within_tolerance": difference <= reproduction_tolerance,
                    }
                )
    initialization_groups = {}
    for row in normalized_final:
        if row.get("status") != "completed":
            continue
        key = (int(row["rank"]), int(row["seed"]))
        initialization_groups.setdefault(key, set()).add(row["initialization_hash_sha256"])
    initialization_verification = [
        {
            "rank": rank,
            "seed": seed,
            "observed_hashes": sorted(hashes),
            "identical_across_completed_budgets": len(hashes) == 1,
        }
        for (rank, seed), hashes in sorted(initialization_groups.items())
    ]
    inconsistent = [
        item for item in initialization_verification if not item["identical_across_completed_budgets"]
    ]
    if inconsistent:
        raise RuntimeError(f"Initialization identity verification failed: {inconsistent}")

    figure_outputs = []
    figure_generation_failures = []
    try:
        figure_outputs, figure_diagnostics = write_cp_iteration_sensitivity_figure(
            all_rows, _paper_figures_dir(config, root)
        )
        row_diagnostics.extend(figure_diagnostics)
    except Exception as exc:
        figure_generation_failures.append(
            {"figure": "cp_iteration_sensitivity", "exception": repr(exc), "nonfatal": True}
        )

    audit_path = resolve_path(
        sensitivity_cfg.get(
            "audit_path", "results/paper/cp_iteration_sensitivity_audit.md"
        ),
        root,
    )
    audit_outputs = []
    audit_generation_failures = []
    try:
        written_audit, audit_complete = write_cp_iteration_sensitivity_audit(
            aggregates,
            audit_path,
            canonical_ranks=canonical_ranks,
            canonical_seeds=canonical_seeds,
            canonical_iteration_budgets=canonical_budgets,
            failures=failures,
            rank_ordering=rank_ordering,
            canonical_ten_iteration_reproduction=canonical_ten_iteration_reproduction,
        )
        audit_outputs.append(written_audit)
    except Exception as exc:
        audit_complete = False
        audit_generation_failures.append(
            {"artifact": "cp_iteration_sensitivity_audit", "exception": repr(exc), "nonfatal": True}
        )
    metadata_path = save_manifest(
        output_dir / "cp_iteration_sensitivity_metadata.json",
        {
            "kind": "cp_iteration_budget_sensitivity",
            "experiment_id": config.get("experiment_id"),
            "canonical_ranks": canonical_ranks,
            "canonical_cp_seeds": canonical_seeds,
            "canonical_iteration_budgets": canonical_budgets,
            "execution_ranks": ranks,
            "execution_cp_seeds": seeds,
            "execution_iteration_budgets": iteration_budgets,
            "cp_initializer": init,
            "independent_initialization_protocol": (
                "For every rank/seed/budget, hash an independently constructed zero-iteration CP "
                "initialization, then reset the same seed immediately before an independent fitted run. "
                "No warm start is used."
            ),
            "initialization_verification": initialization_verification,
            "initialization_diagnostics_from_execution": initialization_diagnostics,
            "memory_efficient_mttkrp": memory_efficient_mttkrp,
            "mttkrp_rank_chunk_size": mttkrp_rank_chunk_size,
            "mttkrp_max_explicit_bytes": mttkrp_max_explicit_bytes,
            "numerical_tolerance": tolerance,
            "device_requested_and_used_for_cp_fitting": device.type,
            "dense_checkpoint": (
                "synthetic" if sensitivity_cfg.get("synthetic_tensor_shape") else str(dense_checkpoint_path(config, root))
            ),
            "dense_checkpoint_sha256": checkpoint_hash,
            "target_layer": model_config(config).get("target_layer", "classifier.0"),
            "spectral_reference_source": reference_source,
            "diagnostic": diagnostic,
            "rank_aggregates": aggregates,
            "rank_ordering_diagnostic": rank_ordering,
            "canonical_ten_iteration_reproduction": canonical_ten_iteration_reproduction,
            "software_versions": _software_versions(),
            "failures": failures,
            "row_normalization_diagnostics": row_diagnostics,
            "figure_outputs": figure_outputs,
            "figure_generation_failures": figure_generation_failures,
            "audit_outputs": audit_outputs,
            "audit_complete": audit_complete,
            "audit_generation_failures": audit_generation_failures,
            "nonfatal_artifact_policy": (
                "Every computation row is saved before aggregation, figure generation, or audit generation. "
                "Figure and audit failures do not erase completed rows."
            ),
            "convergence_metadata_note": (
                "Rows record completed requested budgets. Tensorly-torch 0.5 does not expose a certified "
                "iteration history through FactorizedConv.from_conv, so no convergence claim is made."
            ),
            "peak_process_resident_memory_raw_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "summary_path": str(summary_path),
            "rank_summary_path": str(rank_summary_path),
        },
        device=device,
    )
    if conv is not None:
        del conv
    gc.collect()
    return metadata_path


def _reconstruction_lookup(rows):
    normalized, _, _ = normalize_reconstruction_rows(
        rows,
        context="structural zero-shot reconstruction lookup",
    )
    lookup = {}
    for row in normalized:
        if row.get("status") != "completed":
            continue
        seed = "" if row["seed"] == "" else str(int(row["seed"]))
        lookup[(row["method"], str(int(row["rank"])), seed)] = row
    return lookup


def _structural_eval_rows(raw_rows, method, rank, seed, common):
    output = []
    for row in raw_rows:
        output.append(
            {
                "method": method,
                "rank": rank,
                "seed": seed,
                "split": row["split"],
                "status": "completed",
                "present_class_miou": row["mean_iou_present_classes"],
                "all_class_miou": row["mean_iou_all_classes"],
                "pixel_accuracy": row["pixel_accuracy"],
                "frequency_weighted_iou": row["frequency_weighted_iou"],
                **common,
            }
        )
    return output


def run_structural_zero_shot(config, root, device, max_batches=None):
    '''Evaluate dense, fitted CP, and matrix-SVD models under the validated mask policy.'''
    stage = "structural_zero_shot"
    output_dir = prepare_run_outputs(config, root, stage)
    reconstruction_path = output_dir / "reconstruction_summary.csv"
    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            f"{reconstruction_path} is required; complete --stage reconstruction first"
        )
    reconstruction_rows = _read_reconstruction_rows(reconstruction_path)
    lookup = _reconstruction_lookup(reconstruction_rows)
    recon_cfg = reconstruction_config(config)
    canonical_ranks = [int(value) for value in recon_cfg.get("ranks", cp_config(config).get("ranks", []))]
    canonical_seeds = [int(value) for value in recon_cfg.get("seeds", [0, 1, 2])]
    ranks = [int(value) for value in recon_cfg.get("execution_ranks", canonical_ranks)]
    seeds = [int(value) for value in recon_cfg.get("execution_seeds", canonical_seeds)]
    tolerance = float(recon_cfg.get("residual_reproduction_tolerance", 1e-5))
    init = recon_cfg.get("init", cp_config(config).get("init", "random"))
    n_iter_max = int(recon_cfg.get("n_iter_max", cp_config(config).get("n_iter_max", 10)))
    memory_efficient_mttkrp = bool(recon_cfg.get("memory_efficient_mttkrp", True))
    mttkrp_rank_chunk_size = int(recon_cfg.get("mttkrp_rank_chunk_size", 64))
    mttkrp_max_explicit_bytes = int(recon_cfg.get("mttkrp_max_explicit_bytes", 512 * 1024**2))
    summary_path = output_dir / "structural_zero_shot_summary.csv"
    validation_report = output_dir / "dataset_validation_report.json"
    if not validation_report.is_file():
        raise FileNotFoundError(f"Validated dataset report is missing: {validation_report}")
    with open(validation_report) as handle:
        validation_payload = json.load(handle)
    expected_unknown_pixels = dataset_config(config).get("unknown_color_resolution", {}).get("affected_pixel_count")
    if validation_payload.get("status") != "pass":
        raise RuntimeError(f"Dataset validation report is not passing: {validation_report}")
    if expected_unknown_pixels is not None and int(validation_payload.get("unknown_pixels_mapped_to_ignore", -1)) != int(expected_unknown_pixels):
        raise RuntimeError(
            "Dataset validation report does not match the configured approved unknown-pixel count: "
            f"{validation_payload.get('unknown_pixels_mapped_to_ignore')} != {expected_unknown_pixels}"
        )
    if validation_payload.get("unknown_color_ignore_index") != dataset_config(config).get("ignore_index"):
        raise RuntimeError("Dataset validation report does not map unknown RGB pixels to the configured ignore index")
    validation_report_reference = str(validation_report)
    validation_report_sha256 = sha256_file(validation_report)
    checkpoint = dense_checkpoint_path(config, root)
    checkpoint_sha256 = sha256_file(checkpoint)
    parameter_ref = parameter_reference(config, root)
    target_layer = parameter_ref["target_layer"]
    persisted_evaluations = {}
    if summary_path.is_file():
        with open(summary_path, newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("method", ""), row.get("rank", ""), row.get("seed", ""), row.get("split", ""))
                persisted_evaluations[key] = row

    def persist_evaluations(new_rows):
        for row in new_rows:
            key = tuple(str(row.get(field, "")) for field in ("method", "rank", "seed", "split"))
            persisted_evaluations[key] = row
        write_csv(summary_path, persisted_evaluations.values())

    failures = []

    dense_model = load_dense_model(config, root)
    dense_conv = get_module(dense_model, target_layer)
    weight = dense_conv.weight.detach().cpu()
    shape = tuple(weight.shape)
    dense_target_parameters = dense_conv_parameter_count(shape, dense_conv.bias is not None)
    h, w = image_size(config)
    input_size = tuple(profile_config(config).get("input_size", [1, 3, h, w]))
    dense_full_macs, dense_records = manual_macs(dense_model.to(device), input_size, device=device)
    target_records = [record for record in dense_records if record["name"] == target_layer]
    if len(target_records) != 1:
        raise RuntimeError(f"Expected one MAC record for {target_layer}, found {len(target_records)}")
    target_output_shape = target_records[0]["output_shape"]
    output_spatial_elements = int(target_output_shape[0] * target_output_shape[2] * target_output_shape[3])
    dense_target_macs = representation_macs(shape, None, "dense", output_spatial_elements)
    dense_eval, _, skipped = evaluate_splits(
        dense_model,
        config,
        root,
        device,
        label="dense_structural_reference",
        rank="",
        max_batches=max_batches,
        model_kind="dense",
        checkpoint=str(checkpoint),
        parameter_ref=parameter_ref,
    )
    dense_common = {
        "target_layer_parameter_count": dense_target_parameters,
        "dense_target_layer_parameter_count": dense_target_parameters,
        "full_model_parameter_count": parameter_ref["dense_total_parameters"],
        "target_layer_macs": dense_target_macs,
        "full_model_macs": dense_full_macs,
        "actual_relative_squared_frobenius_error": 0.0,
        "actual_relative_frobenius_error": 0.0,
        "max_unfolding_tail_bound_squared": 0.0,
        "output_mode_tail_bound_squared": 0.0,
        "dataset_validation_report_reference": validation_report_reference,
        "dataset_validation_report_sha256": validation_report_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_kind": "dense",
    }
    persist_evaluations(_structural_eval_rows(dense_eval, "dense", "", "", dense_common))
    del dense_model
    gc.collect()

    u, s, vh, _, _ = output_mode_svd(weight, device=device)
    for rank in ranks:
        key = ("matrix_svd_output_unfolding", str(rank), "")
        source = lookup.get(key)
        if source is None:
            failures.append({"method": key[0], "rank": rank, "seed": "", "exception": "missing successful reconstruction row"})
            continue
        try:
            model = load_dense_model(config, root)
            conv = get_module(model, target_layer)
            replacement = MatrixLowRankConv2d.from_svd(conv, rank, u, s, vh)
            set_module(model, target_layer, replacement)
            raw, _, method_skips = evaluate_splits(
                model, config, root, device, f"matrix_svd_rank_{rank}_zero_shot", rank,
                max_batches=max_batches, model_kind="matrix_svd_output_unfolding_zero_shot",
                checkpoint=str(checkpoint), parameter_ref=parameter_ref,
            )
            skipped.extend(method_skips)
            target_parameters = matrix_svd_conv_parameter_count(shape, rank, conv.bias is not None)
            target_macs = representation_macs(shape, rank, "matrix_svd_output_unfolding", output_spatial_elements)
            common = {
                "target_layer_parameter_count": target_parameters,
                "dense_target_layer_parameter_count": dense_target_parameters,
                "full_model_parameter_count": parameter_ref["dense_total_parameters"] - dense_target_parameters + target_parameters,
                "target_layer_macs": target_macs,
                "full_model_macs": dense_full_macs - dense_target_macs + target_macs,
                "actual_relative_squared_frobenius_error": source["actual_relative_squared_frobenius_error"],
                "actual_relative_frobenius_error": source["actual_relative_frobenius_error"],
                "max_unfolding_tail_bound_squared": source["max_unfolding_tail_bound_squared"],
                "output_mode_tail_bound_squared": source["output_mode_tail_bound_squared"],
                "dataset_validation_report_reference": validation_report_reference,
                "dataset_validation_report_sha256": validation_report_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "model_kind": "matrix_svd_output_unfolding_zero_shot",
            }
            persist_evaluations(_structural_eval_rows(raw, key[0], rank, "", common))
            del model, replacement
        except Exception as exc:
            failures.append({"method": key[0], "rank": rank, "seed": "", "exception": repr(exc)})
        gc.collect()
    del u, s, vh
    gc.collect()

    for rank in ranks:
        for seed in seeds:
            key = ("cp", str(rank), str(seed))
            source = lookup.get(key)
            if source is None:
                failures.append({"method": "cp", "rank": rank, "seed": seed, "exception": "missing successful reconstruction row"})
                continue
            try:
                set_seed(seed, deterministic=True)
                model = load_dense_model(config, root)
                conv = get_module(model, target_layer)
                fitted, approximation, _, _ = fit_cp_approximation(
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
                expected = float(source["actual_relative_squared_frobenius_error"])
                if abs(squared - expected) > tolerance:
                    raise AssertionError(f"Re-fitted CP residual {squared} differs from reconstruction-stage value {expected}")
                set_module(model, target_layer, fitted)
                raw, _, method_skips = evaluate_splits(
                    model, config, root, device, f"cp_rank_{rank}_seed_{seed}_structural_zero_shot", rank,
                    max_batches=max_batches, model_kind="cp_structural_zero_shot",
                    checkpoint=str(checkpoint), parameter_ref=parameter_ref,
                )
                skipped.extend(method_skips)
                target_parameters = cp_conv_parameter_count(shape, rank, conv.bias is not None)
                target_macs = representation_macs(shape, rank, "cp", output_spatial_elements)
                common = {
                    "target_layer_parameter_count": target_parameters,
                    "dense_target_layer_parameter_count": dense_target_parameters,
                    "full_model_parameter_count": parameter_ref["dense_total_parameters"] - dense_target_parameters + target_parameters,
                    "target_layer_macs": target_macs,
                    "full_model_macs": dense_full_macs - dense_target_macs + target_macs,
                    "actual_relative_squared_frobenius_error": squared,
                    "actual_relative_frobenius_error": ordinary,
                    "max_unfolding_tail_bound_squared": source["max_unfolding_tail_bound_squared"],
                    "output_mode_tail_bound_squared": source["output_mode_tail_bound_squared"],
                    "dataset_validation_report_reference": validation_report_reference,
                    "dataset_validation_report_sha256": validation_report_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_kind": "cp_structural_zero_shot",
                }
                persist_evaluations(_structural_eval_rows(raw, "cp", rank, seed, common))
                del model, fitted, approximation
            except Exception as exc:
                failures.append({"method": "cp", "rank": rank, "seed": seed, "exception": repr(exc)})
            gc.collect()

    all_rows = list(persisted_evaluations.values())
    row_normalization_diagnostics = []
    tradeoff_rows = join_structural_tradeoffs(
        reconstruction_rows,
        all_rows,
        diagnostics=row_normalization_diagnostics,
    )
    tradeoff_path = output_dir / "structural_tradeoff_summary.csv"
    write_csv(tradeoff_path, tradeoff_rows)
    correlations = exploratory_correlations(tradeoff_rows)
    correlation_path = save_manifest(
        output_dir / "structural_tradeoff_correlations.json",
        {"kind": "exploratory_structural_correlations", **correlations},
        device=device,
    )
    figure_outputs = []
    figure_generation_failures = []
    try:
        figure_outputs = write_zero_shot_figure(
            all_rows,
            _paper_figures_dir(config, root),
            split="test",
            diagnostics=row_normalization_diagnostics,
        )
    except Exception as exc:
        figure_generation_failures.append(
            {"figure": "zero_shot_present_class_miou", "exception": repr(exc), "nonfatal": True}
        )
    return save_manifest(
        output_dir / "structural_zero_shot_metadata.json",
        {
            "kind": "structural_zero_shot_evaluation",
            "experiment_id": config.get("experiment_id"),
            "canonical_ranks": canonical_ranks,
            "canonical_cp_seeds": canonical_seeds,
            "execution_ranks": ranks,
            "execution_cp_seeds": seeds,
            "dataset_validation_report_reference": validation_report_reference,
            "dataset_validation_report_sha256": validation_report_sha256,
            "dense_checkpoint": str(checkpoint),
            "dense_checkpoint_sha256": checkpoint_sha256,
            "reconstruction_summary_reference": str(reconstruction_path),
            "tradeoff_summary_reference": str(tradeoff_path),
            "exploratory_correlations_reference": str(correlation_path),
            "figure_outputs": figure_outputs,
            "row_normalization_diagnostics": row_normalization_diagnostics,
            "figure_generation_failures": figure_generation_failures,
            "failures": failures,
            "skipped_splits": skipped,
            "software_versions": _software_versions(),
            "peak_process_resident_memory_raw_ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        device=device,
    )
