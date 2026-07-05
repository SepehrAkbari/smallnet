'''
Canonical CamVid/VGG/CP experiment pipeline.
'''

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from src.smallnet.config import ensure_dir
from src.smallnet.data import camvid_split_available, load_camvid_class_names, make_camvid_loader
from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.evaluation import evaluate_segmentation
from src.smallnet.factorization import build_factorized_model_from_dense
from src.smallnet.models import build_vgg16_fcn32s
from src.smallnet.modules import count_parameters
from src.smallnet.profiling import latency_ms, manual_macs
from src.smallnet.reproducibility import set_seed
from src.smallnet.results import save_config_snapshot, save_manifest, write_csv


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


def loader_for_split(config, root, split, shuffle=False):
    train_cfg = training_config(config)
    eval_cfg = eval_config(config)
    return make_camvid_loader(
        data_root(config, root),
        split=split,
        batch_size=train_cfg.get("batch_size", eval_cfg.get("batch_size", 4)),
        num_workers=train_cfg.get("num_workers", eval_cfg.get("num_workers", 2)),
        image_size=image_size(config),
        class_dict_path=class_dict_path(config, root),
        shuffle=shuffle,
    )


def evaluate_splits(model, config, root, device, label, rank, max_batches=None, splits=None):
    names = load_camvid_class_names(class_dict_path(config, root))
    ignore_index, ignore_name = ignore_index_and_name(config)
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
                "parameters": count_parameters(model),
                "ignore_index": "" if ignore_index is None else ignore_index,
                "ignore_class": ignore_name or "",
            }
        )
        rows.append(
            {
                "label": label,
                "rank": rank,
                "split": split,
                "parameters": count_parameters(model),
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
    rows, evaluations, skipped = evaluate_splits(
        model,
        config,
        root,
        device,
        label="dense",
        rank="dense",
        max_batches=max_batches,
    )
    write_csv(output_dir / f"{stage}_summary.csv", rows)
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "dense_baseline_evaluation",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "dense_checkpoint": str(dense_checkpoint_path(config, root)),
            "evaluations": evaluations,
            "skipped_splits": skipped,
        },
        device=device,
    )


def iter_rank_seed(config):
    cp_cfg = cp_config(config)
    ranks = cp_cfg.get("ranks", [64, 128, 256])
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
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_checkpoints = bool(cp_config(config).get("save_zero_shot_checkpoints", False))

    for rank, seed in iter_rank_seed(config):
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
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_checkpoints = bool(cp_config(config).get("save_finetuned_checkpoints", False))
    ignore_index, _ = ignore_index_and_name(config)

    train_cfg = training_config(config)
    epochs = int(train_cfg.get("fine_tuning_epochs", 0))
    if epochs <= 0:
        raise ValueError("training.fine_tuning_epochs must be positive for cp_finetune")

    for rank, seed in iter_rank_seed(config):
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
            "training_records": training_records,
            "evaluations": all_evaluations,
            "skipped_splits": skipped_splits,
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

    models = [("dense", "dense", load_dense_model(config, root))]
    for rank, seed in iter_rank_seed(config):
        set_seed(seed, deterministic=training_config(config).get("deterministic", True))
        models.append((f"cp_rank_{rank}_seed_{seed}_zero_shot", rank, make_cp_model(config, root, rank)))

    for label, rank, model in models:
        model = model.to(device)
        macs, records = manual_macs(model, input_size, device=device)
        latency = None
        if run_latency:
            latency = latency_ms(
                model,
                input_size,
                device,
                warmup=int(prof_cfg.get("warmup", 10)),
                iterations=int(prof_cfg.get("iterations", 50)),
            )
        row = {
            "label": label,
            "rank": rank,
            "parameters": count_parameters(model),
            "macs": macs,
            "latency_ms": latency,
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
    return save_manifest(
        output_dir / f"{stage}_metadata.json",
        {
            "kind": "rank_energy_diagnostics",
            "experiment_id": config.get("experiment_id"),
            "config": config,
            "dense_checkpoint": str(dense_checkpoint_path(config, root)),
            "diagnostics": diagnostics,
        },
        device=device,
    )
