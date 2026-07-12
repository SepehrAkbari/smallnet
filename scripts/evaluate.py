'''
Evaluate dense or CP-factorized VGG16-FCN32s checkpoints on CamVid.
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import tltorch

from src.dataset import CamVidDataset
from src.model import VGG16_FCN32s
from src.utils import fast_hist, summarize_hist


def load_class_names(class_dict_path):
    names = []
    with open(class_dict_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names.append(row[0])
    return names


def infer_cp_rank(state_dict):
    weights_key = "classifier.0.weight.weights"
    if weights_key in state_dict:
        return int(state_dict[weights_key].numel())
    return None


def build_model(num_classes, rank=None):
    model = VGG16_FCN32s(num_classes=num_classes, pretrained=False)
    if rank is not None:
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0],
            rank=rank,
            factorization="cp",
            decomposition_kwargs={"init": "random", "n_iter_max": 0},
        )
    return model


def load_checkpoint_model(checkpoint_path, num_classes):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    rank = infer_cp_rank(state_dict)
    model = build_model(num_classes=num_classes, rank=rank)
    model.load_state_dict(state_dict)
    return model, rank


def make_camvid_loader(data_root, split, batch_size, num_workers, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data_root = Path(data_root)
    dataset = CamVidDataset(
        data_root / split,
        data_root / f"{split}_labels",
        data_root / "class_dict.csv",
        transform=transform,
        image_size=image_size,
        mask_suffix_to_remove="_L",
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_model(model, loader, num_classes, device, ignore_index=None, max_batches=None):
    model = model.to(device)
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        gts = masks.numpy()
        hist += fast_hist(gts.flatten(), preds.flatten(), num_classes, ignore_index=ignore_index)

    return hist


def write_outputs(summary, hist, out_dir, run_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "confusion_matrix": hist.astype(int).tolist()}, f, indent=2)

    summary_path = out_dir / f"{run_name}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            if key != "per_class":
                writer.writerow([key, value])

    per_class_path = out_dir / f"{run_name}_per_class.csv"
    with open(per_class_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "support", "iou", "excluded"])
        writer.writeheader()
        writer.writerows(summary["per_class"])

    return json_path, summary_path, per_class_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model/best_model.pth")
    parser.add_argument("--data-root", default="data/CamVid")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--ignore-class", default=None, help="Optional class name to ignore, e.g. Void.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--out-dir", default="res/eval")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    class_dict_path = Path(args.data_root) / "class_dict.csv"
    class_names = load_class_names(class_dict_path)
    ignore_index = class_names.index(args.ignore_class) if args.ignore_class else None

    model, rank = load_checkpoint_model(args.checkpoint, num_classes=len(class_names))
    loader = make_camvid_loader(
        args.data_root,
        args.split,
        args.batch_size,
        args.num_workers,
        image_size=(args.height, args.width),
    )
    hist = evaluate_model(
        model,
        loader,
        len(class_names),
        device,
        ignore_index=ignore_index,
        max_batches=args.max_batches,
    )
    summary = summarize_hist(
        hist,
        class_names=class_names,
        exclude_indices=[ignore_index] if ignore_index is not None else None,
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "rank": "dense" if rank is None else rank,
            "ignore_class": args.ignore_class or "",
            "device": device.type,
        }
    )

    run_name = f"{Path(args.checkpoint).stem}_{args.split}"
    if args.ignore_class:
        run_name += f"_ignore_{args.ignore_class}"
    json_path, summary_path, per_class_path = write_outputs(summary, hist, args.out_dir, run_name)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Rank: {'dense' if rank is None else rank}")
    print(f"Pixel Accuracy: {summary['pixel_accuracy']:.4f}")
    print(f"mIoU (all classes): {summary['mean_iou_all_classes']:.4f}")
    print(f"mIoU (present classes): {summary['mean_iou_present_classes']:.4f}")
    print(f"FWIoU: {summary['frequency_weighted_iou']:.4f}")
    print(f"Wrote: {json_path}, {summary_path}, {per_class_path}")


if __name__ == "__main__":
    main()
