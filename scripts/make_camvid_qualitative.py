'''
Generate a CamVid qualitative panel with input, ground truth, dense, and compressed predictions.
'''

import argparse
import csv
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
from PIL import Image
from torchvision import transforms

from src.dataset import CamVidDataset
from src.smallnet.config import ensure_dir, load_config
from src.smallnet.models import load_vgg16_fcn32s_checkpoint


def load_colors(class_dict_path):
    names = []
    colors = []
    with open(class_dict_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names.append(row[0])
            colors.append([int(value.strip()) for value in row[1:4]])
    return names, np.asarray(colors, dtype=np.uint8)


def decode(mask, colors):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        rgb[mask == idx] = color
    return rgb


def label_name_for_image(image_name):
    stem = Path(image_name).stem
    return f"{stem}_L.png"


@torch.no_grad()
def predict(model, image_tensor, device):
    model = model.to(device)
    model.eval()
    output = model(image_tensor.to(device))
    return output.argmax(dim=1).squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spl/paper_assets.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    qualitative = config["qualitative"]
    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    data_root = Path(qualitative["data_root"])
    split = qualitative.get("split", "val")
    image_name = qualitative["image"]
    image_size = tuple(qualitative.get("image_size", [352, 480]))
    _, colors = load_colors(data_root / "class_dict.csv")

    image = Image.open(data_root / split / image_name).convert("RGB")
    raw_resized = image.resize((image_size[1], image_size[0]))
    label = Image.open(data_root / f"{split}_labels" / label_name_for_image(image_name)).convert("RGB")
    dataset = CamVidDataset(
        data_root / split,
        data_root / f"{split}_labels",
        data_root / "class_dict.csv",
        mask_suffix_to_remove="_L",
    )
    gt = dataset._rgb_to_index(np.array(label))
    gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize((image_size[1], image_size[0]), resample=Image.Resampling.NEAREST))

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    panels = [("Input", np.array(raw_resized)), ("Ground Truth", decode(gt, colors))]
    for model_cfg in qualitative["models"]:
        model, _ = load_vgg16_fcn32s_checkpoint(model_cfg["checkpoint"], num_classes=len(colors))
        pred = predict(model, input_tensor, device)
        panels.append((model_cfg["label"], decode(pred, colors)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()

    out_dir = ensure_dir(config["output_dir"])
    out_path = out_dir / "figure_3_qualitative.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
