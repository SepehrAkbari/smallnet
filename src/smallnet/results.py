'''
Result manifest and paper table helpers.
'''

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.smallnet.reproducibility import device_name


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)}")


def environment_metadata(device=None):
    device = torch.device(device) if device is not None else None
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if device is not None:
        metadata["device"] = device.type
        metadata["device_name"] = device_name(device)
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        metadata["git_commit"] = ""
        metadata["git_dirty"] = None
    return metadata


def save_manifest(path, payload, device=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "smallnet.manifest.v1",
        "environment": environment_metadata(device=device),
        **payload,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=json_default)
    return path


def save_config_snapshot(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=json_default)
    return path


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def flatten_eval_summary(label, split, summary):
    row = {
        "label": label,
        "split": split,
        "pixel_accuracy": summary.get("pixel_accuracy"),
        "mean_iou_all_classes": summary.get("mean_iou_all_classes"),
        "mean_iou_present_classes": summary.get("mean_iou_present_classes"),
        "frequency_weighted_iou": summary.get("frequency_weighted_iou"),
    }
    for key in ["rank", "parameters", "macs", "latency_ms", "checkpoint", "ignore_class"]:
        if key in summary:
            row[key] = summary[key]
    return row
