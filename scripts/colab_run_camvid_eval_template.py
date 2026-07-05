"""Shared helper for one-checkpoint CamVid/VGG evaluation on Colab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")


def run_label(label: str) -> None:
    os.chdir(PROJECT)
    output_dir = f"res/spl_ready/camvid_vgg_{label}"
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_spl_camvid.py",
        "--config",
        "configs/spl/camvid_vgg.json",
        "--device",
        "cuda",
        "--num-workers",
        "2",
        "--skip-profile",
        "--skip-rank",
        "--labels",
        label,
        "--output-dir",
        output_dir,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
