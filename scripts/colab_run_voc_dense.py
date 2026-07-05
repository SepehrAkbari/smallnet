"""Run VOC/DeepLabV3 dense baseline, rank diagnostics, and profile on Colab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")
os.chdir(PROJECT)
cmd = [
    sys.executable,
    "-u",
    "scripts/run_spl_voc.py",
    "--config",
    "configs/spl/voc_deeplab.json",
    "--device",
    "cuda",
    "--download",
    "--dense-only",
    "--output-dir",
    "res/spl_ready/voc_deeplab_dense",
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
