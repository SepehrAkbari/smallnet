"""Run one targeted VOC/DeepLabV3 factorized-layer stress test on Colab."""

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
    "--factor-layers",
    "classifier.0.convs.1.0",
    "--factor-ranks",
    "128",
    "--output-dir",
    "res/spl_ready/voc_deeplab_factorized_one",
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
