"""Run CamVid/VGG rank-energy diagnostics on Colab."""

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
    "scripts/run_spl_camvid.py",
    "--config",
    "configs/spl/camvid_vgg.json",
    "--device",
    "cuda",
    "--skip-eval",
    "--skip-profile",
    "--output-dir",
    "res/spl_ready/camvid_vgg_rank",
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
