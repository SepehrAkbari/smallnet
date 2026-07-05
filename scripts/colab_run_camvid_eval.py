"""Run the CamVid/VGG evaluation phase on the Colab VM."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")
os.chdir(PROJECT)
cmd = [
    sys.executable,
    "scripts/run_spl_camvid.py",
    "--config",
    "configs/spl/camvid_vgg.json",
    "--device",
    "cuda",
    "--num-workers",
    "2",
    "--skip-profile",
    "--skip-rank",
]
print("running:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
