"""Run the canonical CamVid/VGG SPL experiment on the Colab VM."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")
os.chdir(PROJECT)
print(f"cwd={Path.cwd()}")
cmd = [
    sys.executable,
    "scripts/run_spl_camvid.py",
    "--config",
    "configs/spl/camvid_vgg.json",
    "--device",
    "cuda",
    "--num-workers",
    "2",
]
print("running:", " ".join(cmd))
proc = subprocess.run(cmd, text=True, capture_output=True)
print("returncode:", proc.returncode)
print("stdout:")
print(proc.stdout)
print("stderr:")
print(proc.stderr)
proc.check_returncode()
