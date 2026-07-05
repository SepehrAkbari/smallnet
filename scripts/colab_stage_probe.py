"""Inspect how colab-cli stages scripts on the remote VM."""

from __future__ import annotations

import json
import os
from pathlib import Path


cwd = Path.cwd()
payload = {
    "cwd": str(cwd),
    "script_file": globals().get("__file__"),
    "cwd_listing": sorted(p.name for p in cwd.iterdir())[:50],
    "parent_listing": sorted(p.name for p in cwd.parent.iterdir())[:50],
    "env_subset": {
        key: os.environ.get(key)
        for key in ["HOME", "PYTHONPATH", "COLAB_GPU", "CUDA_VISIBLE_DEVICES"]
        if os.environ.get(key) is not None
    },
}

print(json.dumps(payload, indent=2))
