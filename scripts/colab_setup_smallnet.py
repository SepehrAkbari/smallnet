"""Install/check dependencies for smallnet SPL experiments on Colab."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path("/content/smallnet")
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "pillow",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "tensorly": "tensorly",
    "tltorch": "tensorly-torch",
    "thop": "thop",
}


def missing_packages() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_name)
    return missing


missing = missing_packages()
print(json.dumps({"missing": missing}, indent=2))
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

sys.path.insert(0, str(PROJECT))
import torch
import torchvision

payload = {
    "project_exists": PROJECT.exists(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "missing_after_install": missing_packages(),
}
print(json.dumps(payload, indent=2))
if payload["missing_after_install"]:
    raise SystemExit("Dependency setup incomplete")
