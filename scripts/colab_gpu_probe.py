"""Probe a Colab runtime for GPU and PyTorch environment details."""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - remote diagnostic helper
        return f"ERROR: {exc}"


payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "nvidia_smi": run(["nvidia-smi"]),
}

try:
    import torch

    payload.update(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
except Exception as exc:  # pragma: no cover - remote diagnostic helper
    payload["torch_error"] = repr(exc)

print(json.dumps(payload, indent=2))
