'''
Best-effort reproducibility helpers.
'''

import os
import random

import numpy as np
import torch


def set_seed(seed=None, deterministic=True):
    if seed is None:
        return {"seed": None, "deterministic": False}

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    status = {"seed": seed, "deterministic": bool(deterministic)}
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            status["torch_deterministic_algorithms"] = "warn_only"
        except Exception as exc:
            status["torch_deterministic_algorithms"] = f"unavailable: {exc}"
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return status


def auto_device(config_device=None):
    if config_device:
        return torch.device(config_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"
