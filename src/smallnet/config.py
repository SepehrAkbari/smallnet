'''
JSON configuration helpers for reproducible experiment runners.
'''

import json
from pathlib import Path


def load_config(path):
    path = Path(path)
    with open(path) as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    return config


def require_keys(mapping, keys, context="config"):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing required keys in {context}: {missing}")


def repo_path(path, root=None):
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(root) if root is not None else Path.cwd()
    return root / path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
