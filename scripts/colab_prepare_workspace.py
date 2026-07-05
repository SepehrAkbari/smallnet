"""Prepare directories on the Colab VM for smallnet experiments."""

from __future__ import annotations

from pathlib import Path

for path in [
    Path("/content/smallnet_parts"),
    Path("/content/smallnet"),
    Path("/content/smallnet_outputs"),
]:
    path.mkdir(parents=True, exist_ok=True)
    print(f"ready: {path}")
