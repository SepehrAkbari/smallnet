"""Reassemble and unpack the smallnet Colab experiment bundle."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path


PARTS_DIR = Path("/content/smallnet_parts")
BUNDLE = Path("/content/smallnet_spl_bundle_clean.tgz")
TARGET = Path("/content/smallnet")
EXPECTED_SHA256 = "e721e802081f283e9b04784809f3ec2977b6348902cb772bd04201592f663419"

parts = sorted(PARTS_DIR.glob("smallnet_spl_bundle_clean_25m.part.*"))
print(f"parts: {len(parts)}")
if len(parts) != 56:
    raise SystemExit(f"Expected 56 parts, found {len(parts)}")

with BUNDLE.open("wb") as out:
    for part in parts:
        with part.open("rb") as inp:
            shutil.copyfileobj(inp, out)

digest = hashlib.sha256()
with BUNDLE.open("rb") as fh:
    for block in iter(lambda: fh.read(1024 * 1024), b""):
        digest.update(block)
sha = digest.hexdigest()
print(f"sha256: {sha}")
if sha != EXPECTED_SHA256:
    raise SystemExit(f"SHA-256 mismatch: {sha} != {EXPECTED_SHA256}")

if TARGET.exists():
    # Keep this conservative: only clear the known upload target.
    shutil.rmtree(TARGET)
TARGET.mkdir(parents=True, exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    tar.extractall(TARGET)

print("unpacked")
for path in ["src", "scripts", "configs", "data/CamVid", "model"]:
    p = TARGET / path
    print(f"{path}: exists={p.exists()}")
