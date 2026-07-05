"""Reassemble and unpack the model/code-only Colab experiment bundle."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path


PARTS_DIR = Path("/content/smallnet_parts")
BUNDLE = Path("/content/smallnet_spl_model_bundle.tgz")
TARGET = Path("/content/smallnet")
EXPECTED_SHA256 = "eadd8126ad7b9c9d55ed7bf4c2d5194347d37d408312b3fad0cb32ed99008b93"
EXPECTED_PARTS = 33

parts = sorted(PARTS_DIR.glob("smallnet_spl_model_bundle_25m.part.*"))
print(f"parts: {len(parts)}")
if len(parts) != EXPECTED_PARTS:
    raise SystemExit(f"Expected {EXPECTED_PARTS} parts, found {len(parts)}")

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
    shutil.rmtree(TARGET)
TARGET.mkdir(parents=True, exist_ok=True)
with tarfile.open(BUNDLE, "r:gz") as tar:
    tar.extractall(TARGET)

print("unpacked")
for path in ["src", "scripts", "configs", "model"]:
    p = TARGET / path
    print(f"{path}: exists={p.exists()}")
