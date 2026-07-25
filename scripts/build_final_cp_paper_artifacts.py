#!/usr/bin/env python3
"""Build final CP-only paper artifacts from completed canonical rows."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smallnet.final_cp_paper import build_final_cp_artifacts


if __name__ == "__main__":
    result = build_final_cp_artifacts(ROOT)
    print(
        f"Built final CP-only artifacts for "
        f"{result['validation']['canonical_cp_row_count']} rows."
    )
