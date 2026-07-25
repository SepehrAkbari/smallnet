#!/usr/bin/env python3
"""Validate final CP-only paper artifacts without data, CUDA, or checkpoint."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smallnet.final_cp_paper import validate_paper_artifacts


if __name__ == "__main__":
    result = validate_paper_artifacts(ROOT)
    print(
        "Final CP paper artifacts valid: "
        f"{result['canonical_cp_rows']} rows, "
        f"{result['rank_aggregates']} rank aggregates, "
        f"{result['paper_artifacts_checked']} paper artifacts, "
        f"{result['manifest_entries_checked']} manifest entries."
    )
