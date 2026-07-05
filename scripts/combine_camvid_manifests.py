"""Combine per-label CamVid manifests into canonical SPL-ready manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smallnet.results import load_manifest, save_manifest


LABEL_DIRS = [
    "dense",
    "cp_rank_256",
    "cp_rank_128",
    "cp_rank_64",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="res/spl_ready")
    parser.add_argument("--eval-output", default="res/spl_ready/camvid_vgg_eval_manifest_combined.json")
    parser.add_argument("--profile-output", default="res/spl_ready/camvid_vgg_profile_manifest_combined.json")
    args = parser.parse_args()

    root = Path(args.root)
    evaluations = []
    profiles = []

    for label in LABEL_DIRS:
        eval_manifest = load_manifest(root / f"camvid_vgg_{label}" / "camvid_eval_manifest.json")
        evaluations.extend(eval_manifest.get("evaluations", []))

        profile_manifest = load_manifest(root / f"camvid_vgg_profile_{label}" / "camvid_profile_manifest.json")
        profiles.extend(profile_manifest.get("profiles", []))

    save_manifest(
        args.eval_output,
        {
            "experiment_id": "spl_camvid_vgg_rank_energy_combined",
            "kind": "camvid_evaluation_combined",
            "source_labels": LABEL_DIRS,
            "evaluations": evaluations,
        },
    )
    save_manifest(
        args.profile_output,
        {
            "experiment_id": "spl_camvid_vgg_rank_energy_combined",
            "kind": "camvid_profile_combined",
            "source_labels": LABEL_DIRS,
            "profiles": profiles,
        },
    )


if __name__ == "__main__":
    main()
