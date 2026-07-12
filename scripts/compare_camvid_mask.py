"""Compare a local CamVid mask with an externally supplied reference mask."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smallnet.mask_forensics import compare_camvid_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--class-dict", default="data/CamVid/class_dict.csv")
    args = parser.parse_args()
    print(json.dumps(compare_camvid_masks(args.current, args.reference, args.class_dict), indent=2))


if __name__ == "__main__":
    main()
