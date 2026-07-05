'''
Build paper-ready tables, figures, and manifest from available experiment outputs.
'''

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smallnet.config import load_config
from src.smallnet.paper import build_paper_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camvid_vgg_cp_paper.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    manifest_path = build_paper_artifacts(config, ROOT)
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
