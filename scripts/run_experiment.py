'''
Run the reproducible CamVid/VGG16-FCN32s CP diagnostic pipeline.
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
from src.smallnet.experiment import (
    run_cp_finetune,
    run_cp_zero_shot,
    run_dataset_validation,
    run_dense_evaluation,
    run_existing_finetuned_evaluation,
    run_profiling,
    run_rank_diagnostics,
)
from src.smallnet.reproducibility import auto_device


STAGES = {
    "validate-data": [run_dataset_validation],
    "dense": [run_dense_evaluation],
    "zero-shot": [run_cp_zero_shot],
    "eval-finetuned": [run_existing_finetuned_evaluation],
    "finetune": [run_cp_finetune],
    "profile": [run_profiling],
    "rank": [run_rank_diagnostics],
    "full": [
        run_dense_evaluation,
        run_cp_zero_shot,
        run_existing_finetuned_evaluation,
        run_cp_finetune,
        run_profiling,
        run_rank_diagnostics,
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camvid_vgg_cp.json")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--device", default=None, help="Override config device, e.g. cpu, cuda, mps.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit for evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    device = auto_device(args.device or config.get("device"))

    outputs = []
    for runner in STAGES[args.stage]:
        if runner is run_rank_diagnostics:
            outputs.append(runner(config, ROOT, device=device))
        elif runner is run_profiling:
            outputs.append(runner(config, ROOT, device))
        else:
            outputs.append(runner(config, ROOT, device, max_batches=args.max_batches))

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
