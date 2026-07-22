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
    run_cp_iteration_sensitivity,
    run_cp_zero_shot,
    run_dataset_validation,
    run_dense_evaluation,
    run_existing_finetuned_evaluation,
    run_profiling,
    run_rank_diagnostics,
    run_reconstruction,
    run_reconstruction_figures,
    run_structural_zero_shot,
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
    "reconstruction": [run_reconstruction],
    "reconstruction-figures": [run_reconstruction_figures],
    "cp-iteration-sensitivity": [run_cp_iteration_sensitivity],
    "structural-zero-shot": [run_structural_zero_shot],
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
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run reconstruction on a small deterministic synthetic Conv2d without CamVid or a checkpoint.",
    )
    parser.add_argument("--ranks", nargs="+", type=int, help="Optional reconstruction/structural rank subset.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional reconstruction/structural CP seed subset.")
    parser.add_argument(
        "--iteration-budgets",
        nargs="+",
        type=int,
        help="Optional CP iteration-budget subset for the sensitivity stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    subset_section = (
        "cp_iteration_sensitivity"
        if args.stage == "cp-iteration-sensitivity"
        else "reconstruction"
    )
    if args.ranks:
        config.setdefault(subset_section, {})["execution_ranks"] = args.ranks
    if args.seeds:
        config.setdefault(subset_section, {})["execution_seeds"] = args.seeds
    if args.iteration_budgets:
        if args.stage != "cp-iteration-sensitivity":
            raise ValueError("--iteration-budgets is supported only with --stage cp-iteration-sensitivity")
        config.setdefault("cp_iteration_sensitivity", {})[
            "execution_iteration_budgets"
        ] = args.iteration_budgets
    if args.synthetic_smoke:
        if args.stage not in {"reconstruction", "cp-iteration-sensitivity"}:
            raise ValueError(
                "--synthetic-smoke is supported only with --stage reconstruction or "
                "cp-iteration-sensitivity"
            )
        if args.stage == "reconstruction":
            if not args.output_dir:
                config["output_dir"] = "results/camvid_vgg_cp/synthetic_reconstruction_smoke"
            config.setdefault("reconstruction", {}).update(
                {
                    "synthetic_tensor_shape": [8, 6, 3, 3],
                    "synthetic_seed": 123,
                    "ranks": [1, 2, 4],
                    "seeds": [0, 1, 2],
                }
            )
        else:
            if not args.output_dir:
                config["output_dir"] = (
                    "results/camvid_vgg_cp/synthetic_cp_iteration_sensitivity_smoke"
                )
            config.setdefault("cp_iteration_sensitivity", {}).update(
                {
                    "synthetic_tensor_shape": [8, 6, 3, 3],
                    "synthetic_seed": 123,
                    "ranks": [1, 2, 4],
                    "seeds": [0, 1, 2],
                    "iteration_budgets": [1, 2, 3, 4],
                    "audit_path": f"{config['output_dir']}/cp_iteration_sensitivity_audit.md",
                }
            )
        config.setdefault("paper", {})["figures_dir"] = f"{config['output_dir']}/figures"
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
