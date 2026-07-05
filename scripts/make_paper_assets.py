'''
Generate SPL paper tables and figures from saved manifests.
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

from src.smallnet.assets import plot_pareto, plot_rank_spectrum, write_pareto_table, write_rank_table
from src.smallnet.config import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spl/paper_assets.json")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["output_dir"])
    written = []

    def exists(path):
        ok = Path(path).exists()
        if not ok and not args.skip_missing:
            raise FileNotFoundError(path)
        return ok

    rank_manifest = config.get("rank_manifest")
    if rank_manifest and exists(rank_manifest):
        written.append(write_rank_table(rank_manifest, out_dir / "table_1_rank_diagnostics.csv"))
        written.append(
            plot_rank_spectrum(
                rank_manifest,
                out_dir / "figure_1_rank_spectrum.png",
                layer=config.get("rank_plot_layer"),
                mode=config.get("rank_plot_mode", "0"),
            )
        )

    eval_manifest = config.get("eval_manifest")
    profile_manifest = config.get("profile_manifest")
    if eval_manifest and exists(eval_manifest) and (not profile_manifest or exists(profile_manifest)):
        written.append(write_pareto_table(eval_manifest, profile_manifest, out_dir / "table_2_pareto.csv"))
        written.append(plot_pareto(eval_manifest, profile_manifest, out_dir / "figure_2_pareto.png"))

    print("Wrote assets:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
