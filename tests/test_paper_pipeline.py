import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from src.smallnet.diagnostics import rank_energy_diagnostic
from src.smallnet.experiment import (
    parameter_accounting,
    run_existing_finetuned_evaluation,
    run_profiling,
    run_rank_diagnostics,
)
from src.smallnet.paper import build_paper_artifacts, write_rank_energy_artifacts
from src.smallnet.profiling import summarize_latency_timings


class AccountingTests(unittest.TestCase):
    def test_target_layer_parameter_accounting(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 2, 1))
        reference = {
            "target_layer": "0",
            "dense_total_parameters": sum(param.numel() for param in model.parameters()),
            "dense_target_layer_parameters": sum(param.numel() for param in model[0].parameters()),
        }
        accounting = parameter_accounting(model, reference)

        self.assertEqual(accounting["total_parameters"], 122)
        self.assertEqual(accounting["target_layer_parameters"], 112)
        self.assertEqual(accounting["dense_target_layer_parameters"], 112)
        self.assertEqual(accounting["target_layer_compression_ratio"], 1.0)
        self.assertEqual(accounting["total_compression_ratio"], 1.0)


class LatencyStatsTests(unittest.TestCase):
    def test_latency_summary_shape_and_values(self):
        stats = summarize_latency_timings([3.0, 1.0, 2.0], warmup=5, iterations=3, device_name="unit")

        self.assertEqual(stats["latency_mean_ms"], 2.0)
        self.assertAlmostEqual(stats["latency_std_ms"], (2.0 / 3.0) ** 0.5)
        self.assertEqual(stats["latency_median_ms"], 2.0)
        self.assertEqual(stats["latency_min_ms"], 1.0)
        self.assertEqual(stats["latency_max_ms"], 3.0)
        self.assertEqual(stats["latency_warmup_iterations"], 5)
        self.assertEqual(stats["latency_iterations"], 3)
        self.assertEqual(stats["device_name"], "unit")


class ExistingCheckpointTests(unittest.TestCase):
    def test_missing_existing_checkpoint_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "experiment_id": "unit",
                "output_dir": "out",
                "model": {"target_layer": "0", "dense_checkpoint": "dense.pth"},
                "existing_finetuned_checkpoints": {"64": "missing_rank64.pth"},
            }
            reference = {
                "target_layer": "0",
                "dense_total_parameters": 10,
                "dense_target_layer_parameters": 5,
            }
            with mock.patch("src.smallnet.experiment.parameter_reference", return_value=reference):
                path = run_existing_finetuned_evaluation(config, root, torch.device("cpu"), max_batches=0)

            with open(path) as f:
                metadata = json.load(f)

            self.assertEqual(metadata["kind"], "existing_finetuned_evaluation")
            self.assertEqual(metadata["evaluations"], [])
            self.assertEqual(metadata["skipped"][0]["rank"], 64)
            self.assertIn("missing", metadata["skipped"][0]["reason"])


class StageScopeTests(unittest.TestCase):
    def test_run_profiling_does_not_reference_rank_diagnostics_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = nn.Sequential(nn.Conv2d(3, 2, kernel_size=1))
            reference = {
                "target_layer": "0",
                "dense_total_parameters": sum(param.numel() for param in model.parameters()),
                "dense_target_layer_parameters": sum(param.numel() for param in model[0].parameters()),
            }
            config = {
                "experiment_id": "unit",
                "output_dir": "out",
                "model": {"target_layer": "0", "dense_checkpoint": "dense.pth"},
                "cp": {"profile_ranks": []},
                "profiling": {
                    "input_size": [1, 3, 4, 4],
                    "latency": False,
                    "include_existing_finetuned": False,
                },
            }

            with (
                mock.patch("src.smallnet.experiment.parameter_reference", return_value=reference),
                mock.patch("src.smallnet.experiment.load_dense_model", return_value=model),
                mock.patch(
                    "src.smallnet.experiment.write_rank_energy_artifacts",
                    side_effect=AssertionError("profiling must not write rank artifacts"),
                ),
            ):
                metadata_path = run_profiling(config, root, torch.device("cpu"))

            self.assertTrue((root / "out/profile_summary.csv").is_file())
            self.assertTrue((root / "out/profile_metadata.json").is_file())
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.assertEqual(metadata["kind"], "profile")

    def test_run_rank_diagnostics_metadata_contains_paper_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "dense.pth"
            torch.save({"0.weight": torch.randn(4, 3, 1, 1)}, checkpoint)
            config = {
                "experiment_id": "unit",
                "output_dir": "out",
                "model": {"dense_checkpoint": str(checkpoint), "target_layer": "0"},
                "cp": {"ranks": [1, 2]},
                "rank_diagnostics": {
                    "layers": ["0.weight"],
                    "modes": [0, 1],
                    "thresholds": [0.9],
                    "fixed_ranks": [1, 2],
                },
                "paper": {
                    "tables_dir": "paper/tables",
                    "figures_dir": "paper/figures",
                    "manifest_path": "paper/MANIFEST.json",
                },
            }

            metadata_path = run_rank_diagnostics(config, root, device=torch.device("cpu"))
            self.assertTrue((root / "out/rank_diagnostics_summary.csv").is_file())
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.assertEqual(metadata["kind"], "rank_energy_diagnostics")
            self.assertIn("paper_outputs", metadata)
            self.assertTrue(Path(metadata["paper_outputs"]["rank_energy_summary_csv"]).is_file())


class RankEnergyArtifactTests(unittest.TestCase):
    def test_rank_energy_tables_from_synthetic_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = torch.zeros(4, 4, 1, 1)
            weight[:, :, 0, 0] = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
            diagnostic = rank_energy_diagnostic(weight, modes=[0, 1], thresholds=[0.9], fixed_ranks=[1, 2])
            config = {
                "cp": {"ranks": [1, 2]},
                "rank_diagnostics": {"fixed_ranks": [1, 2]},
                "paper": {
                    "tables_dir": "paper/tables",
                    "figures_dir": "paper/figures",
                    "manifest_path": "paper/MANIFEST.json",
                },
            }

            outputs = write_rank_energy_artifacts({"layer.weight": diagnostic}, config, root)

            self.assertTrue(Path(outputs["rank_energy_summary_csv"]).is_file())
            self.assertTrue(Path(outputs["rank_energy_summary_tex"]).is_file())
            self.assertTrue(Path(outputs["singular_values_by_mode_csv"]).is_file())
            self.assertTrue((root / "paper/figures/singular_value_decay_by_mode.pdf").is_file())
            self.assertTrue((root / "paper/figures/cumulative_energy_by_mode.pdf").is_file())


class AggregationTests(unittest.TestCase):
    def test_aggregation_succeeds_with_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "output_dir": "results/camvid_vgg_cp",
                "paper": {
                    "tables_dir": "results/paper/tables",
                    "figures_dir": "results/paper/figures",
                    "manifest_path": "results/paper/MANIFEST.json",
                },
            }

            manifest_path = build_paper_artifacts(config, root)
            with open(manifest_path) as f:
                manifest = json.load(f)

            self.assertTrue((root / "results/paper/tables/main_results.csv").is_file())
            self.assertTrue((root / "results/paper/tables/profile_results.csv").is_file())
            self.assertGreaterEqual(len(manifest["missing_inputs"]), 1)


if __name__ == "__main__":
    unittest.main()
