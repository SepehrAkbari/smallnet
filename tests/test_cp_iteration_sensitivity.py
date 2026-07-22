import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from src.smallnet.cp_iteration_sensitivity import (
    aggregate_cp_iteration_rows,
    apply_residual_reduction_comparisons,
    cp_initialization_hash,
    cp_iteration_rows_for_conv,
    normalize_cp_iteration_rows,
    reference_diagnostic_for_conv,
    write_cp_iteration_sensitivity_figure,
)
from src.smallnet.experiment import run_cp_iteration_sensitivity


def completed_row(rank, seed, budget, residual, bound=0.2, svd=0.15):
    return {
        "method": "cp",
        "rank": rank,
        "seed": seed,
        "iteration_budget": budget,
        "status": "completed",
        "actual_relative_squared_frobenius_error": residual,
        "max_unfolding_tail_bound_squared": bound,
        "output_mode_svd_residual_squared": svd,
        "initialization_hash_sha256": f"hash-{rank}-{seed}",
    }


class CPIterationSensitivityMathTests(unittest.TestCase):
    def test_identical_initialization_hash_across_budgeted_runs(self):
        conv = nn.Conv2d(3, 5, 3)
        hashes = [
            cp_initialization_hash(conv, 2, 7, "random", torch.device("cpu"))
            for _budget in (1, 2, 4)
        ]
        self.assertEqual(len(set(hashes)), 1)
        different_seed = cp_initialization_hash(conv, 2, 8, "random", torch.device("cpu"))
        self.assertNotEqual(hashes[0], different_seed)

    def test_residual_difference_calculations(self):
        rows = [
            completed_row(128, 0, budget, residual)
            for budget, residual in ((10, 0.80), (25, 0.76), (50, 0.75), (100, 0.748))
        ]
        compared, diagnostics = apply_residual_reduction_comparisons(rows)
        self.assertEqual(diagnostics, [])
        row = compared[0]
        self.assertAlmostEqual(row["absolute_squared_residual_reduction_10_to_25"], 0.04)
        self.assertAlmostEqual(row["absolute_squared_residual_reduction_25_to_50"], 0.01)
        self.assertAlmostEqual(row["absolute_squared_residual_reduction_50_to_100"], 0.002)
        self.assertAlmostEqual(row["relative_squared_residual_reduction_10_to_100"], 0.065)

    def test_rank_budget_aggregation(self):
        rows = [
            completed_row(128, seed, budget, residual)
            for budget, residuals in ((10, (0.8, 0.82, 0.78)), (100, (0.79, 0.80, 0.77)))
            for seed, residual in enumerate(residuals)
        ]
        aggregate, diagnostics = aggregate_cp_iteration_rows(rows, expected_seed_count=3)
        self.assertEqual(diagnostics, [])
        self.assertEqual(len(aggregate), 2)
        at_10 = next(row for row in aggregate if row["iteration_budget"] == 10)
        self.assertAlmostEqual(at_10["actual_relative_squared_frobenius_error_mean"], 0.8)
        self.assertAlmostEqual(at_10["actual_relative_squared_frobenius_error_seed_range"], 0.04)
        self.assertEqual(at_10["completed_seed_count"], 3)
        self.assertAlmostEqual(at_10["mean_relative_squared_residual_reduction_10_to_100"], 1 / 60)


class CPIterationSensitivityArtifactTests(unittest.TestCase):
    def test_numeric_key_normalization_and_duplicate_removal(self):
        rows = [
            completed_row(128, 0, 10, 0.8),
            completed_row("128", "0", "10", 0.8),
            completed_row("256", "1", "25", 0.7),
            completed_row("mean", 0, 10, 0.1),
            completed_row(128, "all", 25, 0.1),
            completed_row(128, 0, "", 0.1),
        ]
        with self.assertWarns(RuntimeWarning):
            normalized, rejected, diagnostics = normalize_cp_iteration_rows(rows)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(len(rejected), 3)
        self.assertEqual(len(diagnostics), 4)
        self.assertTrue(all(type(row["rank"]) is int for row in normalized))
        self.assertTrue(all(type(row["seed"]) is int for row in normalized))
        self.assertTrue(all(type(row["iteration_budget"]) is int for row in normalized))

    def test_figure_generation_accepts_partial_and_complete_tables(self):
        partial = [completed_row(128, seed, 10, 0.8 + 0.01 * seed) for seed in (0, 1)]
        complete = [
            completed_row(rank, seed, budget, 0.9 - rank / 2000 - budget / 10000 + seed / 10000)
            for rank in (128, 256)
            for seed in (0, 1)
            for budget in (10, 25)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            partial_paths, _ = write_cp_iteration_sensitivity_figure(partial, Path(tmp) / "partial")
            complete_paths, _ = write_cp_iteration_sensitivity_figure(complete, Path(tmp) / "complete")
            self.assertTrue(all(Path(path).is_file() for path in partial_paths + complete_paths))
            with open(complete_paths[0], newline="") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual({int(row["rank"]) for row in exported}, {128, 256})
        self.assertIn("matrix_svd_reference", {row["series"] for row in exported})
        self.assertIn("strongest_unfolding_bound", {row["series"] for row in exported})

    def test_one_failed_budget_is_recorded_without_losing_completed_rows(self):
        conv = nn.Conv2d(3, 5, 3)
        diagnostic = reference_diagnostic_for_conv(conv.weight, [2], torch.device("cpu"))
        from src.smallnet.cp_iteration_sensitivity import (
            fit_cp_approximation_with_initialization_capture as real_fit,
        )

        def fail_second_budget(*args, **kwargs):
            if int(args[4]) == 2:
                raise RuntimeError("simulated budget failure")
            return real_fit(*args, **kwargs)

        with mock.patch(
            "src.smallnet.cp_iteration_sensitivity.fit_cp_approximation_with_initialization_capture",
            side_effect=fail_second_budget,
        ):
            rows, failures, _ = cp_iteration_rows_for_conv(
                conv,
                [2],
                [0],
                [1, 2],
                "random",
                torch.device("cpu"),
                diagnostic,
                "synthetic",
            )
        self.assertEqual([row["status"] for row in rows], ["completed", "failed"])
        self.assertEqual(len(failures), 1)
        self.assertIn("simulated budget failure", failures[0]["exception"])

    def test_stage_is_resumable_unique_and_does_not_touch_canonical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            canonical_path = output / "reconstruction_summary.csv"
            canonical_path.write_text("canonical-sentinel\n")
            config = {
                "experiment_id": "sensitivity_resume_unit",
                "output_dir": "out",
                "model": {"target_layer": "0"},
                "reconstruction": {
                    "memory_efficient_mttkrp": True,
                    "mttkrp_rank_chunk_size": 2,
                    "mttkrp_max_explicit_bytes": 0,
                },
                "cp_iteration_sensitivity": {
                    "synthetic_tensor_shape": [5, 3, 2, 2],
                    "synthetic_seed": 9,
                    "ranks": [1],
                    "seeds": [0, 1],
                    "iteration_budgets": [1, 2],
                    "audit_path": "out/audit.md",
                },
                "paper": {"figures_dir": "out/figures"},
            }
            metadata_path = run_cp_iteration_sensitivity(config, root, torch.device("cpu"))
            summary_path = output / "cp_iteration_sensitivity_summary.csv"
            with open(summary_path, newline="") as handle:
                before = list(csv.DictReader(handle))
            self.assertEqual(len(before), 4)
            self.assertTrue(
                all(
                    row["initialization_hash_sha256"]
                    == row["actual_fit_initialization_hash_sha256"]
                    for row in before
                )
            )
            self.assertEqual(canonical_path.read_text(), "canonical-sentinel\n")
            with mock.patch(
                "src.smallnet.experiment.cp_iteration_rows_for_conv",
                side_effect=AssertionError("completed rows must not be recomputed"),
            ):
                rerun_metadata_path = run_cp_iteration_sensitivity(
                    config, root, torch.device("cpu")
                )
            with open(summary_path, newline="") as handle:
                after = list(csv.DictReader(handle))
            normalized, rejected, diagnostics = normalize_cp_iteration_rows(after, warn=False)
            self.assertEqual(len(normalized), 4)
            self.assertEqual(rejected, [])
            self.assertEqual(diagnostics, [])
            self.assertEqual(canonical_path.read_text(), "canonical-sentinel\n")
            for path in (
                metadata_path,
                rerun_metadata_path,
                output / "cp_iteration_sensitivity_rank_summary.csv",
                output / "cp_iteration_sensitivity_config_used.json",
                output / "audit.md",
            ):
                self.assertTrue(Path(path).is_file())
            with open(rerun_metadata_path) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["failures"], [])
            self.assertTrue(
                all(
                    item["identical_across_completed_budgets"]
                    for item in metadata["initialization_verification"]
                )
            )


if __name__ == "__main__":
    unittest.main()
