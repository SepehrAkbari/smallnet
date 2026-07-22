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
    cp_iteration_budget_transition_rows,
    cp_iteration_rows_for_conv,
    normalize_cp_iteration_rows,
    normalize_iteration_budget_grid,
    reference_diagnostic_for_conv,
    sensitivity_rank_ordering_changes,
    write_cp_iteration_sensitivity_audit,
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
    def test_arbitrary_budget_grid_must_be_unique_and_increasing(self):
        self.assertEqual(
            normalize_iteration_budget_grid([5, "20", 75, 300]),
            [5, 20, 75, 300],
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalize_iteration_budget_grid([10, 50, 25])
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_iteration_budget_grid([10, 25, 25])

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

    def test_generalized_adjacent_budget_transitions_and_highest_pair(self):
        budgets = [10, 25, 50, 100, 200, 400]
        means = {10: 0.90, 25: 0.87, 50: 0.85, 100: 0.83, 200: 0.82, 400: 0.8195}
        rows = [
            completed_row(128, seed, budget, mean + (seed - 1) * 0.001)
            for budget, mean in means.items()
            for seed in (0, 1, 2)
        ]
        compared, _ = apply_residual_reduction_comparisons(rows, budgets)
        self.assertAlmostEqual(
            compared[0]["absolute_squared_residual_reduction_200_to_400"],
            0.0005,
        )
        self.assertEqual(compared[0]["smallest_completed_iteration_budget"], 10)
        self.assertEqual(compared[0]["largest_completed_iteration_budget"], 400)
        aggregates, _ = aggregate_cp_iteration_rows(
            compared, expected_seed_count=3, iteration_budgets=budgets
        )
        transitions = cp_iteration_budget_transition_rows(aggregates, budgets, 3)
        self.assertEqual(len(transitions), 5)
        final = transitions[-1]
        self.assertEqual((final["lower_budget"], final["upper_budget"]), (200, 400))
        self.assertAlmostEqual(final["mean_absolute_squared_residual_reduction"], 0.0005)
        self.assertTrue(final["mean_absolute_change_below_1e_minus_3"])
        aggregate = aggregates[0]
        self.assertEqual(aggregate["highest_two_complete_lower_budget"], 200)
        self.assertEqual(aggregate["highest_two_complete_upper_budget"], 400)
        self.assertAlmostEqual(
            aggregate[
                "mean_relative_squared_residual_reduction_smallest_to_largest_complete_budget"
            ],
            (0.90 - 0.8195) / 0.90,
        )


class CPIterationSensitivityArtifactTests(unittest.TestCase):
    def _audit_for_final_change(self, root, final_change):
        budgets = [10, 25, 50, 100, 200, 400]
        rows = []
        for rank, offset in ((128, 0.0), (256, -0.04), (512, -0.08)):
            means = {
                10: 0.90 + offset,
                25: 0.87 + offset,
                50: 0.85 + offset,
                100: 0.83 + offset,
                200: 0.82 + offset,
                400: 0.82 + offset - final_change,
            }
            rows.extend(
                completed_row(rank, seed, budget, mean + (seed - 1) * 0.0001)
                for budget, mean in means.items()
                for seed in (0, 1, 2)
            )
        aggregates, _ = aggregate_cp_iteration_rows(
            rows, expected_seed_count=3, iteration_budgets=budgets
        )
        transitions = cp_iteration_budget_transition_rows(aggregates, budgets, 3)
        audit_path = Path(root) / f"audit_{final_change}.md"
        write_cp_iteration_sensitivity_audit(
            aggregates,
            audit_path,
            canonical_ranks=[128, 256, 512],
            canonical_seeds=[0, 1, 2],
            canonical_iteration_budgets=budgets,
            failures=[],
            rank_ordering=sensitivity_rank_ordering_changes(aggregates),
            budget_transitions=transitions,
        )
        return audit_path.read_text()

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

    def test_partial_200_or_400_results_defer_final_stopping_decision(self):
        budgets = [10, 25, 50, 100, 200, 400]
        rows = [
            completed_row(128, seed, budget, 0.9 - budget / 10000 + seed / 10000)
            for budget in (10, 25, 50, 100, 200)
            for seed in (0, 1, 2)
        ]
        aggregates, _ = aggregate_cp_iteration_rows(
            rows, expected_seed_count=3, iteration_budgets=budgets
        )
        transitions = cp_iteration_budget_transition_rows(aggregates, budgets, 3)
        self.assertEqual((transitions[-1]["lower_budget"], transitions[-1]["upper_budget"]), (100, 200))
        self.assertFalse(any(row["upper_budget"] == 400 for row in transitions))
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "partial.md"
            _, complete = write_cp_iteration_sensitivity_audit(
                aggregates,
                audit,
                canonical_ranks=[128],
                canonical_seeds=[0, 1, 2],
                canonical_iteration_budgets=budgets,
                failures=[],
                rank_ordering=sensitivity_rank_ordering_changes(aggregates),
                budget_transitions=transitions,
            )
            text = audit.read_text()
        self.assertFalse(complete)
        self.assertIn("400-iteration plateau decision: deferred", text)

    def test_audit_800_rule_for_below_and_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            stable = self._audit_for_final_change(tmp, 0.0005)
            unstable = self._audit_for_final_change(tmp, 0.0015)
        self.assertIn("Is one final 800-iteration check needed?** `False`", stable)
        self.assertIn("Is one final 800-iteration check needed?** `True`", unstable)
        self.assertIn("Use the common budget `200`", stable)
        self.assertIn("Use the common budget `400`", unstable)

    def test_backward_compatible_four_budget_rows_gain_generalized_fields(self):
        old_rows = [
            completed_row(128, seed, budget, 0.9 - budget / 1000 + seed / 10000)
            for budget in (10, 25, 50, 100)
            for seed in (0, 1, 2)
        ]
        compared, diagnostics = apply_residual_reduction_comparisons(
            old_rows, [10, 25, 50, 100, 200, 400]
        )
        self.assertEqual(diagnostics, [])
        self.assertTrue(all("adjacent_budget_residual_reductions_json" in row for row in compared))
        self.assertTrue(all(row["absolute_squared_residual_reduction_100_to_200"] == "" for row in compared))
        self.assertEqual({row["largest_completed_iteration_budget"] for row in compared}, {100})

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

    def test_old_four_budget_stage_extends_only_missing_200_and_400_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            canonical_path = output / "reconstruction_summary.csv"
            canonical_path.write_text("canonical-sentinel\n")
            base = {
                "experiment_id": "six_budget_extension_unit",
                "output_dir": "out",
                "model": {"target_layer": "0"},
                "reconstruction": {
                    "memory_efficient_mttkrp": True,
                    "mttkrp_rank_chunk_size": 1,
                    "mttkrp_max_explicit_bytes": 0,
                },
                "cp_iteration_sensitivity": {
                    "synthetic_tensor_shape": [3, 2, 2, 2],
                    "synthetic_seed": 11,
                    "ranks": [1],
                    "seeds": [0],
                    "iteration_budgets": [10, 25, 50, 100, 200, 400],
                    "execution_iteration_budgets": [10, 25, 50, 100],
                    "audit_path": "out/audit.md",
                },
                "paper": {"figures_dir": "out/figures"},
            }
            run_cp_iteration_sensitivity(base, root, torch.device("cpu"))
            summary_path = output / "cp_iteration_sensitivity_summary.csv"
            with summary_path.open(newline="") as handle:
                old_rows = list(csv.DictReader(handle))
            self.assertEqual(len(old_rows), 4)
            scientific_before = {
                (row["rank"], row["seed"], row["iteration_budget"]): (
                    row["actual_relative_squared_frobenius_error"],
                    row["initialization_hash_sha256"],
                    row["decomposition_runtime_seconds"],
                )
                for row in old_rows
            }
            legacy_fields = [
                field
                for field in old_rows[0]
                if not field.startswith("absolute_squared_residual_reduction_100_to_")
                and not field.startswith("absolute_squared_residual_reduction_200_to_")
                and not field.startswith("relative_squared_residual_reduction_100_to_")
                and not field.startswith("relative_squared_residual_reduction_200_to_")
                and field
                not in {
                    "adjacent_budget_residual_reductions_json",
                    "smallest_completed_iteration_budget",
                    "largest_completed_iteration_budget",
                    "relative_squared_residual_reduction_smallest_to_largest_completed",
                }
            ]
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=legacy_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(
                    {field: row.get(field, "") for field in legacy_fields} for row in old_rows
                )

            extended = json.loads(json.dumps(base))
            extended["cp_iteration_sensitivity"]["execution_iteration_budgets"] = [
                10,
                25,
                50,
                100,
                200,
                400,
            ]
            from src.smallnet.cp_iteration_sensitivity import (
                fit_cp_approximation_with_initialization_capture as real_fit,
            )

            fitted_budgets = []

            def record_budget(*args, **kwargs):
                fitted_budgets.append(int(args[4]))
                return real_fit(*args, **kwargs)

            with mock.patch(
                "src.smallnet.cp_iteration_sensitivity.fit_cp_approximation_with_initialization_capture",
                side_effect=record_budget,
            ):
                run_cp_iteration_sensitivity(extended, root, torch.device("cpu"))
            self.assertEqual(fitted_budgets, [200, 400])
            with summary_path.open(newline="") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual(len(final_rows), 6)
            self.assertEqual(
                {(int(row["iteration_budget"])) for row in final_rows},
                {10, 25, 50, 100, 200, 400},
            )
            for row in final_rows:
                key = (row["rank"], row["seed"], row["iteration_budget"])
                if key in scientific_before:
                    self.assertEqual(
                        (
                            row["actual_relative_squared_frobenius_error"],
                            row["initialization_hash_sha256"],
                            row["decomposition_runtime_seconds"],
                        ),
                        scientific_before[key],
                    )
            self.assertEqual(len({row["initialization_hash_sha256"] for row in final_rows}), 1)
            with (output / "cp_iteration_sensitivity_budget_transitions.csv").open(
                newline=""
            ) as handle:
                transitions = list(csv.DictReader(handle))
            self.assertEqual(len(transitions), 5)
            self.assertEqual(canonical_path.read_text(), "canonical-sentinel\n")


if __name__ == "__main__":
    unittest.main()
