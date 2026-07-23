import csv
import importlib
import importlib.metadata
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from src.smallnet.experiment import run_final_structural
from src.smallnet.factorization import MatrixLowRankConv2d, factorized_conv_from_conv
from src.smallnet.final_structural import (
    ActivationDistortionAccumulator,
    FINAL_CP_METHOD,
    FINAL_ITERATION_BUDGET,
    FINAL_SVD_METHOD,
    final_structural_key,
    fitted_factor_hash,
    load_factor_artifact,
    normalize_final_structural_rows,
    save_factor_artifact,
    verify_matrix_svd_reconstruction,
    verify_completed_row_schema,
    write_final_structural_audit,
    write_final_structural_figures,
)


def metric_row(method, rank, seed="", status="completed"):
    row = {
        "method": method,
        "rank": rank,
        "seed": seed,
        "iteration_budget": FINAL_ITERATION_BUDGET
        if method == FINAL_CP_METHOD
        else "",
        "status": status,
        "actual_relative_squared_frobenius_error": 0.8,
        "actual_relative_frobenius_error": 0.8**0.5,
        "gap_above_max_bound": 0.1,
        "activation_normalized_squared_error": 0.5,
        "activation_relative_frobenius_error": 0.5**0.5,
        "activation_cosine_similarity": 0.9,
        "dense_activation_norm": 10.0,
        "compressed_activation_norm": 9.0,
        "activation_absolute_mse": 0.1,
        "activation_example_normalized_squared_error_mean": 0.5,
        "activation_example_normalized_squared_error_std_population": 0.1,
        "activation_example_normalized_squared_error_min": 0.4,
        "activation_example_normalized_squared_error_max": 0.6,
        "validation_present_class_miou": 0.3,
        "test_present_class_miou": 0.25,
        "target_layer_parameter_count": 100,
        "target_layer_parameter_ratio": 0.1,
        "full_model_parameter_count": 1000,
        "target_layer_macs": 200,
        "full_model_macs": 2000,
        "compression_factor": 10.0,
    }
    if method == FINAL_CP_METHOD:
        row.update(
            {
                "initialization_hash_sha256": "init",
                "final_factor_hash_sha256": "factor",
                "factor_artifact_sha256": "artifact",
                "decomposition_runtime_seconds": 1.0,
            }
        )
    else:
        row.update(
            {
                "current_output_mode_tail_bound_squared": 0.8,
                "stored_diagnostic_output_mode_tail_bound_squared": 0.81,
                "current_vs_direct_residual_absolute_difference": 0.0,
                "current_vs_stored_output_tail_absolute_difference": 0.01,
                "matrix_svd_verification_tolerance": 1e-5,
                "matrix_kernel_vs_direct_svd_max_abs_difference": 0.0,
                "matrix_kernel_vs_direct_svd_relative_frobenius_difference": 0.0,
            }
        )
    return row


class FinalStructuralMathTests(unittest.TestCase):
    def test_same_svd_tail_passes_and_mismatched_legacy_tail_is_recorded(self):
        singular_spectrum = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        conv = nn.Conv2d(4, 4, kernel_size=1, bias=False).double()
        conv.weight.data.copy_(torch.diag(singular_spectrum).reshape(4, 4, 1, 1))
        matrix = conv.weight.detach().reshape(4, 4)
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        replacement = MatrixLowRankConv2d.from_svd(
            conv, 2, u, singular_values, vh
        )
        result = verify_matrix_svd_reconstruction(
            conv.weight,
            replacement,
            u,
            singular_values,
            vh,
            2,
            stored_diagnostic_output_tail_squared=0.25,
            tolerance=1e-5,
            svd_computation_device="cpu",
        )
        expected_tail = float((2.0**2 + 1.0**2) / (4.0**2 + 3.0**2 + 2.0**2 + 1.0**2))
        self.assertAlmostEqual(
            result["actual_relative_squared_frobenius_error"],
            expected_tail,
            places=12,
        )
        self.assertAlmostEqual(
            result["current_output_mode_tail_bound_squared"],
            expected_tail,
            places=15,
        )
        self.assertTrue(result["matrix_svd_residual_verification_passed"])
        self.assertFalse(
            result["matrix_svd_stored_tail_agreement_within_tolerance"]
        )
        self.assertAlmostEqual(
            result["current_vs_stored_output_tail_absolute_difference"],
            abs(expected_tail - 0.25),
        )
        self.assertEqual(result["residual_evaluation_precision"], "float64")
        self.assertIn("cpu_float64", result["residual_evaluation_path"])

    def test_incorrect_composed_kernel_is_rejected(self):
        conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        conv.weight.data.copy_(matrix.reshape(4, 4, 1, 1))
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        replacement = MatrixLowRankConv2d.from_svd(
            conv, 2, u, singular_values, vh
        )
        replacement[0].weight.data[0, 0, 0, 0] += 0.1
        with self.assertRaisesRegex(AssertionError, "does not match"):
            verify_matrix_svd_reconstruction(
                conv.weight,
                replacement,
                u,
                singular_values,
                vh,
                2,
                stored_diagnostic_output_tail_squared=1 / 6,
                tolerance=1e-5,
            )

    def test_200_iteration_protocol_is_explicit(self):
        self.assertEqual(
            final_structural_key(metric_row(FINAL_CP_METHOD, 128, 0)),
            (FINAL_CP_METHOD, 128, 0, 200),
        )
        bad = metric_row(FINAL_CP_METHOD, 128, 0)
        bad["iteration_budget"] = 199
        with self.assertRaisesRegex(ValueError, "exactly|must use"):
            final_structural_key(bad)

    def test_activation_aggregation_is_not_batch_ratio_average(self):
        accumulator = ActivationDistortionAccumulator()
        accumulator.update(
            torch.tensor([[[[1.0]]], [[[3.0]]]]),
            torch.tensor([[[[0.0]]], [[[3.0]]]]),
        )
        accumulator.update(
            torch.tensor([[[[2.0]]]]),
            torch.tensor([[[[0.0]]]]),
        )
        result = accumulator.finalize()
        self.assertAlmostEqual(
            result["activation_normalized_squared_error"], 5.0 / 14.0
        )
        self.assertAlmostEqual(
            result["activation_example_normalized_squared_error_mean"], 2.0 / 3.0
        )
        self.assertEqual(result["activation_example_count"], 3)

    def test_cp_and_matrix_svd_schemas(self):
        self.assertTrue(verify_completed_row_schema(metric_row(FINAL_CP_METHOD, 128, 0)))
        self.assertTrue(
            verify_completed_row_schema(metric_row(FINAL_SVD_METHOD, 128))
        )
        missing = metric_row(FINAL_CP_METHOD, 128, 0)
        missing["factor_artifact_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "lacks fields"):
            verify_completed_row_schema(missing)

    def test_duplicate_key_never_replaces_completed_with_failed(self):
        completed = metric_row(FINAL_CP_METHOD, "128", "0")
        failed = metric_row(FINAL_CP_METHOD, 128, 0, status="failed")
        with self.assertWarns(RuntimeWarning):
            rows, rejected, diagnostics = normalize_final_structural_rows(
                [completed, failed]
            )
        self.assertEqual(rejected, [])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "completed")

    def test_unique_failed_row_is_retained_for_retry(self):
        failed = metric_row(FINAL_CP_METHOD, 256, 2, status="failed")
        failed["failure_exception"] = "simulated failure"
        rows, rejected, diagnostics = normalize_final_structural_rows([failed])
        self.assertEqual(rejected, [])
        self.assertEqual(diagnostics, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["failure_exception"], "simulated failure")


class FinalStructuralArtifactTests(unittest.TestCase):
    def test_declared_tensorly_distributions_are_importable(self):
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text()
        )
        dependency_names = {
            requirement.split(">=", 1)[0].split("==", 1)[0]
            for requirement in project["project"]["dependencies"]
        }
        self.assertIn("tensorly", dependency_names)
        self.assertIn("tensorly-torch", dependency_names)
        self.assertTrue(importlib.metadata.version("tensorly"))
        self.assertTrue(importlib.metadata.version("tensorly-torch"))
        self.assertIsNotNone(importlib.import_module("tensorly"))
        self.assertIsNotNone(importlib.import_module("tltorch"))

    def test_factor_artifact_round_trip_and_hash_verification(self):
        conv = nn.Conv2d(3, 5, 3)
        fitted = factorized_conv_from_conv(
            conv, rank=2, factorization="cp", init="random", n_iter_max=0
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factor.pt"
            saved = save_factor_artifact(
                path,
                fitted,
                rank=2,
                seed=1,
                iteration_budget=200,
                initializer="random",
                initialization_hash="init",
                checkpoint_hash="checkpoint",
                dataset_validation_hash="dataset",
                target_tensor_hash="tensor",
                decomposition_runtime_seconds=3.0,
            )
            loaded, payload, identity = load_factor_artifact(
                path,
                conv,
                expected_rank=2,
                expected_seed=1,
                expected_iteration_budget=200,
                expected_initializer="random",
                expected_checkpoint_hash="checkpoint",
                expected_dataset_validation_hash="dataset",
                expected_target_tensor_hash="tensor",
            )
            self.assertEqual(fitted_factor_hash(loaded), fitted_factor_hash(fitted))
            self.assertEqual(identity["factor_artifact_sha256"], saved["factor_artifact_sha256"])
            self.assertEqual(payload["decomposition_runtime_seconds"], 3.0)

    def test_partial_and_complete_figure_audit_generation(self):
        partial = [metric_row(FINAL_CP_METHOD, 32, 0)]
        complete = [
            metric_row(FINAL_CP_METHOD, rank, seed)
            for rank in (32, 64)
            for seed in (0, 1)
        ] + [metric_row(FINAL_SVD_METHOD, rank) for rank in (32, 64)]
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = write_final_structural_figures(partial, Path(tmp) / "figures")
            self.assertTrue(all(Path(path).is_file() for path in paths))
            partial_audit = Path(tmp) / "partial.md"
            _, partial_complete = write_final_structural_audit(
                partial,
                [],
                partial_audit,
                expected_ranks=[32, 64],
                expected_seeds=[0, 1],
            )
            self.assertFalse(partial_complete)
            from src.smallnet.final_structural import aggregate_final_structural_rows

            rank_summary, _ = aggregate_final_structural_rows(complete)
            complete_audit = Path(tmp) / "complete.md"
            _, is_complete = write_final_structural_audit(
                complete,
                rank_summary,
                complete_audit,
                expected_ranks=[32, 64],
                expected_seeds=[0, 1],
            )
            self.assertTrue(is_complete)

    def test_synthetic_stage_fits_once_reuses_factors_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preliminary = root / "preliminary"
            preliminary.mkdir()
            sentinels = [
                preliminary / "reconstruction_summary.csv",
                preliminary / "structural_zero_shot_summary.csv",
                preliminary / "cp_iteration_sensitivity_summary.csv",
            ]
            for path in sentinels:
                path.write_text("unchanged\n")
            output = root / "final"
            config = {
                "experiment_id": "final-structural-test",
                "output_dir": str(preliminary),
                "final_structural": {
                    "synthetic_tensor_shape": [5, 3, 3, 3],
                    "synthetic_seed": 4,
                    "ranks": [1],
                    "seeds": [0],
                    "iteration_budget": 200,
                    "init": "random",
                    "output_dir": str(output),
                    "figures_dir": str(output / "figures"),
                    "audit_path": str(output / "audit.md"),
                },
            }
            from src.smallnet.experiment import (
                fit_cp_approximation_with_initialization_capture as real_fit,
            )

            calls = []

            def counting_fit(*args, **kwargs):
                calls.append((args[1], args[2], args[4]))
                return real_fit(*args, **kwargs)

            with mock.patch(
                "src.smallnet.experiment.fit_cp_approximation_with_initialization_capture",
                side_effect=counting_fit,
            ):
                run_final_structural(config, root, torch.device("cpu"))
            self.assertEqual(calls, [(1, 0, 200)])
            rows = list(
                csv.DictReader((output / "final_structural_summary.csv").open())
            )
            self.assertEqual(len(rows), 2)
            cp = next(row for row in rows if row["method"] == FINAL_CP_METHOD)
            self.assertEqual(cp["same_fitted_factors_reused_for_all_metrics"], "True")
            self.assertEqual(
                {
                    cp["final_factor_hash_sha256"],
                    cp["factor_hash_after_reconstruction"],
                    cp["factor_hash_after_activation_distortion"],
                    cp["factor_hash_after_zero_shot"],
                }.__len__(),
                1,
            )
            self.assertTrue(Path(cp["factor_artifact_path"]).is_file())
            self.assertTrue(all(path.read_text() == "unchanged\n" for path in sentinels))
            with mock.patch(
                "src.smallnet.experiment.fit_cp_approximation_with_initialization_capture",
                side_effect=AssertionError("completed key must not refit"),
            ):
                run_final_structural(config, root, torch.device("cpu"))
            self.assertEqual(
                len(
                    list(
                        csv.DictReader(
                            (output / "final_structural_summary.csv").open()
                        )
                    )
                ),
                2,
            )

    def test_completed_row_rejects_tampered_factor_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "final"
            config = {
                "experiment_id": "identity-test",
                "output_dir": str(root / "preliminary"),
                "final_structural": {
                    "synthetic_tensor_shape": [5, 3, 3, 3],
                    "ranks": [1],
                    "seeds": [0],
                    "iteration_budget": 200,
                    "output_dir": str(output),
                    "figures_dir": str(output / "figures"),
                    "audit_path": str(output / "audit.md"),
                },
            }
            run_final_structural(config, root, torch.device("cpu"))
            factor = next((output / "factors").glob("*.pt"))
            with factor.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "file hash changed"):
                run_final_structural(config, root, torch.device("cpu"))

    def test_retry_failed_svd_skips_completed_cp_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "final"
            config = {
                "experiment_id": "svd-retry-test",
                "output_dir": str(root / "preliminary"),
                "final_structural": {
                    "synthetic_tensor_shape": [5, 3, 3, 3],
                    "ranks": [1],
                    "seeds": [0],
                    "iteration_budget": 200,
                    "output_dir": str(output),
                    "figures_dir": str(output / "figures"),
                    "audit_path": str(output / "audit.md"),
                },
            }
            run_final_structural(config, root, torch.device("cpu"))
            summary = output / "final_structural_summary.csv"
            rows = list(csv.DictReader(summary.open()))
            cp_before = next(row for row in rows if row["method"] == FINAL_CP_METHOD)
            svd = next(row for row in rows if row["method"] == FINAL_SVD_METHOD)
            svd = {
                "method": FINAL_SVD_METHOD,
                "rank": svd["rank"],
                "seed": "",
                "iteration_budget": "",
                "status": "failed",
                "failure_exception": "legacy-tail mismatch",
            }
            from src.smallnet.results import write_csv

            write_csv(summary, [cp_before, svd])
            with mock.patch(
                "src.smallnet.experiment.fit_cp_approximation_with_initialization_capture",
                side_effect=AssertionError("completed CP row must be skipped"),
            ):
                run_final_structural(config, root, torch.device("cpu"))
            retried = list(csv.DictReader(summary.open()))
            cp_after = next(
                row for row in retried if row["method"] == FINAL_CP_METHOD
            )
            svd_after = next(
                row for row in retried if row["method"] == FINAL_SVD_METHOD
            )
            self.assertEqual(cp_after["factor_artifact_sha256"], cp_before["factor_artifact_sha256"])
            self.assertEqual(cp_after["status"], "completed")
            self.assertEqual(svd_after["status"], "completed")
            self.assertTrue(
                float(svd_after["current_vs_direct_residual_absolute_difference"])
                <= float(svd_after["matrix_svd_verification_tolerance"])
            )


if __name__ == "__main__":
    unittest.main()
