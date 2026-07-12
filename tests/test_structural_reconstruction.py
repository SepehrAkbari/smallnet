import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.smallnet.diagnostics import cp_lower_bound_tail, cp_tensor_from_factors, unfold_tensor
from src.smallnet.experiment import run_reconstruction
from src.smallnet.factorization import MatrixLowRankConv2d
from src.smallnet.reproducibility import set_seed
from src.smallnet.structural import (
    _rank_chunked_mttkrp,
    aggregate_cp_reconstruction_rows,
    cp_conv_parameter_count,
    dense_conv_parameter_count,
    fit_cp_approximation,
    join_structural_tradeoffs,
    matrix_svd_conv_parameter_count,
    normalized_frobenius_residual,
    output_mode_svd,
    reconstruction_rows_for_conv,
    write_reconstruction_figure,
)
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao as explicit_mttkrp


class StructuralMathTests(unittest.TestCase):
    def test_memory_bounded_mttkrp_matches_tensorly_explicit_contraction(self):
        tensor = torch.randn(5, 4, 3, 2)
        factors = [torch.randn(size, 4) for size in tensor.shape]
        weights = torch.randn(4)
        for mode in range(tensor.ndim):
            expected = explicit_mttkrp(tensor, (weights, factors), mode)
            actual = _rank_chunked_mttkrp(tensor, (weights, factors), mode, rank_chunk_size=2)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_frobenius_norm_is_invariant_under_every_unfolding(self):
        tensor = torch.randn(5, 4, 3, 2)
        expected = torch.linalg.vector_norm(tensor)
        for mode in range(tensor.ndim):
            self.assertTrue(torch.allclose(expected, torch.linalg.vector_norm(unfold_tensor(tensor, mode))))

    def test_cp_unfolding_rank_is_at_most_cp_rank(self):
        factors = [torch.randn(size, 3) for size in (5, 4, 3, 2)]
        tensor = cp_tensor_from_factors(factors)
        for mode in range(4):
            self.assertLessEqual(int(torch.linalg.matrix_rank(unfold_tensor(tensor, mode))), 3)

    def test_fitted_cp_reconstruction_has_expected_shape(self):
        set_seed(0)
        conv = nn.Conv2d(3, 5, 3)
        _, approximation, _, backend = fit_cp_approximation(
            conv,
            2,
            0,
            "random",
            2,
            torch.device("cpu"),
            memory_efficient_mttkrp=True,
            mttkrp_rank_chunk_size=1,
            mttkrp_max_explicit_bytes=0,
        )
        self.assertEqual(tuple(approximation.shape), tuple(conv.weight.shape))
        self.assertEqual(backend, "smallnet_hybrid_memory_bounded_mttkrp")

    def test_squared_and_ordinary_normalized_residuals_are_distinct_and_correct(self):
        reference = torch.tensor([3.0, 4.0])
        approximation = torch.tensor([0.0, 4.0])
        squared, ordinary = normalized_frobenius_residual(reference, approximation)
        self.assertAlmostEqual(squared, 9.0 / 25.0)
        self.assertAlmostEqual(ordinary, 3.0 / 5.0)

    def test_cp_residual_satisfies_strongest_unfolding_lower_bound(self):
        set_seed(1)
        conv = nn.Conv2d(4, 6, 3)
        _, approximation, _, _ = fit_cp_approximation(conv, 2, 1, "random", 2, torch.device("cpu"))
        squared, _ = normalized_frobenius_residual(conv.weight, approximation)
        bound = cp_lower_bound_tail(conv.weight, 2)["max_tail_energy"]
        self.assertGreaterEqual(squared + 1e-6, bound)

    def test_output_svd_achieves_eckart_young_residual(self):
        conv = nn.Conv2d(3, 5, 3)
        u, s, vh, _, _ = output_mode_svd(conv.weight)
        rank = 2
        approximation = (u[:, :rank] * s[:rank]) @ vh[:rank]
        squared, _ = normalized_frobenius_residual(conv.weight, approximation.reshape_as(conv.weight))
        expected = float(torch.sum(s[rank:].double().square()) / torch.sum(s.double().square()))
        self.assertAlmostEqual(squared, expected, places=6)

    def test_matrix_svd_composed_kernel_matches_truncated_tensor(self):
        conv = nn.Conv2d(3, 5, 3, bias=True)
        u, s, vh, _, _ = output_mode_svd(conv.weight)
        module = MatrixLowRankConv2d.from_svd(conv, 2, u, s, vh)
        expected = ((u[:, :2] * s[:2]) @ vh[:2]).reshape_as(conv.weight)
        self.assertTrue(torch.allclose(module.composed_kernel(), expected, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(module[1].bias, conv.bias))

    def test_exact_parameter_count_formulas(self):
        shape = (5, 3, 3, 3)
        self.assertEqual(dense_conv_parameter_count(shape, True), 140)
        self.assertEqual(cp_conv_parameter_count(shape, 2, True), 35)
        self.assertEqual(matrix_svd_conv_parameter_count(shape, 2, True), 69)
        conv = nn.Conv2d(3, 5, 3, bias=True)
        matrix = MatrixLowRankConv2d.from_conv(conv, 2)
        self.assertEqual(sum(parameter.numel() for parameter in matrix.parameters()), 69)


class StructuralArtifactTests(unittest.TestCase):
    def test_cp_seed_aggregation_retains_mean_std_min_and_max(self):
        rows = [
            {"method": "cp", "rank": 2, "seed": seed, "status": "completed", "actual_relative_squared_frobenius_error": value}
            for seed, value in enumerate((0.4, 0.5, 0.6))
        ]
        aggregate = aggregate_cp_reconstruction_rows(rows)[0]
        self.assertAlmostEqual(aggregate["actual_relative_squared_frobenius_error_mean"], 0.5)
        self.assertAlmostEqual(aggregate["actual_relative_squared_frobenius_error_min"], 0.4)
        self.assertAlmostEqual(aggregate["actual_relative_squared_frobenius_error_max"], 0.6)
        self.assertGreater(aggregate["actual_relative_squared_frobenius_error_std_population"], 0.0)

    def test_figure_data_contains_all_ranks_and_methods(self):
        rows = []
        for rank in (1, 2):
            rows.append({"method": "matrix_svd_output_unfolding", "rank": rank, "seed": "", "status": "completed", "actual_relative_squared_frobenius_error": 0.5, "max_unfolding_tail_bound_squared": 0.4})
            for seed in (0, 1):
                rows.append({"method": "cp", "rank": rank, "seed": seed, "status": "completed", "actual_relative_squared_frobenius_error": 0.7, "max_unfolding_tail_bound_squared": 0.4})
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_reconstruction_figure(rows, tmp)
            with open(paths[0], newline="") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual({int(row["rank"]) for row in exported}, {1, 2})
        self.assertEqual({row["method"] for row in exported}, {"cp", "matrix_svd_output_unfolding"})

    def test_reconstruction_stage_runs_without_camvid_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "experiment_id": "synthetic_unit",
                "output_dir": "out",
                "model": {"target_layer": "0"},
                "cp": {"init": "random", "n_iter_max": 2},
                "reconstruction": {
                    "synthetic_tensor_shape": [6, 4, 3, 3],
                    "synthetic_seed": 4,
                    "ranks": [1, 2],
                    "seeds": [0, 1, 2],
                    "n_iter_max": 2,
                },
                "paper": {"figures_dir": "out/figures"},
            }
            metadata_path = run_reconstruction(config, root, torch.device("cpu"))
            self.assertTrue((root / "out/reconstruction_summary.csv").is_file())
            self.assertTrue((root / "out/reconstruction_config_used.json").is_file())
            with open(metadata_path) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["failures"], [])
            self.assertEqual(metadata["canonical_cp_seeds"], [0, 1, 2])

    def test_structural_join_matches_reconstruction_and_split_rows(self):
        reconstruction = [{
            "method": "cp", "rank": 2, "seed": 0, "status": "completed",
            "max_unfolding_tail_bound_squared": 0.4,
            "actual_relative_squared_frobenius_error": 0.6,
            "actual_relative_frobenius_error": 0.6 ** 0.5,
        }]
        evaluations = [
            {
                "method": "cp", "rank": 2, "seed": 0, "split": split,
                "present_class_miou": score, "target_layer_parameter_count": 20,
                "dense_target_layer_parameter_count": 100, "full_model_parameter_count": 120,
                "target_layer_macs": 30, "full_model_macs": 300,
            }
            for split, score in (("val", 0.2), ("test", 0.1))
        ]
        row = join_structural_tradeoffs(reconstruction, evaluations)[0]
        self.assertEqual(row["actual_relative_squared_frobenius_error"], 0.6)
        self.assertEqual(row["zero_shot_validation_present_class_miou"], 0.2)
        self.assertEqual(row["zero_shot_test_present_class_miou"], 0.1)
        self.assertEqual(row["target_layer_parameter_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
