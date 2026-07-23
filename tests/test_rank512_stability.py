import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from src.smallnet.experiment import run_rank512_stability
from src.smallnet.rank512_stability import (
    FLOAT64_PRECISION,
    PRIMARY_PRECISION,
    STABILITY_METHOD,
    aggregate_rank512_stability_rows,
    captured_shared_initialization,
    factor_scaling_diagnostics,
    finalize_degeneracy_indicators,
    normalize_rank512_stability_rows,
    scientific_rank512_stability_key,
    verify_cp_reconstruction,
    write_rank512_stability_audit,
    write_rank512_stability_figure,
)


def completed_row(seed, budget, residual, repetition=0, precision=PRIMARY_PRECISION):
    return {
        "method": STABILITY_METHOD,
        "rank": 512,
        "seed": seed,
        "iteration_budget": budget,
        "repetition": repetition,
        "optimization_precision": precision,
        "status": "completed",
        "actual_relative_squared_frobenius_error": residual,
        "actual_relative_frobenius_error": residual**0.5,
        "output_mode_svd_residual_squared": 0.3,
        "max_unfolding_tail_bound_squared": 0.4,
        "initialization_hash_sha256": f"init-{seed}-{precision}",
        "actual_fit_initialization_hash_sha256": f"init-{seed}-{precision}",
        "shared_float32_initialization_hash_sha256": f"shared-{seed}",
        "checkpoint_sha256": "checkpoint",
        "target_tensor_sha256": "target",
        "residual_is_fresh_for_scientific_key": True,
        "residual_verification_passed": True,
        "reconstruction_all_finite": True,
        "factor_degeneracy_detected": False,
        "component_scaling_spread_max": 2.0,
        "component_contribution_sum_to_reconstructed_norm_ratio": 1.2,
        "factor_0_max_abs_value": 1.0,
        "factor_1_max_abs_value": 1.0,
        "factor_2_max_abs_value": 1.0,
        "factor_3_max_abs_value": 1.0,
    }


def fake_cp(weights, factors):
    return SimpleNamespace(weight=SimpleNamespace(weights=weights, factors=factors))


class Rank512DiagnosticMathTests(unittest.TestCase):
    def test_shared_initialization_identity_across_precisions(self):
        conv = nn.Conv2d(3, 5, 2)
        float32_init, shared32, precision32 = captured_shared_initialization(
            conv, 2, 7, "random", torch.device("cpu"), PRIMARY_PRECISION,
            memory_efficient_mttkrp=True,
            mttkrp_rank_chunk_size=2,
            mttkrp_max_explicit_bytes=1024**2,
        )
        float64_init, shared64, precision64 = captured_shared_initialization(
            conv, 2, 7, "random", torch.device("cpu"), FLOAT64_PRECISION,
            memory_efficient_mttkrp=True,
            mttkrp_rank_chunk_size=2,
            mttkrp_max_explicit_bytes=1024**2,
        )
        self.assertEqual(shared32, shared64)
        self.assertNotEqual(precision32, precision64)
        for factor32, factor64 in zip(float32_init[1], float64_init[1]):
            torch.testing.assert_close(factor32.double(), factor64, rtol=0, atol=0)

    def test_repetition_key_numeric_normalization_and_duplicates(self):
        rows = [
            completed_row(0, 200, 0.8),
            completed_row("0", "200", 0.8),
            completed_row("1", "400", 0.9, repetition="1"),
            {**completed_row(0, 200, 0.8), "rank": "all"},
        ]
        with self.assertWarns(RuntimeWarning):
            normalized, rejected, diagnostics = normalize_rank512_stability_rows(rows)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(scientific_rank512_stability_key(normalized[0])[1:], (512, 0, 200, 0, "float32"))
        self.assertIs(normalized[0]["factor_degeneracy_detected"], False)

    def test_direct_residual_recomputation_and_nonfinite_detection(self):
        weights = torch.tensor([1.0, 0.5])
        factors = [torch.randn(size, 2, generator=torch.Generator().manual_seed(size)) for size in (3, 2, 2, 2)]
        approximation = torch.einsum("or,ir,hr,wr,r->oihw", *factors, weights)
        reference = approximation + 0.01
        diagnostic = verify_cp_reconstruction(reference, approximation, fake_cp(weights, factors))
        self.assertTrue(diagnostic["residual_verification_passed"])
        self.assertAlmostEqual(
            diagnostic["actual_relative_frobenius_error"] ** 2,
            diagnostic["actual_relative_squared_frobenius_error"],
        )
        bad_factors = [factor.clone() for factor in factors]
        bad_factors[0][0, 0] = float("nan")
        bad = verify_cp_reconstruction(reference, approximation, fake_cp(weights, bad_factors))
        self.assertFalse(bad["reconstruction_all_finite"])
        self.assertFalse(bad["residual_verification_passed"])

    def test_factor_scaling_and_synthetic_degeneracy_detection(self):
        weights = torch.ones(2)
        factors = [torch.ones(3, 2), torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2)]
        factors[0][:, 0] *= 1e8
        factors[1][:, 0] *= 1e-8
        diagnostic = factor_scaling_diagnostics(
            fake_cp(weights, factors),
            extreme_norm_threshold=1e6,
            scaling_spread_threshold=1e6,
        )
        finalized = finalize_degeneracy_indicators(
            diagnostic,
            {"reconstructed_tensor_norm": 10.0, "dense_tensor_norm": 10.0},
        )
        self.assertTrue(finalized["factor_degeneracy_detected"])
        self.assertGreater(finalized["extremely_large_component_count"], 0)
        self.assertGreater(finalized["extreme_scaling_with_bounded_reconstruction_count"], 0)

    def test_aggregation_detects_later_budget_deterioration_without_repeat_overweighting(self):
        rows = []
        for seed, value in enumerate((0.7807, 0.7808, 0.7810)):
            rows.extend(completed_row(seed, 200, value, repetition=rep) for rep in (0, 1))
        for seed, value in enumerate((0.7900, 0.8000, 0.8100)):
            rows.extend(completed_row(seed, 400, value, repetition=rep) for rep in (0, 1))
        aggregates, seed_rows, diagnostics = aggregate_rank512_stability_rows(rows)
        self.assertEqual(diagnostics, [])
        self.assertEqual(len(seed_rows), 6)
        at_400 = next(row for row in aggregates if row["iteration_budget"] == 400)
        self.assertTrue(at_400["mean_squared_residual_deteriorated_from_previous_budget"])
        self.assertAlmostEqual(at_400["actual_relative_squared_frobenius_error_mean"], 0.8)


class Rank512DiagnosticArtifactTests(unittest.TestCase):
    def test_complete_audit_distinguishes_deterioration_from_improvement(self):
        budgets = [150, 200, 400, 800]
        means = {150: 0.79, 200: 0.78, 400: 0.80, 800: 0.81}
        rows = []
        for seed in (0, 1, 2):
            for budget in budgets:
                repetitions = (0, 1) if budget in (200, 400) else (0,)
                rows.extend(
                    completed_row(seed, budget, means[budget] + seed * 1e-4, repetition)
                    for repetition in repetitions
                )
        aggregates, _, _ = aggregate_rank512_stability_rows(rows)
        keys = {scientific_rank512_stability_key(row) for row in rows}
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit.md"
            _, complete = write_rank512_stability_audit(
                rows,
                aggregates,
                audit,
                budgets=budgets,
                seeds=[0, 1, 2],
                tolerance=1e-5,
                expected_primary_keys=keys,
                expected_float64_keys=set(),
                failures=[],
            )
            text = audit.read_text()
        self.assertTrue(complete)
        self.assertIn("deterioration begin?** `400`", text)
        self.assertIn("largest adjacent increase", text)
        self.assertIn("budget 800 stable, better, or worse?** `worse", text)

    def test_partial_figure_and_audit_are_generated(self):
        rows = [completed_row(seed, budget, 0.8 + seed * 0.01 - budget * 1e-5) for seed in (0, 1) for budget in (150, 200)]
        aggregates, _, _ = aggregate_rank512_stability_rows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = write_rank512_stability_figure(rows, Path(tmp) / "figures")
            audit = Path(tmp) / "audit.md"
            _, complete = write_rank512_stability_audit(
                rows,
                aggregates,
                audit,
                budgets=[150, 200, 400],
                seeds=[0, 1],
                tolerance=1e-5,
                expected_primary_keys={
                    (STABILITY_METHOD, 512, seed, budget, 0, PRIMARY_PRECISION)
                    for seed in (0, 1)
                    for budget in (150, 200, 400)
                },
                expected_float64_keys=set(),
                failures=[],
            )
            self.assertTrue(all(Path(path).is_file() for path in paths))
            self.assertFalse(complete)
            self.assertIn("Status: **incomplete**", audit.read_text())

    def test_synthetic_stage_resumes_without_recomputation_or_canonical_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            diagnostic = canonical / "rank512_stability"
            canonical.mkdir()
            sentinels = [
                canonical / "cp_iteration_sensitivity_summary.csv",
                canonical / "reconstruction_summary.csv",
                canonical / "structural_zero_shot_summary.csv",
            ]
            for sentinel in sentinels:
                sentinel.write_text("sentinel\n")
            config = {
                "experiment_id": "synthetic-rank512-test",
                "output_dir": str(canonical),
                "model": {"target_layer": "synthetic"},
                "reconstruction": {"numerical_tolerance": 1e-5},
                "rank512_stability": {
                    "synthetic_tensor_shape": [5, 3, 2, 2],
                    "synthetic_seed": 9,
                    "rank": 2,
                    "seeds": [0],
                    "iteration_budgets": [1, 2],
                    "primary_repetitions": [0],
                    "repeatability_budgets": [1],
                    "repeatability_repetitions": [0, 1],
                    "float64_seeds": [],
                    "float64_iteration_budgets": [],
                    "output_dir": str(diagnostic),
                    "figures_dir": str(diagnostic / "figures"),
                    "audit_path": str(diagnostic / "audit.md"),
                },
            }
            run_rank512_stability(config, root, torch.device("cpu"))
            summary = diagnostic / "rank512_stability_summary.csv"
            first = list(csv.DictReader(summary.open()))
            self.assertEqual(len(first), 3)
            self.assertEqual(len({tuple(row[field] for field in ("rank", "seed", "iteration_budget", "repetition", "optimization_precision")) for row in first}), 3)
            self.assertTrue(all(path.read_text() == "sentinel\n" for path in sentinels))
            hashes = {row["initialization_hash_sha256"] for row in first}
            self.assertEqual(len(hashes), 1)
            repeated = [row for row in first if row["iteration_budget"] == "1"]
            self.assertEqual(len(repeated), 2)
            self.assertEqual(
                {row["actual_relative_squared_frobenius_error"] for row in repeated}.__len__(),
                1,
            )
            with mock.patch(
                "src.smallnet.experiment.fit_rank512_stability_row",
                side_effect=AssertionError("completed keys must be skipped"),
            ):
                run_rank512_stability(config, root, torch.device("cpu"))
            second = list(csv.DictReader(summary.open()))
            self.assertEqual(len(second), 3)
            self.assertTrue(all(path.read_text() == "sentinel\n" for path in sentinels))

    def test_failed_later_budget_is_retained_and_partial_artifacts_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "diag"
            config = {
                "experiment_id": "failure-test",
                "output_dir": str(root / "canonical"),
                "model": {"target_layer": "synthetic"},
                "reconstruction": {"numerical_tolerance": 1e-5},
                "rank512_stability": {
                    "synthetic_tensor_shape": [4, 3, 2, 2],
                    "rank": 2,
                    "seeds": [0],
                    "iteration_budgets": [1, 2],
                    "primary_repetitions": [0],
                    "repeatability_budgets": [],
                    "repeatability_repetitions": [],
                    "float64_seeds": [],
                    "float64_iteration_budgets": [],
                    "output_dir": str(output),
                    "figures_dir": str(output / "figures"),
                    "audit_path": str(output / "audit.md"),
                },
            }
            from src.smallnet.experiment import fit_rank512_stability_row as real_fit

            def fail_budget(*args, **kwargs):
                if kwargs["budget"] == 2:
                    raise RuntimeError("simulated later-budget failure")
                return real_fit(*args, **kwargs)

            with mock.patch("src.smallnet.experiment.fit_rank512_stability_row", side_effect=fail_budget):
                run_rank512_stability(config, root, torch.device("cpu"))
            rows = list(csv.DictReader((output / "rank512_stability_summary.csv").open()))
            self.assertEqual({row["status"] for row in rows}, {"completed", "failed"})
            self.assertTrue((output / "figures" / "rank512_stability.png").is_file())
            self.assertTrue((output / "audit.md").is_file())


if __name__ == "__main__":
    unittest.main()
