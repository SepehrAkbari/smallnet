import math
import unittest

from src.smallnet.final_cp_paper import (
    FINITE_FIELDS,
    ITERATIONS,
    METHOD,
    RANKS,
    SEEDS,
    aggregate_cp_rows,
    validate_cp_rows,
)


def synthetic_row(rank, seed):
    row = {
        "method": METHOD,
        "rank": str(rank),
        "seed": str(seed),
        "iteration_budget": str(ITERATIONS),
        "status": "completed",
        "same_fitted_factors_reused_for_all_metrics": "True",
        "final_factor_hash_sha256": f"factor-{rank}-{seed}",
        "factor_hash_after_reconstruction": f"factor-{rank}-{seed}",
        "factor_hash_after_activation_distortion": f"factor-{rank}-{seed}",
        "factor_hash_after_zero_shot": f"factor-{rank}-{seed}",
        "residual_verification_passed": "True",
        "factor_diagnostics_finite": "True",
        "factor_degeneracy_detected": "False",
        "checkpoint_sha256": "checkpoint",
        "dataset_validation_report_sha256": "dataset",
        "target_tensor_sha256": "tensor",
        "actual_relative_squared_frobenius_error": str(rank / 1000 + seed / 10000),
        "activation_normalized_squared_error": str(rank / 2000 + seed / 10000),
        "activation_cosine_similarity": str(0.8 + rank / 10000),
        "validation_present_class_miou": str(0.2 + rank / 10000 + seed / 1000),
        "test_present_class_miou": str(0.18 + rank / 10000 + seed / 1000),
        "target_layer_parameter_count": str(rank * 10),
        "target_layer_parameter_ratio": "0.1",
        "full_model_parameter_count": str(1000 + rank),
        "target_layer_macs": str(rank * 100),
        "full_model_macs": str(10000 + rank),
        "compression_factor": "10",
    }
    for field in FINITE_FIELDS:
        row.setdefault(field, "0.1")
    return row


class FinalCpPaperTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            synthetic_row(rank, seed) for rank in RANKS for seed in SEEDS
        ]

    def test_exact_protocol_and_population_aggregation(self):
        validated = validate_cp_rows(self.rows)
        self.assertEqual(validated["row_count"], 15)
        aggregates = aggregate_cp_rows(validated["rows"])
        self.assertEqual([row["rank"] for row in aggregates], list(RANKS))
        self.assertTrue(all(row["seed_count"] == 3 for row in aggregates))
        expected_std = math.sqrt(2 / 3) / 10000
        self.assertAlmostEqual(
            aggregates[0]["relative_squared_weight_error_population_std"],
            expected_std,
        )

    def test_duplicate_or_wrong_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "15 unique"):
            validate_cp_rows([*self.rows, self.rows[0]])
        wrong = [dict(row) for row in self.rows]
        wrong[0]["iteration_budget"] = "100"
        with self.assertRaisesRegex(ValueError, "200"):
            validate_cp_rows(wrong)

    def test_factor_stage_mismatch_is_rejected(self):
        bad = [dict(row) for row in self.rows]
        bad[0]["factor_hash_after_zero_shot"] = "different"
        with self.assertRaisesRegex(ValueError, "factor hash mismatch"):
            validate_cp_rows(bad)

    def test_nonfinite_scientific_value_is_rejected(self):
        bad = [dict(row) for row in self.rows]
        bad[0]["activation_normalized_squared_error"] = "nan"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_cp_rows(bad)


if __name__ == "__main__":
    unittest.main()
