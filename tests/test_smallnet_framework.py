import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.smallnet.config import load_config
from src.smallnet.diagnostics import (
    cp_tensor_from_factors,
    matrix_rank_by_mode,
    rank_energy_diagnostic,
    rank_for_energy,
)
from src.smallnet.factorization import MatrixLowRankConv2d
from src.smallnet.modules import get_module, set_module
from src.smallnet.profiling import manual_macs
from src.smallnet.results import load_manifest, save_manifest


class DiagnosticTests(unittest.TestCase):
    def test_cp_rank_bounds_unfolding_ranks(self):
        torch.manual_seed(0)
        factors = [
            torch.randn(4, 3),
            torch.randn(5, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ]
        tensor = cp_tensor_from_factors(factors)
        for mode in range(4):
            self.assertLessEqual(matrix_rank_by_mode(tensor, mode, tol=1e-4), 3)

    def test_rank_energy_threshold_and_tail_proxy(self):
        weight = torch.zeros(4, 4, 1, 1)
        weight[:, :, 0, 0] = torch.diag(torch.tensor([3.0, 1.0, 0.5, 0.5]))
        diagnostic = rank_energy_diagnostic(weight, modes=[0], thresholds=[0.9], fixed_ranks=[1, 2])

        cumulative = diagnostic["modes"]["0"]["cumulative_energy"]
        self.assertEqual(rank_for_energy(cumulative, 0.9), 2)
        self.assertEqual(diagnostic["rank_energy_thresholds"]["0.900"], 2)
        self.assertGreater(diagnostic["cp_necessary_tail_energy"]["1"], 0.0)


class ModuleReplacementTests(unittest.TestCase):
    def test_dotted_get_and_set_for_nested_modules(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.Sequential(nn.ReLU(), nn.Conv2d(4, 5, 1)),
        )
        self.assertIsInstance(get_module(model, "1.1"), nn.Conv2d)
        set_module(model, "1.1", nn.Identity())
        self.assertIsInstance(get_module(model, "1.1"), nn.Identity)


class FactorizationTests(unittest.TestCase):
    def test_matrix_low_rank_conv_shape_and_profile(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        low_rank = MatrixLowRankConv2d.from_conv(conv, rank=2)
        out = low_rank(torch.randn(1, 3, 8, 8))
        self.assertEqual(tuple(out.shape), (1, 5, 8, 8))

        macs, records = manual_macs(low_rank, (1, 3, 8, 8), device=torch.device("cpu"))
        self.assertEqual(len(records), 2)
        self.assertGreater(macs, 0)


class ConfigAndManifestTests(unittest.TestCase):
    def test_config_loading_and_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config_path = tmp / "config.json"
            with open(config_path, "w") as f:
                json.dump({"experiment_id": "unit"}, f)

            config = load_config(config_path)
            self.assertEqual(config["experiment_id"], "unit")
            self.assertEqual(config["_config_path"], str(config_path))

            manifest_path = save_manifest(tmp / "manifest.json", {"kind": "unit", "values": [1, 2]})
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest["schema"], "smallnet.manifest.v1")
            self.assertEqual(manifest["kind"], "unit")
            self.assertIn("environment", manifest)


if __name__ == "__main__":
    unittest.main()
