import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import tltorch

from src.dataset import CamVidDataset
from src.utils import fast_hist, summarize_hist


ROOT = Path(__file__).resolve().parents[1]


def load_profile_module():
    path = ROOT / "scripts" / "profile.py"
    spec = importlib.util.spec_from_file_location("smallnet_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricTests(unittest.TestCase):
    def test_summary_metrics_from_confusion_matrix(self):
        labels = np.array([0, 1, 1, 2])
        preds = np.array([0, 1, 2, 2])
        hist = fast_hist(labels, preds, 3)
        summary = summarize_hist(hist, class_names=["a", "b", "c"])

        self.assertAlmostEqual(summary["pixel_accuracy"], 0.75)
        self.assertAlmostEqual(summary["mean_iou_all_classes"], (1.0 + 0.5 + 0.5) / 3.0)
        self.assertAlmostEqual(summary["mean_iou_present_classes"], (1.0 + 0.5 + 0.5) / 3.0)
        self.assertAlmostEqual(summary["frequency_weighted_iou"], 0.625)
        self.assertEqual(summary["per_class"][1]["class_name"], "b")

    def test_summary_can_exclude_class_from_mean_iou(self):
        hist = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ]
        )
        summary = summarize_hist(hist, class_names=["a", "b", "void"], exclude_indices=[2])

        self.assertAlmostEqual(summary["mean_iou_all_classes"], (1.0 + 0.5) / 2.0)
        self.assertTrue(summary["per_class"][2]["excluded"])

    def test_ignore_index_removes_ground_truth_class(self):
        labels = np.array([0, 1, 2])
        preds = np.array([0, 2, 2])
        hist = fast_hist(labels, preds, 3, ignore_index=2)

        self.assertEqual(hist.sum(), 2)
        self.assertEqual(hist[2].sum(), 0)


class DatasetTests(unittest.TestCase):
    def test_color_mapping_and_mask_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()

            with open(root / "class_dict.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "r", "g", "b"])
                writer.writerow(["black", 0, 0, 0])
                writer.writerow(["red", 255, 0, 0])

            Image.new("RGB", (2, 2), color=(10, 20, 30)).save(images / "sample.png")
            mask = Image.fromarray(
                np.array(
                    [
                        [[0, 0, 0], [255, 0, 0]],
                        [[255, 0, 0], [0, 0, 0]],
                    ],
                    dtype=np.uint8,
                )
            )
            mask.save(masks / "sample.png")

            dataset = CamVidDataset(images, masks, root / "class_dict.csv")
            _, mask_tensor = dataset[0]

            self.assertEqual(tuple(mask_tensor.shape), (352, 480))
            self.assertEqual(set(mask_tensor.unique().tolist()), {0, 1})


class ProfileMathTests(unittest.TestCase):
    def test_dense_conv_macs_formula(self):
        profile = load_profile_module()
        conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        macs = profile.conv2d_macs(conv, (1, 3, 10, 10), (1, 5, 10, 10))
        self.assertEqual(macs, 1 * 10 * 10 * 5 * 3 * 3 * 3)

    def test_cp_conv_macs_formula_and_rank(self):
        profile = load_profile_module()
        conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        layer = tltorch.FactorizedConv.from_conv(
            conv,
            rank=2,
            factorization="cp",
            decomposition_kwargs={"init": "random", "n_iter_max": 0},
        )
        macs = profile.cp_conv2d_macs(layer, (1, 3, 10, 10), (1, 5, 10, 10))

        expected = (
            1 * 10 * 10 * 3 * 2
            + 1 * 10 * 10 * 2 * 3
            + 1 * 10 * 10 * 2 * 3
            + 1 * 10 * 10 * 2 * 5
            + 1 * 10 * 10 * 2
        )
        self.assertEqual(profile.get_cp_rank(layer), 2)
        self.assertEqual(macs, expected)

    def test_checkpoint_parameter_counts_when_available(self):
        expected = {
            "model/best_model.pth": 134_391_648,
            "model/finetuned_rank_256.pth": 32_814_688,
            "model/finetuned_rank_128.pth": 32_222_944,
            "model/finetuned_rank_64.pth": 31_927_072,
        }
        missing = [path for path in expected if not (ROOT / path).exists()]
        if missing:
            self.skipTest(f"Local checkpoint files are absent: {missing}")

        for rel_path, count in expected.items():
            state_dict = torch.load(ROOT / rel_path, map_location="cpu")
            actual = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
            self.assertEqual(actual, count)


if __name__ == "__main__":
    unittest.main()
