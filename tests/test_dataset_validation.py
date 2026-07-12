import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.smallnet.data import (
    CamVidDataset,
    ClassDefinition,
    ClassDictionaryError,
    DatasetPairingError,
    UnknownMaskColorError,
    parse_camvid_class_dict,
    rgb_mask_to_index,
    validate_class_definitions,
)
from src.smallnet.experiment import run_dataset_validation


ROOT = Path(__file__).resolve().parents[1]


def write_class_dict(path, rows=None):
    rows = rows or [
        ("black", 0, 0, 0),
        ("red", 255, 0, 0),
        ("Void", 1, 1, 1),
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "r", "g", "b"])
        writer.writerows(rows)


def write_rgb_image(path, size=(2, 2), color=(10, 20, 30)):
    Image.new("RGB", size, color=color).save(path)


def write_mask(path, pixels=None):
    pixels = pixels if pixels is not None else [
        [[0, 0, 0], [255, 0, 0]],
        [[1, 1, 1], [0, 0, 0]],
    ]
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(path)


def make_split(root, split="train"):
    images = root / split
    masks = root / f"{split}_labels"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    return images, masks


def make_config(root, out):
    return {
        "experiment_id": "unit_dataset_validation",
        "output_dir": str(out),
        "dataset": {
            "name": "CamVid",
            "source_description": "synthetic unit fixture",
            "source_url": None,
            "split_convention": "unit train split",
            "root": str(root),
            "class_dict_path": str(root / "class_dict.csv"),
            "image_size": [2, 2],
            "num_classes": 3,
            "splits": ["train"],
            "ignore_index": 2,
            "ignore_class_name": "Void",
            "required_class_at_index": {"index": 2, "name": "Void"},
            "pairing": {"image_key": "stem", "mask_key": "stem", "mask_suffix_to_remove": "_L"},
            "unknown_color_policy": "strict",
            "strict_unknown_colors": True,
        },
    }


class DatasetPairingTests(unittest.TestCase):
    def test_valid_exact_stem_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample.png")

            dataset = CamVidDataset(images, masks, root / "class_dict.csv", image_size=(2, 2), num_classes=3)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.pairs[0].key, "sample")
            self.assertEqual(dataset.pairs[0].image_path.name, "sample.png")
            self.assertEqual(dataset.pairs[0].mask_path.name, "sample.png")

    def test_valid_pairing_with_configured_mask_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample_L.png")

            dataset = CamVidDataset(
                images,
                masks,
                root / "class_dict.csv",
                image_size=(2, 2),
                mask_suffix_to_remove="_L",
                num_classes=3,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.pairs[0].key, "sample")

    def test_missing_mask_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")

            with self.assertRaisesRegex(DatasetPairingError, "images have no matching mask"):
                CamVidDataset(images, masks, root / "class_dict.csv", image_size=(2, 2), num_classes=3)

    def test_unmatched_mask_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample.png")
            write_mask(masks / "extra.png")

            with self.assertRaisesRegex(DatasetPairingError, "masks have no matching image"):
                CamVidDataset(images, masks, root / "class_dict.csv", image_size=(2, 2), num_classes=3)

    def test_duplicate_normalized_key_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_rgb_image(images / "sample.jpg")
            write_mask(masks / "sample.png")

            with self.assertRaisesRegex(DatasetPairingError, "duplicate image normalized keys"):
                CamVidDataset(images, masks, root / "class_dict.csv", image_size=(2, 2), num_classes=3)


class StrictRgbTests(unittest.TestCase):
    def test_unknown_rgb_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = make_split(root)
            write_class_dict(root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample.png", pixels=[[[3, 3, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
            dataset = CamVidDataset(images, masks, root / "class_dict.csv", image_size=(2, 2), num_classes=3)

            with self.assertRaises(UnknownMaskColorError) as ctx:
                dataset[0]

            self.assertEqual(ctx.exception.unknown_colors[0]["rgb"], [3, 3, 3])
            self.assertEqual(ctx.exception.unknown_colors[0]["pixel_count"], 1)
            self.assertEqual(ctx.exception.unknown_colors[0]["example_coordinates_yx"], [[0, 0]])

    def test_unknown_rgb_is_not_mapped_to_class_zero(self):
        class_definitions = [
            ClassDefinition(index=0, name="black", rgb=(0, 0, 0)),
            ClassDefinition(index=1, name="red", rgb=(255, 0, 0)),
        ]
        mask = np.array([[[3, 3, 3], [0, 0, 0]]], dtype=np.uint8)
        mapped, unknown = rgb_mask_to_index(
            mask,
            class_definitions,
            strict_unknown_colors=False,
            unknown_color_ignore_index=255,
        )

        self.assertEqual(mapped[0, 0], 255)
        self.assertEqual(mapped[0, 1], 0)
        self.assertEqual(unknown[0]["rgb"], [3, 3, 3])

    def test_duplicate_rgb_entries_in_class_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_path = root / "class_dict.csv"
            write_class_dict(class_path, rows=[("a", 0, 0, 0), ("b", 0, 0, 0)])

            with self.assertRaisesRegex(ClassDictionaryError, "RGB values must be unique"):
                validate_class_definitions(parse_camvid_class_dict(class_path), expected_num_classes=2)

    def test_incorrect_void_index_or_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_path = root / "class_dict.csv"
            write_class_dict(class_path, rows=[("black", 0, 0, 0), ("Voidish", 1, 1, 1), ("red", 255, 0, 0)])

            with self.assertRaisesRegex(ClassDictionaryError, "required class at index 1"):
                validate_class_definitions(
                    parse_camvid_class_dict(class_path),
                    expected_num_classes=3,
                    required_class_at_index={"index": 1, "name": "Void"},
                )


class DatasetValidationReportTests(unittest.TestCase):
    def test_valid_end_to_end_validation_report_and_class_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_root = tmp / "CamVid"
            out = tmp / "results"
            images, masks = make_split(data_root)
            write_class_dict(data_root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample_L.png")
            config = make_config(data_root, out)

            report_path = run_dataset_validation(config, ROOT, device=torch.device("cpu"))

            self.assertTrue(report_path.is_file())
            self.assertTrue((out / "dataset_validation_summary.csv").is_file())
            self.assertTrue((out / "dataset_class_counts.csv").is_file())
            with open(report_path) as f:
                report = json.load(f)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["pairing_rule"]["mask_suffix_to_remove"], "_L")
            self.assertEqual(report["splits"][0]["matched_pair_count"], 1)

    def test_class_count_output_on_small_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_root = tmp / "CamVid"
            out = tmp / "results"
            images, masks = make_split(data_root)
            write_class_dict(data_root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample_L.png")
            config = make_config(data_root, out)

            run_dataset_validation(config, ROOT, device=torch.device("cpu"))

            with open(out / "dataset_class_counts.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            counts = {row["class_name"]: int(row["pixel_count"]) for row in rows}
            self.assertEqual(counts["black"], 2)
            self.assertEqual(counts["red"], 1)
            self.assertEqual(counts["Void"], 1)

    def test_failure_report_generation_and_nonzero_command_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_root = tmp / "CamVid"
            out = tmp / "results"
            images, masks = make_split(data_root)
            write_class_dict(data_root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample_L.png", pixels=[[[3, 3, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
            config_path = tmp / "config.json"
            with open(config_path, "w") as f:
                json.dump(make_config(data_root, out), f)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_experiment.py"),
                    "--config",
                    str(config_path),
                    "--stage",
                    "validate-data",
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            report_path = out / "dataset_validation_report.json"
            self.assertTrue(report_path.is_file())
            with open(report_path) as f:
                report = json.load(f)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["splits"][0]["unknown_rgb_values"][0]["rgb"], [3, 3, 3])

    def test_map_to_ignore_report_records_exact_mapped_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_root = tmp / "CamVid"
            out = tmp / "results"
            images, masks = make_split(data_root)
            write_class_dict(data_root / "class_dict.csv")
            write_rgb_image(images / "sample.png")
            write_mask(masks / "sample_L.png", pixels=[[[3, 3, 3], [0, 0, 0]], [[3, 3, 3], [255, 0, 0]]])
            config = make_config(data_root, out)
            config["dataset"].update(
                {
                    "unknown_color_policy": "map_to_ignore",
                    "strict_unknown_colors": False,
                    "unknown_color_ignore_index": 2,
                }
            )

            report_path = run_dataset_validation(config, ROOT, device=torch.device("cpu"))

            with open(report_path) as f:
                report = json.load(f)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["unknown_color_policy"], "map_to_ignore")
            self.assertEqual(report["unknown_color_ignore_index"], 2)
            self.assertEqual(report["unknown_pixels_mapped_to_ignore"], 2)


if __name__ == "__main__":
    unittest.main()
