import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.smallnet.data import ClassDefinition, UnknownMaskColorError, rgb_mask_to_index, strict_unknown_colors_from_config
from src.smallnet.mask_forensics import (
    aggregate_unknown_colors_by_file,
    compare_camvid_masks,
    inspect_mask_forensics,
)


def write_class_dict(path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "r", "g", "b"])
        writer.writerows([("black", 0, 0, 0), ("red", 255, 0, 0), ("Void", 1, 1, 1)])


def write_mask(path, pixels):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB").save(path)


class MaskForensicsTests(unittest.TestCase):
    def test_aggregate_unknown_colors_by_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            write_mask(path, [[[0, 0, 0], [3, 3, 3]], [[3, 3, 3], [255, 0, 0]]])
            definitions = [
                ClassDefinition(0, "black", (0, 0, 0)),
                ClassDefinition(1, "red", (255, 0, 0)),
            ]
            rows = aggregate_unknown_colors_by_file([path], definitions)
            self.assertEqual(rows[0]["total_unknown_pixels"], 2)
            self.assertEqual(rows[0]["distinct_unknown_rgb_values"], 1)
            self.assertEqual(rows[0]["bounding_box_ymin_xmin_ymax_xmax"], [0, 0, 1, 1])
            self.assertEqual(rows[0]["connected_region_count_4_neighbor"], 2)
            self.assertFalse(rows[0]["all_unknown_pixels_in_one_connected_region"])

    def test_forensic_metadata_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_dict = root / "class_dict.csv"
            mask = root / "mask.png"
            write_class_dict(class_dict)
            write_mask(mask, [[[0, 0, 0], [127, 0, 0]], [[255, 0, 0], [1, 1, 1]]])
            result = inspect_mask_forensics(mask, class_dict)
            self.assertEqual(result["image_mode"], "RGB")
            self.assertEqual(result["file_format"], "PNG")
            self.assertEqual(len(result["sha256"]), 64)
            self.assertEqual(result["unknown_summary"]["total_unknown_pixels"], 1)
            self.assertTrue(result["colors_and_counts_rgb"])

    def test_compare_identical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_dict = root / "class_dict.csv"
            mask = root / "mask.png"
            write_class_dict(class_dict)
            write_mask(mask, [[[0, 0, 0], [255, 0, 0]]])
            result = compare_camvid_masks(mask, mask, class_dict)
            self.assertEqual(result["exact_differing_pixel_count"], 0)
            self.assertTrue(result["replacing_file_would_eliminate_all_unknown_colors"])

    def test_compare_small_number_of_differing_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_dict = root / "class_dict.csv"
            current, reference = root / "current.png", root / "reference.png"
            write_class_dict(class_dict)
            write_mask(current, [[[3, 3, 3], [0, 0, 0]]])
            write_mask(reference, [[[255, 0, 0], [0, 0, 0]]])
            result = compare_camvid_masks(current, reference, class_dict)
            self.assertEqual(result["exact_differing_pixel_count"], 1)
            self.assertEqual(result["rgb_transition_summary"][0]["current_rgb"], [3, 3, 3])
            self.assertTrue(result["replacing_file_would_eliminate_all_unknown_colors"])

    def test_strict_mode_fails_and_map_to_ignore_changes_only_unknown(self):
        definitions = [ClassDefinition(0, "black", (0, 0, 0)), ClassDefinition(1, "red", (255, 0, 0))]
        mask = np.asarray([[[3, 3, 3], [0, 0, 0], [255, 0, 0]]], dtype=np.uint8)
        with self.assertRaises(UnknownMaskColorError):
            rgb_mask_to_index(mask, definitions, strict_unknown_colors=True)
        mapped, unknown = rgb_mask_to_index(
            mask, definitions, strict_unknown_colors=False, unknown_color_ignore_index=30
        )
        np.testing.assert_array_equal(mapped, [[30, 0, 1]])
        self.assertEqual(sum(item["pixel_count"] for item in unknown), 1)

    def test_nearest_color_policy_is_not_implemented(self):
        with self.assertRaisesRegex(ValueError, "nearest-color assignment"):
            strict_unknown_colors_from_config({"unknown_color_policy": "nearest"})

    def test_map_to_ignore_requires_configured_ignore_index(self):
        config = {
            "unknown_color_policy": "map_to_ignore",
            "strict_unknown_colors": False,
            "ignore_index": 30,
            "unknown_color_ignore_index": 30,
        }
        self.assertFalse(strict_unknown_colors_from_config(config))
        with self.assertRaisesRegex(ValueError, "explicitly configured ignore_index"):
            strict_unknown_colors_from_config({**config, "unknown_color_ignore_index": 0})


if __name__ == "__main__":
    unittest.main()
