'''
Dataset loaders and validation helpers for CamVid and VOC experiments.
'''

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ClassDefinition:
    index: int
    name: str
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class CamVidPair:
    key: str
    image_path: Path
    mask_path: Path


class ClassDictionaryError(ValueError):
    pass


class DatasetPairingError(ValueError):
    pass


class UnknownMaskColorError(ValueError):
    def __init__(self, mask_path, unknown_colors):
        self.mask_path = str(mask_path)
        self.unknown_colors = list(unknown_colors)
        details = ", ".join(
            f"rgb={item['rgb']} count={item['pixel_count']}" for item in self.unknown_colors[:5]
        )
        super().__init__(f"Unknown RGB values in mask {self.mask_path}: {details}")


VOC_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def is_image_file(path):
    path = Path(path)
    return path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".") and not path.name.startswith("._")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_camvid_class_dict(class_dict_path):
    class_dict_path = Path(class_dict_path)
    definitions = []
    with open(class_dict_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ClassDictionaryError(f"Class dictionary is empty: {class_dict_path}")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        name_col = columns.get("name") or columns.get("class") or columns.get("class_name")
        index_col = columns.get("index") or columns.get("class_index")
        missing = [col for col in ["r", "g", "b"] if col not in columns]
        if name_col is None or missing:
            raise ClassDictionaryError(
                f"Class dictionary must contain name,r,g,b columns: {class_dict_path}"
            )
        for row_number, row in enumerate(reader):
            try:
                index = int(row[index_col].strip()) if index_col else row_number
                name = row[name_col].strip()
                rgb = (
                    int(row[columns["r"]].strip()),
                    int(row[columns["g"]].strip()),
                    int(row[columns["b"]].strip()),
                )
            except Exception as exc:
                raise ClassDictionaryError(
                    f"Invalid class dictionary row {row_number + 2} in {class_dict_path}: {row}"
                ) from exc
            definitions.append(ClassDefinition(index=index, name=name, rgb=rgb))
    return definitions


def load_camvid_class_names(class_dict_path):
    return [definition.name for definition in parse_camvid_class_dict(class_dict_path)]


def validate_class_definitions(
    class_definitions,
    *,
    expected_num_classes=None,
    ignore_index=None,
    ignore_class_name=None,
    allow_non_contiguous_indices=False,
    required_class_at_index=None,
):
    errors = []
    indices = [definition.index for definition in class_definitions]
    names = [definition.name for definition in class_definitions]
    rgbs = [definition.rgb for definition in class_definitions]

    if expected_num_classes is not None and len(class_definitions) != int(expected_num_classes):
        errors.append(
            f"class_dict has {len(class_definitions)} classes but dataset.num_classes is {expected_num_classes}"
        )
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        errors.append(f"class indices must be unique; duplicates: {duplicates}")
    if not allow_non_contiguous_indices:
        expected = list(range(len(indices)))
        if sorted(indices) != expected:
            errors.append(f"class indices must be contiguous 0..{len(indices) - 1}; got {sorted(indices)}")
    duplicate_rgbs = [rgb for rgb, count in Counter(rgbs).items() if count > 1]
    if duplicate_rgbs:
        errors.append(f"RGB values must be unique; duplicates: {duplicate_rgbs}")
    empty_names = [definition.index for definition in class_definitions if not definition.name]
    if empty_names:
        errors.append(f"class names must be nonempty; empty names at indices: {empty_names}")
    by_index = {definition.index: definition for definition in class_definitions}
    if ignore_index is not None:
        ignore_index = int(ignore_index)
        if ignore_index not in by_index:
            errors.append(f"dataset.ignore_index {ignore_index} is not in class index range")
        elif ignore_class_name is not None and by_index[ignore_index].name != ignore_class_name:
            errors.append(
                f"class_names[{ignore_index}] is {by_index[ignore_index].name!r}, "
                f"expected dataset.ignore_class_name {ignore_class_name!r}"
            )
    if required_class_at_index:
        required_index = int(required_class_at_index["index"])
        required_name = required_class_at_index["name"]
        if required_index not in by_index:
            errors.append(f"required class index {required_index} is missing")
        elif by_index[required_index].name != required_name:
            errors.append(
                f"required class at index {required_index} is {by_index[required_index].name!r}, "
                f"expected {required_name!r}"
            )
    if errors:
        raise ClassDictionaryError("; ".join(errors))
    return True


def class_validation_options(dataset_cfg):
    return {
        "expected_num_classes": dataset_cfg.get("num_classes"),
        "ignore_index": dataset_cfg.get("ignore_index"),
        "ignore_class_name": dataset_cfg.get("ignore_class_name"),
        "allow_non_contiguous_indices": bool(dataset_cfg.get("allow_non_contiguous_class_indices", False)),
        "required_class_at_index": dataset_cfg.get("required_class_at_index"),
    }


def pairing_rule_from_config(dataset_cfg):
    pairing_cfg = dataset_cfg.get("pairing", {})
    return {
        "image_key": pairing_cfg.get("image_key", "stem"),
        "mask_key": pairing_cfg.get("mask_key", "stem"),
        "mask_suffix_to_remove": pairing_cfg.get("mask_suffix_to_remove"),
    }


def strict_unknown_colors_from_config(dataset_cfg):
    policy = dataset_cfg.get("unknown_color_policy", "strict")
    if policy not in {"strict", "map_to_ignore"}:
        raise ValueError(
            "dataset.unknown_color_policy must be 'strict' or 'map_to_ignore'; "
            "nearest-color assignment and class-zero fallback are not implemented"
        )
    if "strict_unknown_colors" in dataset_cfg:
        configured = bool(dataset_cfg["strict_unknown_colors"])
        if configured != (policy == "strict"):
            raise ValueError("dataset.strict_unknown_colors conflicts with dataset.unknown_color_policy")
    if policy == "map_to_ignore":
        ignore_index = dataset_cfg.get("ignore_index")
        mapping_index = dataset_cfg.get("unknown_color_ignore_index")
        if ignore_index is None or mapping_index is None or int(mapping_index) != int(ignore_index):
            raise ValueError(
                "map_to_ignore requires dataset.unknown_color_ignore_index to equal the explicitly configured ignore_index"
            )
    return policy == "strict"


def normalize_pairing_key(path, *, kind, mask_suffix_to_remove=None):
    path = Path(path)
    key = path.stem
    if kind == "mask" and mask_suffix_to_remove:
        if key.endswith(mask_suffix_to_remove):
            key = key[: -len(mask_suffix_to_remove)]
    return key


def build_key_map(paths, *, kind, mask_suffix_to_remove=None):
    key_to_paths = defaultdict(list)
    for path in sorted(paths, key=lambda item: item.name):
        key = normalize_pairing_key(path, kind=kind, mask_suffix_to_remove=mask_suffix_to_remove)
        key_to_paths[key].append(path)
    duplicates = {
        key: [str(path) for path in values]
        for key, values in sorted(key_to_paths.items())
        if len(values) > 1
    }
    unique = {key: values[0] for key, values in key_to_paths.items() if len(values) == 1}
    return unique, duplicates


def resolve_camvid_pairs(images_dir, masks_dir, *, mask_suffix_to_remove=None, raise_on_error=True):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"CamVid image split is missing: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"CamVid label split is missing: {masks_dir}")

    image_files = [path for path in images_dir.iterdir() if is_image_file(path)]
    mask_files = [path for path in masks_dir.iterdir() if is_image_file(path)]
    image_map, duplicate_image_keys = build_key_map(image_files, kind="image")
    mask_map, duplicate_mask_keys = build_key_map(
        mask_files,
        kind="mask",
        mask_suffix_to_remove=mask_suffix_to_remove,
    )
    image_keys = set(image_map)
    mask_keys = set(mask_map)
    missing_masks = [
        {"key": key, "image": str(image_map[key])}
        for key in sorted(image_keys - mask_keys)
        if key not in duplicate_image_keys
    ]
    unmatched_masks = [
        {"key": key, "mask": str(mask_map[key])}
        for key in sorted(mask_keys - image_keys)
        if key not in duplicate_mask_keys
    ]
    duplicate_or_ambiguous = {
        "image_keys": duplicate_image_keys,
        "mask_keys": duplicate_mask_keys,
        "ambiguous_keys": {},
    }
    pairs = [
        CamVidPair(key=key, image_path=image_map[key], mask_path=mask_map[key])
        for key in sorted(image_keys & mask_keys)
    ]
    errors = []
    if missing_masks:
        errors.append(f"{len(missing_masks)} images have no matching mask")
    if unmatched_masks:
        errors.append(f"{len(unmatched_masks)} masks have no matching image")
    if duplicate_image_keys:
        errors.append(f"{len(duplicate_image_keys)} duplicate image normalized keys")
    if duplicate_mask_keys:
        errors.append(f"{len(duplicate_mask_keys)} duplicate mask normalized keys")
    if errors and raise_on_error:
        raise DatasetPairingError(
            "; ".join(errors)
            + f" for images_dir={images_dir}, masks_dir={masks_dir}, "
            f"mask_suffix_to_remove={mask_suffix_to_remove!r}"
        )
    return {
        "pairs": pairs,
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "missing_masks": missing_masks,
        "unmatched_masks": unmatched_masks,
        "duplicate_or_ambiguous": duplicate_or_ambiguous,
        "errors": errors,
    }


def encode_rgb(mask_np):
    mask_np = mask_np.astype(np.uint32, copy=False)
    return (mask_np[:, :, 0] << 16) + (mask_np[:, :, 1] << 8) + mask_np[:, :, 2]


def decode_rgb_code(code):
    code = int(code)
    return [(code >> 16) & 255, (code >> 8) & 255, code & 255]


def color_lookup(class_definitions):
    return {
        (definition.rgb[0] << 16) + (definition.rgb[1] << 8) + definition.rgb[2]: definition.index
        for definition in class_definitions
    }


def unknown_colors_in_mask(mask_np, class_definitions, mask_path="", max_examples=5):
    encoded = encode_rgb(mask_np)
    unique_codes, counts = np.unique(encoded, return_counts=True)
    known = color_lookup(class_definitions)
    unknown = []
    for code, count in zip(unique_codes.tolist(), counts.tolist()):
        if int(code) in known:
            continue
        coords = np.argwhere(encoded == int(code))[:max_examples]
        unknown.append(
            {
                "rgb": decode_rgb_code(code),
                "pixel_count": int(count),
                "mask": str(mask_path),
                "example_coordinates_yx": [[int(y), int(x)] for y, x in coords],
            }
        )
    return unknown


def rgb_mask_to_index(
    mask_np,
    class_definitions,
    *,
    mask_path="",
    strict_unknown_colors=True,
    unknown_color_ignore_index=None,
    max_unknown_examples=5,
):
    encoded = encode_rgb(mask_np)
    unique_codes, inverse = np.unique(encoded, return_inverse=True)
    lookup = color_lookup(class_definitions)
    mapped_values = np.empty(len(unique_codes), dtype=np.int64)
    unknown = []
    for i, code in enumerate(unique_codes.tolist()):
        if int(code) in lookup:
            mapped_values[i] = lookup[int(code)]
            continue
        if strict_unknown_colors:
            mapped_values[i] = -1
        else:
            if unknown_color_ignore_index is None:
                raise ValueError("unknown_color_ignore_index is required when strict_unknown_colors=False")
            mapped_values[i] = int(unknown_color_ignore_index)
        coords = np.argwhere(encoded == int(code))[:max_unknown_examples]
        unknown.append(
            {
                "rgb": decode_rgb_code(code),
                "pixel_count": int((encoded == int(code)).sum()),
                "mask": str(mask_path),
                "example_coordinates_yx": [[int(y), int(x)] for y, x in coords],
            }
        )
    if unknown and strict_unknown_colors:
        raise UnknownMaskColorError(mask_path, unknown)
    return mapped_values[inverse].reshape(encoded.shape), unknown


class CamVidDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        class_dict_path,
        transform=None,
        image_size=(352, 480),
        mask_suffix_to_remove=None,
        strict_unknown_colors=True,
        unknown_color_ignore_index=None,
        num_classes=None,
        ignore_index=None,
        ignore_class_name=None,
        allow_non_contiguous_class_indices=False,
        required_class_at_index=None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = tuple(image_size)
        self.mask_suffix_to_remove = mask_suffix_to_remove
        self.strict_unknown_colors = bool(strict_unknown_colors)
        self.unknown_color_ignore_index = unknown_color_ignore_index
        self.class_definitions = parse_camvid_class_dict(class_dict_path)
        validate_class_definitions(
            self.class_definitions,
            expected_num_classes=num_classes,
            ignore_index=ignore_index,
            ignore_class_name=ignore_class_name,
            allow_non_contiguous_indices=allow_non_contiguous_class_indices,
            required_class_at_index=required_class_at_index,
        )
        self.color_to_index = {definition.rgb: definition.index for definition in self.class_definitions}
        pairing = resolve_camvid_pairs(
            self.images_dir,
            self.masks_dir,
            mask_suffix_to_remove=mask_suffix_to_remove,
            raise_on_error=True,
        )
        self.pairs = pairing["pairs"]
        self.images = [pair.image_path.name for pair in self.pairs]
        self.masks = [pair.mask_path.name for pair in self.pairs]

    @staticmethod
    def _is_image_file(filename):
        return is_image_file(filename)

    def _rgb_to_index(self, mask_np, mask_path=""):
        index_mask, _ = rgb_mask_to_index(
            mask_np,
            self.class_definitions,
            mask_path=mask_path,
            strict_unknown_colors=self.strict_unknown_colors,
            unknown_color_ignore_index=self.unknown_color_ignore_index,
        )
        return index_mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask_np = np.array(mask)
        index_mask, _ = rgb_mask_to_index(
            mask_np,
            self.class_definitions,
            mask_path=pair.mask_path,
            strict_unknown_colors=self.strict_unknown_colors,
            unknown_color_ignore_index=self.unknown_color_ignore_index,
        )
        mask_tensor = torch.from_numpy(index_mask).long()
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=self.image_size,
            mode="nearest",
        ).squeeze().long()
        return image, mask_tensor


def imagenet_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def camvid_split_paths(data_root, split):
    data_root = Path(data_root)
    return data_root / split, data_root / f"{split}_labels"


def camvid_split_available(data_root, split):
    images_dir, masks_dir = camvid_split_paths(data_root, split)
    return images_dir.is_dir() and masks_dir.is_dir()


def make_camvid_loader(
    data_root,
    split,
    batch_size=4,
    num_workers=2,
    image_size=(352, 480),
    class_dict_path=None,
    shuffle=False,
    pairing_rule=None,
    strict_unknown_colors=True,
    unknown_color_ignore_index=None,
    num_classes=None,
    ignore_index=None,
    ignore_class_name=None,
    allow_non_contiguous_class_indices=False,
    required_class_at_index=None,
):
    data_root = Path(data_root)
    images_dir, masks_dir = camvid_split_paths(data_root, split)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"CamVid image split is missing: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"CamVid label split is missing: {masks_dir}")
    class_dict_path = Path(class_dict_path) if class_dict_path else data_root / "class_dict.csv"
    pairing_rule = pairing_rule or {}
    dataset = CamVidDataset(
        images_dir,
        masks_dir,
        class_dict_path,
        transform=imagenet_transform(image_size),
        image_size=image_size,
        mask_suffix_to_remove=pairing_rule.get("mask_suffix_to_remove"),
        strict_unknown_colors=strict_unknown_colors,
        unknown_color_ignore_index=unknown_color_ignore_index,
        num_classes=num_classes,
        ignore_index=ignore_index,
        ignore_class_name=ignore_class_name,
        allow_non_contiguous_class_indices=allow_non_contiguous_class_indices,
        required_class_at_index=required_class_at_index,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def dimension_distribution(paths):
    counts = Counter()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        counts[f"{height}x{width}"] += 1
    return dict(sorted(counts.items()))


def validate_camvid_data(config, root):
    dataset_cfg = config.get("dataset", {})
    data_root = Path(dataset_cfg["root"])
    if not data_root.is_absolute():
        data_root = Path(root) / data_root
    class_path = Path(dataset_cfg.get("class_dict_path", data_root / "class_dict.csv"))
    if not class_path.is_absolute():
        class_path = Path(root) / class_path
    pairing_rule = pairing_rule_from_config(dataset_cfg)
    mask_suffix = pairing_rule.get("mask_suffix_to_remove")
    strict_unknown_colors = strict_unknown_colors_from_config(dataset_cfg)
    ignore_index = dataset_cfg.get("ignore_index")
    class_definitions = []
    class_errors = []
    try:
        class_definitions = parse_camvid_class_dict(class_path)
        validate_class_definitions(class_definitions, **class_validation_options(dataset_cfg))
    except Exception as exc:
        class_errors.append(str(exc))
    class_names_by_index = {definition.index: definition.name for definition in class_definitions}
    rgb_by_index = {definition.index: definition.rgb for definition in class_definitions}

    split_reports = []
    summary_rows = []
    class_count_rows = []
    all_unknown = []
    total_inspected_pixels = 0
    splits = dataset_cfg.get("splits", ["train", "val", "test"])

    for split in splits:
        images_dir, masks_dir = camvid_split_paths(data_root, split)
        split_report = {
            "split": split,
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "image_count": 0,
            "mask_count": 0,
            "matched_pair_count": 0,
            "missing_masks": [],
            "unmatched_masks": [],
            "duplicate_or_ambiguous_keys": {"image_keys": {}, "mask_keys": {}, "ambiguous_keys": {}},
            "image_dimension_distribution": {},
            "mask_dimension_distribution": {},
            "unknown_rgb_values": [],
            "per_class_pixel_counts": {},
            "total_mask_pixels": 0,
            "errors": [],
        }
        if not images_dir.is_dir() or not masks_dir.is_dir():
            split_report["errors"].append(f"Missing split directories for {split}")
            split_reports.append(split_report)
            summary_rows.append(
                {
                    "split": split,
                    "status": "fail",
                    "image_count": 0,
                    "mask_count": 0,
                    "matched_pair_count": 0,
                    "missing_mask_count": 0,
                    "unmatched_mask_count": 0,
                    "duplicate_key_count": 0,
                    "unknown_rgb_count": 0,
                    "inspected_mask_pixels": 0,
                }
            )
            continue
        pairing = resolve_camvid_pairs(
            images_dir,
            masks_dir,
            mask_suffix_to_remove=mask_suffix,
            raise_on_error=False,
        )
        split_report["image_count"] = pairing["image_count"]
        split_report["mask_count"] = pairing["mask_count"]
        split_report["matched_pair_count"] = len(pairing["pairs"])
        split_report["missing_masks"] = pairing["missing_masks"]
        split_report["unmatched_masks"] = pairing["unmatched_masks"]
        split_report["duplicate_or_ambiguous_keys"] = pairing["duplicate_or_ambiguous"]
        split_report["errors"].extend(pairing["errors"])
        image_paths = [pair.image_path for pair in pairing["pairs"]]
        mask_paths = [pair.mask_path for pair in pairing["pairs"]]
        split_report["image_dimension_distribution"] = dimension_distribution(image_paths)
        split_report["mask_dimension_distribution"] = dimension_distribution(mask_paths)

        class_counts = Counter()
        for pair in pairing["pairs"]:
            with Image.open(pair.mask_path) as mask:
                mask_np = np.array(mask.convert("RGB"))
            encoded = encode_rgb(mask_np)
            total_pixels = int(encoded.size)
            split_report["total_mask_pixels"] += total_pixels
            total_inspected_pixels += total_pixels
            if class_definitions:
                unique_codes, counts = np.unique(encoded, return_counts=True)
                lookup = color_lookup(class_definitions)
                for code, count in zip(unique_codes.tolist(), counts.tolist()):
                    code = int(code)
                    count = int(count)
                    if code in lookup:
                        class_counts[lookup[code]] += count
                    else:
                        coords = np.argwhere(encoded == code)[:5]
                        unknown_item = {
                            "rgb": decode_rgb_code(code),
                            "pixel_count": count,
                            "mask": str(pair.mask_path),
                            "example_coordinates_yx": [[int(y), int(x)] for y, x in coords],
                        }
                        split_report["unknown_rgb_values"].append(unknown_item)
                        all_unknown.append({**unknown_item, "split": split})
        if strict_unknown_colors and split_report["unknown_rgb_values"]:
            split_report["errors"].append(
                f"{len(split_report['unknown_rgb_values'])} unknown RGB entries found under strict policy"
            )
        split_report["per_class_pixel_counts"] = {
            str(index): int(class_counts.get(index, 0)) for index in sorted(class_names_by_index)
        }
        for index in sorted(class_names_by_index):
            pixel_count = int(class_counts.get(index, 0))
            total = split_report["total_mask_pixels"]
            rgb = rgb_by_index[index]
            class_count_rows.append(
                {
                    "split": split,
                    "class_index": index,
                    "class_name": class_names_by_index[index],
                    "rgb": f"{rgb[0]},{rgb[1]},{rgb[2]}",
                    "pixel_count": pixel_count,
                    "pixel_proportion": pixel_count / total if total else 0.0,
                    "excluded_from_evaluation": bool(ignore_index is not None and index == int(ignore_index)),
                }
            )
        duplicate_key_count = sum(
            len(split_report["duplicate_or_ambiguous_keys"][bucket])
            for bucket in ["image_keys", "mask_keys", "ambiguous_keys"]
        )
        summary_rows.append(
            {
                "split": split,
                "status": "fail" if split_report["errors"] else "pass",
                "image_count": split_report["image_count"],
                "mask_count": split_report["mask_count"],
                "matched_pair_count": split_report["matched_pair_count"],
                "missing_mask_count": len(split_report["missing_masks"]),
                "unmatched_mask_count": len(split_report["unmatched_masks"]),
                "duplicate_key_count": duplicate_key_count,
                "unknown_rgb_count": sum(item["pixel_count"] for item in split_report["unknown_rgb_values"]),
                "inspected_mask_pixels": split_report["total_mask_pixels"],
            }
        )
        split_reports.append(split_report)

    status = "fail" if class_errors or any(split["errors"] for split in split_reports) else "pass"
    unknown_pixel_count = sum(item["pixel_count"] for item in all_unknown)
    unknown_policy = dataset_cfg.get("unknown_color_policy", "strict")
    report = {
        "schema": "smallnet.dataset_validation.v1",
        "status": status,
        "dataset_source": {
            "name": dataset_cfg.get("name"),
            "source_description": dataset_cfg.get("source_description"),
            "source_url": dataset_cfg.get("source_url"),
            "source_archive_identifier": dataset_cfg.get("source_archive_identifier"),
        },
        "split_convention": dataset_cfg.get("split_convention"),
        "class_dict_path": str(class_path),
        "class_dict_sha256": sha256_file(class_path) if class_path.is_file() else "",
        "num_classes": len(class_definitions),
        "configured_num_classes": dataset_cfg.get("num_classes"),
        "ignore_index": ignore_index,
        "ignore_class_name": dataset_cfg.get("ignore_class_name"),
        "pairing_rule": pairing_rule,
        "unknown_color_policy": unknown_policy,
        "strict_unknown_colors": strict_unknown_colors,
        "unknown_color_ignore_index": dataset_cfg.get("unknown_color_ignore_index"),
        "unknown_color_resolution": dataset_cfg.get("unknown_color_resolution"),
        "unknown_pixels_mapped_to_ignore": unknown_pixel_count if unknown_policy == "map_to_ignore" else 0,
        "class_dictionary_errors": class_errors,
        "splits": split_reports,
        "unknown_rgb_values": all_unknown,
        "total_inspected_mask_pixels": total_inspected_pixels,
    }
    return report, summary_rows, class_count_rows


class VOCSegmentationTensorDataset(Dataset):
    def __init__(self, root, year="2012", image_set="val", download=False, image_size=(520, 520)):
        from torchvision.datasets import VOCSegmentation

        self.dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        self.image_size = tuple(image_size)
        self.image_transform = imagenet_transform(self.image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.image_transform(image)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long).reshape(self.image_size)
        return image, mask


def make_voc_loader(root, split="val", batch_size=4, num_workers=2, image_size=(520, 520), download=False):
    dataset = VOCSegmentationTensorDataset(
        root=root,
        image_set=split,
        download=download,
        image_size=image_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
