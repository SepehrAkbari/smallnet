"""Forensic, non-mutating inspection helpers for CamVid RGB masks."""

from collections import Counter, deque
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from src.smallnet.data import encode_rgb, parse_camvid_class_dict, sha256_file


def _jsonable(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _rgb_counts(rgb):
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    return [
        {"rgb": color.astype(int).tolist(), "pixel_count": int(count)}
        for color, count in sorted(zip(colors, counts), key=lambda item: (-int(item[1]), tuple(item[0])))
    ]


def _png_structure(path):
    chunks = Counter()
    ihdr = {}
    with open(path, "rb") as file:
        if file.read(8) != b"\x89PNG\r\n\x1a\n":
            return {"chunks": {}, "ihdr": {}}
        while True:
            length_bytes = file.read(4)
            if len(length_bytes) != 4:
                break
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = file.read(4)
            data = file.read(length)
            file.read(4)
            name = chunk_type.decode("ascii", errors="replace")
            chunks[name] += 1
            if name == "IHDR" and len(data) == 13:
                width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
                ihdr = {
                    "width": width,
                    "height": height,
                    "bit_depth": bit_depth,
                    "color_type": color_type,
                    "color_type_name": {0: "grayscale", 2: "truecolor_rgb", 3: "indexed_palette", 4: "grayscale_alpha", 6: "truecolor_rgba"}.get(color_type, "unknown"),
                    "compression_method": compression,
                    "filter_method": filtering,
                    "interlace_method": interlace,
                }
            if name == "IEND":
                break
    return {"chunks": dict(chunks), "ihdr": ihdr}


def _connected_components(binary_mask):
    height, width = binary_mask.shape
    seen = np.zeros_like(binary_mask, dtype=bool)
    sizes = []
    for y, x in np.argwhere(binary_mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        queue = deque([(y, x)])
        seen[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and binary_mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def summarize_unknown_mask(mask_path, class_definitions):
    mask_path = Path(mask_path)
    with Image.open(mask_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    valid_codes = {
        (definition.rgb[0] << 16) + (definition.rgb[1] << 8) + definition.rgb[2]
        for definition in class_definitions
    }
    encoded = encode_rgb(rgb)
    unknown_mask = ~np.isin(encoded, list(valid_codes))
    coords = np.argwhere(unknown_mask)
    if not len(coords):
        return None
    unknown_codes, counts = np.unique(encoded[unknown_mask], return_counts=True)
    component_sizes = _connected_components(unknown_mask)
    y0, x0 = coords.min(axis=0).astype(int)
    y1, x1 = coords.max(axis=0).astype(int)
    return {
        "mask_path": str(mask_path),
        "total_unknown_pixels": int(unknown_mask.sum()),
        "distinct_unknown_rgb_values": int(len(unknown_codes)),
        "bounding_box_ymin_xmin_ymax_xmax": [int(y0), int(x0), int(y1), int(x1)],
        "affected_proportion": float(unknown_mask.mean()),
        "connected_region_count_4_neighbor": len(component_sizes),
        "all_unknown_pixels_in_one_connected_region": len(component_sizes) == 1,
        "connected_region_sizes": component_sizes,
        "unknown_colors": [
            {"rgb": [(int(code) >> 16) & 255, (int(code) >> 8) & 255, int(code) & 255], "pixel_count": int(count)}
            for code, count in zip(unknown_codes, counts)
        ],
    }


def aggregate_unknown_colors_by_file(mask_paths, class_definitions):
    return [summary for path in mask_paths if (summary := summarize_unknown_mask(path, class_definitions))]


def _between_neighbor_colors(value, colors):
    value = np.asarray(value, dtype=float)
    colors = [np.asarray(color, dtype=float) for color in colors]
    channelwise = False
    segment = False
    for i, first in enumerate(colors):
        for second in colors[i + 1 :]:
            channelwise |= bool(np.all(value >= np.minimum(first, second)) and np.all(value <= np.maximum(first, second)))
            direction = second - first
            denom = float(direction @ direction)
            if denom == 0:
                continue
            t = float(((value - first) @ direction) / denom)
            projected = first + min(1.0, max(0.0, t)) * direction
            if 0.0 <= t <= 1.0 and np.max(np.abs(projected - value)) <= 1.0:
                segment = True
    return channelwise, segment


def inspect_mask_forensics(mask_path, class_dict_path):
    mask_path = Path(mask_path)
    definitions = parse_camvid_class_dict(class_dict_path)
    valid = {definition.rgb for definition in definitions}
    with Image.open(mask_path) as image:
        original_mode = image.mode
        image_format = image.format
        size = list(image.size)
        metadata = {str(key): _jsonable(value) for key, value in image.info.items()}
        palette = image.getpalette() if image.mode == "P" else None
        rgb = np.asarray(image.convert("RGB"))
    summary = summarize_unknown_mask(mask_path, definitions)
    unknown_mask = np.zeros(rgb.shape[:2], dtype=bool)
    if summary:
        unknown_colors = {tuple(item["rgb"]) for item in summary["unknown_colors"]}
        unknown_mask = np.array([[tuple(pixel) in unknown_colors for pixel in row] for row in rgb], dtype=bool)

    evidence_by_color = {}
    boundary_count = channelwise_count = segment_count = 0
    height, width = unknown_mask.shape
    for y, x in np.argwhere(unknown_mask):
        y, x = int(y), int(x)
        neighbors = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and tuple(rgb[ny, nx]) in valid:
                    neighbors.add(tuple(int(v) for v in rgb[ny, nx]))
        value = tuple(int(v) for v in rgb[y, x])
        channelwise, segment = _between_neighbor_colors(value, sorted(neighbors))
        between_regions = len(neighbors) >= 2
        boundary_count += int(between_regions)
        channelwise_count += int(channelwise)
        segment_count += int(segment)
        item = evidence_by_color.setdefault(
            value,
            {"rgb": list(value), "pixel_count": 0, "pixels_with_two_or_more_valid_neighbor_colors": 0,
             "channelwise_intermediate_pixels": 0, "two_color_convex_segment_pixels": 0,
             "neighboring_valid_colors": set()},
        )
        item["pixel_count"] += 1
        item["pixels_with_two_or_more_valid_neighbor_colors"] += int(between_regions)
        item["channelwise_intermediate_pixels"] += int(channelwise)
        item["two_color_convex_segment_pixels"] += int(segment)
        item["neighboring_valid_colors"].update(neighbors)

    color_evidence = []
    for item in evidence_by_color.values():
        item["neighboring_valid_colors"] = [list(color) for color in sorted(item["neighboring_valid_colors"])]
        color_evidence.append(item)
    total = int(unknown_mask.sum())
    evidence = {
        "unknown_pixels": total,
        "unknown_pixels_adjacent_to_two_or_more_valid_colors": boundary_count,
        "unknown_pixels_channelwise_intermediate_between_neighboring_valid_colors": channelwise_count,
        "unknown_pixels_on_two_color_segment_with_rounding_tolerance": segment_count,
        "per_unknown_color": sorted(color_evidence, key=lambda item: item["rgb"]),
        "method_note": (
            "Neighborhoods are 8-connected. Channel-wise intermediacy means the RGB value lies inside the "
            "axis-aligned range of a pair of neighboring valid class colors. The convex-segment test requires "
            "the value to lie within 1 RGB level of a segment joining two neighboring valid colors."
        ),
    }
    if total and boundary_count == total and channelwise_count == total:
        conclusion = (
            "All invalid pixels are boundary-adjacent and channel-wise intermediate. This is consistent with "
            "interpolation or antialiasing, but does not establish provenance without comparison to an authoritative source mask."
        )
    elif total:
        conclusion = (
            "The invalid pixels do not uniformly satisfy the tested boundary/intermediacy criteria; these data alone "
            "do not support attributing every invalid value to interpolation or antialiasing."
        )
    else:
        conclusion = "No unknown RGB values were found."
    return {
        "mask_path": str(mask_path),
        "sha256": sha256_file(mask_path),
        "file_format": image_format,
        "image_mode": original_mode,
        "pixel_encoding": "indexed_palette" if original_mode == "P" else "true_rgb" if original_mode == "RGB" else "other",
        "dimensions_width_height": size,
        "png_metadata": metadata if image_format == "PNG" else {},
        "png_structure": _png_structure(mask_path) if image_format == "PNG" else {},
        "indexed_palette_entry_count": len(palette) // 3 if palette else 0,
        "colors_and_counts_rgb": _rgb_counts(rgb),
        "unknown_summary": summary,
        "spatial_and_color_evidence": evidence,
        "conclusion": conclusion,
    }


def compare_camvid_masks(current_path, reference_path, class_dict_path):
    current_path, reference_path = Path(current_path), Path(reference_path)
    definitions = parse_camvid_class_dict(class_dict_path)
    records = []
    arrays = []
    for path in (current_path, reference_path):
        with Image.open(path) as image:
            records.append({"path": str(path), "sha256": sha256_file(path), "dimensions_width_height": list(image.size), "image_mode": image.mode, "file_format": image.format})
            arrays.append(np.asarray(image.convert("RGB")))
    same_shape = arrays[0].shape == arrays[1].shape
    if same_shape:
        different = np.any(arrays[0] != arrays[1], axis=2)
        transitions = Counter((tuple(arrays[0][y, x]), tuple(arrays[1][y, x])) for y, x in np.argwhere(different))
        transition_rows = [
            {"current_rgb": list(before), "reference_rgb": list(after), "pixel_count": int(count)}
            for (before, after), count in sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
        ]
        differing_pixels = int(different.sum())
    else:
        transition_rows, differing_pixels = [], None
    reference_unknown = summarize_unknown_mask(reference_path, definitions)
    return {
        "current": records[0], "reference": records[1], "dimensions_match": same_shape,
        "exact_differing_pixel_count": differing_pixels, "rgb_transition_summary": transition_rows,
        "reference_unknown_pixel_count": reference_unknown["total_unknown_pixels"] if reference_unknown else 0,
        "replacing_file_would_eliminate_all_unknown_colors": reference_unknown is None,
    }
