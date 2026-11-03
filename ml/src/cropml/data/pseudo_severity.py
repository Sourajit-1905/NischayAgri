"""
Lesion-area-ratio pseudo-severity heuristic.

A SUBJECTIVE HEURISTIC, not ground truth - PlantVillage has no real
severity labels. Operates on the 'segmented' image variant (leaf isolated
on a pure black background), computing what fraction of leaf-tissue
pixels look diseased by color, then bucketing into none/low/moderate/high.
See docs/severity.md (Phase 2.4 step 7) for documented limitations.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image


class SeverityBucket(StrEnum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class SeverityConfig:
    max_rgb_for_background: int
    healthy_hue_min_degrees: float
    healthy_hue_max_degrees: float
    healthy_min_saturation: float
    dark_lesion_value_threshold: float
    none_max: float            # general.none_max
    low_max: float              # general.low_max
    moderate_max: float         # general.moderate_max
    diseased_low_max: float     # known_diseased.low_max
    diseased_moderate_max: float  # known_diseased.moderate_max

    @classmethod
    def from_dict(cls, raw: dict) -> SeverityConfig:
        return cls(
            max_rgb_for_background=int(raw["background"]["max_rgb_for_background"]),
            healthy_hue_min_degrees=float(raw["healthy_hue"]["hue_min_degrees"]),
            healthy_hue_max_degrees=float(raw["healthy_hue"]["hue_max_degrees"]),
            healthy_min_saturation=float(raw["healthy_hue"]["min_saturation"]),
            dark_lesion_value_threshold=float(raw["dark_lesion"]["value_threshold"]),
            none_max=float(raw["severity_buckets"]["general"]["none_max"]),
            low_max=float(raw["severity_buckets"]["general"]["low_max"]),
            moderate_max=float(raw["severity_buckets"]["general"]["moderate_max"]),
            diseased_low_max=float(raw["severity_buckets"]["known_diseased"]["low_max"]),
            diseased_moderate_max=float(raw["severity_buckets"]["known_diseased"]["moderate_max"]),
        )


def bucket_known_diseased(lesion_ratio: float, config: SeverityConfig) -> SeverityBucket:
    """Buckets a lesion_ratio into low/moderate/high ONLY - never 'none' -
    for use when the image's class is already known to be diseased (i.e.
    generating training labels from PlantVillage class labels). Do not use
    this for images whose disease status is unknown; use estimate_severity's
    bucket field for that case instead."""
    if lesion_ratio <= config.diseased_low_max:
        return SeverityBucket.LOW
    elif lesion_ratio <= config.diseased_moderate_max:
        return SeverityBucket.MODERATE
    else:
        return SeverityBucket.HIGH


@dataclass(frozen=True)
class SeverityEstimate:
    lesion_ratio: float
    bucket: SeverityBucket
    leaf_pixel_count: int
    lesion_pixel_count: int


def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """rgb: (n, 3) uint8 -> (n, 3) float array of (hue_degrees, sat, val).
    Vectorized via colorsys applied per-pixel is too slow for whole images,
    so this implements the RGB->HSV conversion directly with NumPy."""
    r, g, b = rgb[:, 0] / 255.0, rgb[:, 1] / 255.0, rgb[:, 2] / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.where(maxc > 0, delta / np.where(maxc == 0, 1, maxc), 0.0)

    # Hue computation, avoiding division by zero where delta == 0.
    safe_delta = np.where(delta == 0, 1, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    hue = np.zeros_like(maxc)
    is_r_max = (maxc == r) & (delta != 0)
    is_g_max = (maxc == g) & (delta != 0)
    is_b_max = (maxc == b) & (delta != 0)
    hue[is_r_max] = (bc - gc)[is_r_max] % 6.0
    hue[is_g_max] = (rc - bc)[is_g_max] + 2.0
    hue[is_b_max] = (gc - rc)[is_b_max] + 4.0
    hue_degrees = (hue * 60.0) % 360.0

    return np.stack([hue_degrees, s, v], axis=1)

from scipy import ndimage  # add to the top imports


def estimate_severity(image: Image.Image, config: SeverityConfig) -> SeverityEstimate:
    """
    image: a PIL Image (expected to be the 'segmented' variant - leaf on
    a near-pure-black background). Returns the lesion-area ratio and its
    bucketed severity level.

    Applies morphological opening to the raw lesion mask before computing
    the ratio, to remove small isolated noise regions (compression
    artifacts, mask-boundary anti-aliasing, single stray shadow/vein
    pixels) that otherwise inflate the lesion ratio even on healthy
    leaves - confirmed as a real problem empirically (healthy classes'
    median ratio was close to diseased classes' 25th percentile before
    this fix, on this project's real PlantVillage data).
    """
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[0], arr.shape[1]
    flat = arr.reshape(-1, 3)

    is_background_flat = np.all(flat <= config.max_rgb_for_background, axis=1)
    is_background = is_background_flat.reshape(h, w)
    leaf_pixel_count = int((~is_background).sum())

    if leaf_pixel_count == 0:
        return SeverityEstimate(
            lesion_ratio=0.0, bucket=SeverityBucket.NONE, leaf_pixel_count=0, lesion_pixel_count=0
        )

    hsv_flat = _rgb_to_hsv_array(flat)
    hue = hsv_flat[:, 0].reshape(h, w)
    sat = hsv_flat[:, 1].reshape(h, w)
    val = hsv_flat[:, 2].reshape(h, w)

    is_dark_lesion = val < config.dark_lesion_value_threshold
    is_healthy_green = (
        (hue >= config.healthy_hue_min_degrees)
        & (hue <= config.healthy_hue_max_degrees)
        & (sat >= config.healthy_min_saturation)
        & ~is_dark_lesion
    )
    raw_lesion_mask = (~is_healthy_green) & (~is_background)

    # Morphological opening: erosion removes small/thin regions (noise),
    # dilation restores the remaining regions' original size. A 3x3
    # structuring element with 1 iteration removes isolated single-pixel
    # and thin-line noise while preserving genuine lesion patches, which
    # are typically many pixels across.
    cleaned_lesion_mask = ndimage.binary_opening(
        raw_lesion_mask, structure=np.ones((3, 3)), iterations=1
    )

    lesion_pixel_count = int(cleaned_lesion_mask.sum())
    lesion_ratio = lesion_pixel_count / leaf_pixel_count

    if lesion_ratio <= config.none_max:
        bucket = SeverityBucket.NONE
    elif lesion_ratio <= config.low_max:
        bucket = SeverityBucket.LOW
    elif lesion_ratio <= config.moderate_max:
        bucket = SeverityBucket.MODERATE
    else:
        bucket = SeverityBucket.HIGH

    return SeverityEstimate(
        lesion_ratio=lesion_ratio,
        bucket=bucket,
        leaf_pixel_count=leaf_pixel_count,
        lesion_pixel_count=lesion_pixel_count,
    )

def estimate_severity_from_path(path: Path, config: SeverityConfig) -> SeverityEstimate:
    with Image.open(path) as img:
        return estimate_severity(img, config)

    