"""
On-the-fly image corruption suite: blur, low light, occlusion, noise,
JPEG compression, each at several severities.

Corruptions are applied to a PIL Image and return a PIL Image, so they
compose with any loading pipeline without needing pre-generated files
on disk.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image, ImageFilter


class CorruptionType(StrEnum):
    BLUR = "blur"
    LOW_LIGHT = "low_light"
    OCCLUSION = "occlusion"
    NOISE = "noise"
    JPEG_COMPRESSION = "jpeg_compression"


# severity 1 (mild) to 5 (severe), chosen to span "barely noticeable" to
# "a realistic worst-case field photo" rather than purely cosmetic levels.
SEVERITY_LEVELS = (1, 2, 3, 4, 5)


def apply_blur(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    radius = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0, 5: 7.0}[severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_low_light(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    factor = {1: 0.8, 2: 0.6, 3: 0.45, 4: 0.3, 5: 0.15}[severity]
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_occlusion(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    frac = {1: 0.05, 2: 0.10, 3: 0.18, 4: 0.28, 5: 0.40}[severity]
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    patch_h, patch_w = int(h * frac**0.5), int(w * frac**0.5)
    y0 = int(rng.integers(0, max(h - patch_h, 1)))
    x0 = int(rng.integers(0, max(w - patch_w, 1)))
    arr[y0 : y0 + patch_h, x0 : x0 + patch_w] = 0
    return Image.fromarray(arr)


def apply_noise(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    std = {1: 5, 2: 10, 3: 18, 4: 28, 5: 40}[severity]
    arr = np.asarray(img).astype(np.float32)
    noise = rng.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_jpeg_compression(
    img: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    quality = {1: 60, 2: 40, 3: 25, 4: 12, 5: 5}[severity]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert(img.mode if img.mode in ("RGB", "L") else "RGB")


_APPLIERS: dict[CorruptionType, Callable[[Image.Image, int, np.random.Generator], Image.Image]] = {
    CorruptionType.BLUR: apply_blur,
    CorruptionType.LOW_LIGHT: apply_low_light,
    CorruptionType.OCCLUSION: apply_occlusion,
    CorruptionType.NOISE: apply_noise,
    CorruptionType.JPEG_COMPRESSION: apply_jpeg_compression,
}


@dataclass(frozen=True)
class CorruptionSpec:
    kind: CorruptionType
    severity: int
    seed: int

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {self.severity}")


def apply_corruption(img: Image.Image, spec: CorruptionSpec) -> Image.Image:
    rng = np.random.default_rng(spec.seed)
    return _APPLIERS[spec.kind](img, spec.severity, rng)


def all_corruption_specs(base_seed: int) -> tuple[CorruptionSpec, ...]:
    """
    Every (kind, severity) combination, each with its own derived seed
    so results are reproducible but not identical across combinations."""
    specs = []
    i = 0
    for kind in CorruptionType:
        for severity in SEVERITY_LEVELS:
            specs.append(CorruptionSpec(kind=kind, severity=severity, seed=base_seed + i))
            i += 1
    return tuple(specs)




