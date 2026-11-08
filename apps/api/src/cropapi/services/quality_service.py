"""Image quality scoring: blur, lighting, framing, and a usability gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityScore:
    blur: float        # 0=sharp, 1=very blurry
    lighting: float    # 0=good, 1=very dark or overexposed
    framing: float     # 0=well-framed, 1=poorly framed
    usable: bool


def _laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper. Computed as variance of a 3x3 Laplacian."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    lap = convolve2d(gray.astype(np.float32), kernel, mode="valid")
    return float(lap.var())


def _lighting_score(gray: np.ndarray) -> float:
    """0=good exposure. Penalizes very dark (<30 mean) or overexposed (>220 mean)."""
    mean = gray.mean()
    if mean < 30:
        return float((30 - mean) / 30)
    if mean > 220:
        return float((mean - 220) / 35)
    return 0.0


def _framing_score(gray: np.ndarray) -> float:
    """Estimates whether the subject occupies a reasonable portion of the frame.
    Uses edge density in the center vs border - a well-framed leaf should
    have more edge content in the center than the border."""
    from scipy.ndimage import sobel
    edges = np.hypot(sobel(gray.astype(np.float32), axis=0),
                     sobel(gray.astype(np.float32), axis=1))
    h, w = edges.shape
    mh, mw = h // 4, w // 4
    center = edges[mh:h - mh, mw:w - mw].mean()
    border = edges.mean()
    if border < 1e-6:
        return 0.5
    ratio = center / border
    # ratio < 1.0 means more content at border than center -> poor framing
    return float(max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 2.0)))


def score_quality(
    image: Image.Image,
    blur_threshold: float = 50.0,
    lighting_bad_threshold: float = 0.4,
    framing_bad_threshold: float = 0.7,
) -> QualityScore:
    gray = np.asarray(image.convert("L"))

    lap_var = _laplacian_variance(gray)
    # Map Laplacian variance to [0,1] blur score: low variance = blurry.
    blur = float(max(0.0, min(1.0, 1.0 - lap_var / (blur_threshold * 10))))

    lighting = float(max(0.0, min(1.0, _lighting_score(gray))))
    framing = float(max(0.0, min(1.0, _framing_score(gray))))

    usable = (
        blur < blur_threshold / 100
        and lighting < lighting_bad_threshold
        and framing < framing_bad_threshold
    )

    return QualityScore(blur=blur, lighting=lighting, framing=framing, usable=usable)