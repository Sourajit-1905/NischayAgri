import numpy as np
from PIL import Image

from cropml.data.pseudo_severity import (
    SeverityBucket,
    SeverityConfig,
    bucket_known_diseased,
    estimate_severity,
)


def _cfg() -> SeverityConfig:
    return SeverityConfig(
        max_rgb_for_background=10,
        healthy_hue_min_degrees=70, healthy_hue_max_degrees=170,
        healthy_min_saturation=0.15, dark_lesion_value_threshold=0.03,
        none_max=0.02, low_max=0.10, moderate_max=0.30,
        diseased_low_max=0.015, diseased_moderate_max=0.09,
    )

def _make(leaf_rgb, size=64, frac=0.6):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    s = int(size * frac); o = (size - s) // 2
    arr[o:o+s, o:o+s] = leaf_rgb
    return Image.fromarray(arr)

def test_green_scores_low():
    r = estimate_severity(_make((34, 139, 34)), _cfg())
    assert r.bucket in (SeverityBucket.NONE, SeverityBucket.LOW)

def test_brown_scores_high():
    r = estimate_severity(_make((101, 67, 33)), _cfg())
    assert r.bucket in (SeverityBucket.MODERATE, SeverityBucket.HIGH)

def test_all_background_returns_none():
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    r = estimate_severity(img, _cfg())
    assert r.bucket == SeverityBucket.NONE and r.leaf_pixel_count == 0

def test_bucket_known_diseased_boundaries():
    cfg = _cfg()
    assert bucket_known_diseased(0.01, cfg) == SeverityBucket.LOW
    assert bucket_known_diseased(0.05, cfg) == SeverityBucket.MODERATE
    assert bucket_known_diseased(0.20, cfg) == SeverityBucket.HIGH