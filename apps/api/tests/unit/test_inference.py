"""
Tests for the real inference pipeline (use_stubs=False path).
"""
import io

import numpy as np
import pytest
from PIL import Image


def _make_jpeg_bytes(size=(256, 256)) -> bytes:
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, (*size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_quality_service_sharp_image():
    from cropapi.services.quality_service import score_quality
    rng = np.random.default_rng(0)
    arr = rng.integers(80, 180, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    q = score_quality(img)
    assert 0.0 <= q.blur <= 1.0
    assert 0.0 <= q.lighting <= 1.0
    assert 0.0 <= q.framing <= 1.0
    assert isinstance(q.usable, bool)


def test_quality_service_dark_image():
    from cropapi.services.quality_service import score_quality
    dark = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 5)
    q = score_quality(dark)
    assert q.lighting > 0.3


def test_strip_exif_returns_valid_jpeg():
    from cropapi.core.security import strip_exif_and_validate
    clean = strip_exif_and_validate(_make_jpeg_bytes())
    assert clean[0] == 0xff and clean[1] == 0xd8


def test_strip_exif_rejects_non_image():
    from cropapi.core.errors import AppError
    from cropapi.core.security import strip_exif_and_validate
    with pytest.raises(AppError) as exc_info:
        strip_exif_and_validate(b"not an image at all")
    assert exc_info.value.code == "unreadable_image"


def test_decision_service_reject_ood():
    import numpy as np
    from cropml.decision.policy import Decision

    from cropapi.services.decision_service import DecisionService
    from cropapi.services.inference_service import RawInferenceResult
    from cropapi.services.quality_service import QualityScore

    svc = DecisionService()
    result = RawInferenceResult(
        request_id="test",
        crop_probs=np.array([0.9] + [0.0] * 13),
        disease_probs=np.array([0.9] + [0.0] * 37),
        crop_logits=np.zeros(14),
        disease_logits=np.zeros(38),
        ood_score=0.95,
        entropy=0.1,
        quality=QualityScore(blur=0.0, lighting=0.0, framing=0.0, usable=True),
        inference_ms=10.0,
        model_version="test",
    )
    decision = svc.apply(result)
    assert decision.action == Decision.REJECT_UNKNOWN


def test_decision_service_accept():
    import numpy as np
    from cropml.decision.policy import Decision

    from cropapi.services.decision_service import DecisionService
    from cropapi.services.inference_service import RawInferenceResult
    from cropapi.services.quality_service import QualityScore

    svc = DecisionService()
    result = RawInferenceResult(
        request_id="test",
        crop_probs=np.array([0.9] + [0.0] * 13),
        disease_probs=np.array([0.9] + [0.0] * 37),
        crop_logits=np.zeros(14),
        disease_logits=np.zeros(38),
        ood_score=0.05,
        entropy=0.1,
        quality=QualityScore(blur=0.0, lighting=0.0, framing=0.0, usable=True),
        inference_ms=10.0,
        model_version="test",
    )
    decision = svc.apply(result)
    assert decision.action == Decision.ACCEPT