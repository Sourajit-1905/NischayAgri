import io

import pytest


def _fake_jpeg_bytes() -> bytes:
    # Minimal valid JPEG header (SOI + APP0) so imghdr.what() detects 'jpeg'.
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300"
    ) + b"\x00" * 128 + bytes.fromhex("ffd9")


@pytest.mark.asyncio
async def test_predict_rejects_empty_upload(client):
    files = {"image": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    resp = await client.post("/v1/predict", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_upload"


@pytest.mark.asyncio
async def test_predict_rejects_non_image(client):
    files = {"image": ("note.txt", io.BytesIO(b"not an image"), "text/plain")}
    resp = await client.post("/v1/predict", files=files)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_image_type"


@pytest.mark.asyncio
async def test_predict_accepts_valid_jpeg_and_returns_decision(client):
    files = {"image": ("leaf.jpg", io.BytesIO(_fake_jpeg_bytes()), "image/jpeg")}
    resp = await client.post("/v1/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"]["action"] in {
        "ACCEPT", "ABSTAIN_RETAKE", "ABSTAIN_EXPERT", "REJECT_UNKNOWN"
    }
    assert 0 <= body["uncertainty"]["calibrated_confidence"] <= 1