import pytest


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.asyncio
async def test_meta_returns_supported_crops(client):
    resp = await client.get("/v1/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert "tomato" in body["supported_crops"]
    assert len(body["supported_diseases"]) > 0


@pytest.mark.asyncio
async def test_request_id_header_present(client):
    resp = await client.get("/v1/health")
    assert "x-request-id" in resp.headers