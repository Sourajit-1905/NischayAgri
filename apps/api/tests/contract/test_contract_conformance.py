"""Validates every contracts/examples/*.json file against its JSON Schema,
and validates live stub responses against the same schemas. This is what
Phase 1.2 (frontend MSW) and this API both rely on staying in sync."""

import json
from pathlib import Path

import jsonschema
import pytest

_ROOT = Path(__file__).resolve().parents[4]  # repo root
_SCHEMAS_DIR = _ROOT / "contracts" / "schemas"
_EXAMPLES_DIR = _ROOT / "contracts" / "examples"


def _load_schema(name: str) -> dict:
    with (_SCHEMAS_DIR / name).open() as f:
        return json.load(f)


def _resolver_for(schema: dict) -> jsonschema.RefResolver:
    store = {}
    for schema_file in _SCHEMAS_DIR.glob("*.json"):
        with schema_file.open() as f:
            loaded = json.load(f)
        store[loaded["$id"]] = loaded
    return jsonschema.RefResolver.from_schema(schema, store=store)


_PREDICT_EXAMPLES = [
    "predict_accept.json",
    "predict_abstain_retake.json",
    "predict_abstain_expert.json",
    "predict_reject_unknown.json",
]

_ADVICE_EXAMPLES = ["advice_full.json", "advice_limited.json", "advice_expert_referral.json"]


@pytest.mark.parametrize("example_file", _PREDICT_EXAMPLES)
def test_predict_examples_conform_to_schema(example_file):
    schema = _load_schema("predict_response.schema.json")
    resolver = _resolver_for(schema)
    with (_EXAMPLES_DIR / example_file).open() as f:
        instance = json.load(f)
    jsonschema.validate(instance=instance, schema=schema, resolver=resolver)


@pytest.mark.parametrize("example_file", _ADVICE_EXAMPLES)
def test_advice_examples_conform_to_schema(example_file):
    schema = _load_schema("advice_response.schema.json")
    resolver = _resolver_for(schema)
    with (_EXAMPLES_DIR / example_file).open() as f:
        instance = json.load(f)
    jsonschema.validate(instance=instance, schema=schema, resolver=resolver)


@pytest.mark.asyncio
async def test_live_predict_response_conforms_to_schema(client):
    import io

    schema = _load_schema("predict_response.schema.json")
    resolver = _resolver_for(schema)
    fake_jpeg = (
        bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffdb004300")
        + b"\x00" * 64
        + bytes.fromhex("ffd9")
    )
    files = {"image": ("leaf.jpg", io.BytesIO(fake_jpeg), "image/jpeg")}
    resp = await client.post("/v1/predict", files=files)
    jsonschema.validate(instance=resp.json(), schema=schema, resolver=resolver)