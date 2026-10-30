# PHASE 2.1 — API Scaffold with Contract-Conformant Stubs

## What this phase does

Creates the `apps/api` FastAPI service using a `cropapi` src-layout package, Pydantic v2 settings, structured JSON logging, request IDs, centralized error handling, and CORS.

Implements `/v1/health`, `/v1/meta`, `/v1/predict`, and `/v1/advice`. The endpoints return **stub data from `contracts/examples/`**, not real inference results, allowing the frontend to integrate with a real HTTP API before the ML pipeline exists.

Also configures `pyproject.toml` (Ruff, strict mypy, pytest), a Dockerfile, `infra/docker-compose.yml` (web + API), and the `ci-api.yml` workflow.

## Done When

The frontend works against the real stub API by changing only the base URL from the mock.


## Implementation Order

**Contract-first development is mandatory:** contracts must be defined and validated before implementing API stubs.

| Step | Task | Purpose |
|---:|---|---|
| **1** | **Define Contracts** — `contracts/openapi.yaml`, JSON Schemas, and examples | Defines response schemas, decision enums, and API response formats. |
| **2** | **Build API Scaffold** | Configures settings, logging, error handling, middleware, and CORS. |
| **3** | **Implement Routes and Stub Services** | Connects API endpoints to contract-compliant example responses. |
| **4** | **Add Tests and Deployment Configuration** | Verifies contract conformance and configures Docker, Compose, and CI. |

### `contracts/openapi.yaml`

This is the single source of truth for the API — the frontend’s generated types and the backend’s contract tests both derive from this file. It defines the four endpoints, the `Decision` enum, and references the request/response schemas (which I’ll put in `contracts/schemas/` as separate files in the next step, since OpenAPI + JSON Schema split keeps things reusable).

### `contracts/schemas/`

Six schema files. These are JSON Schema (2020-12), referenced from `openapi.yaml` via `$ref`. I’m using `$defs` for shared nested objects so `openapi-typescript` generates clean reusable TS types.

The JSON Schema files this references:

- `predict_response.schema.json`
- `advice_response.schema.json`
- `advice_request.schema.json`
- `meta.schema.json`
- `health.schema.json`
- `error.schema.json`

### `contracts/examples/`

Four `PredictionResponse` examples (one per `Decision`) and three `AdviceResponse` examples (one per `AdviceMode`). These feed both the MSW mocks (Phase 1.2) and the API stub services (Phase 2.1), so they need to be realistic and internally consistent — e.g. the `ABSTAIN_EXPERT` advice example should use the same crop/disease as the `ABSTAIN_EXPERT` predict example.

### `apps/api` scaffold

The `apps/api` scaffold itself — `pyproject.toml`, `config.py`, `core/logging.py`, `core/errors.py`, `core/middleware.py`.

# PHASE 2.2 — Data Layer and Evaluation Protocol (`ml/`)

## What this phase does

(as before) loaders for PlantVillage/PlantDoc/FieldPlant, unified label-space mapping, leakage-free DVC-tracked splits, four evaluation sets, and a data sheet.

## Implementation Order, and Why

1. **`ml/pyproject.toml`** — nothing else can be installed or imported without this first.
2. **`configs/data/{plantvillage,plantdoc,fieldplant}.yaml`** — defines each dataset’s `root_dir`, raw structure, and class-name conventions; the loaders in step 4 read these, so the configs must exist first.
3. **`src/cropml/data/label_space.yaml`** + **`label_space.py`** — the unified in-domain/shared/unseen-in-training mapping all three loaders normalize into. Loaders depend on this, so it comes before them, not after.
4. **`src/cropml/data/loaders/base.py`**, then **`plantvillage.py`**, **`plantdoc.py`**, **`fieldplant.py`** — the common loader interface first, then each dataset-specific implementation against it and against `label_space.py`.
5. **`src/cropml/data/dedup.py`** — perceptual-hash near-duplicate clustering, needed by splitting (step 6) for datasets without reliable source-plant metadata.
6. **`src/cropml/data/splitting.py`** — the leakage-free group splitter; depends on loaders (step 4) and dedup (step 5).
7. **`src/cropml/data/manifest.py`** — persists/loads split manifests; depends on splitting’s output shape (step 6).
8. **`src/cropml/data/ood_sets.py`** and **`corruptions.py`** — the four evaluation sets; depend on loaders and the label space, independent of each other, built after the core split logic is solid.
9. **`scripts/build_splits.py`** — the CLI entrypoint tying loaders → dedup → splitting → manifest together; comes after all the pieces it orchestrates exist.
10. **`configs/splits/default.yaml`** — the Hydra config `build_splits.py` reads (seed, ratios, grouping strategy); written alongside or just after the script so the two are developed against each other.
11. **`tests/conftest.py`**, then **`test_dedup.py`**, **`test_splitting.py`**, **`test_leakage.py`** — tests last, against the finished implementation, with the leakage test being the phase’s actual “done when” gate.
12. **`docs/data-sheets/crop-disease-datasets.md`** — written last since it documents the final class counts and licensing once the label space and loaders are settled.

