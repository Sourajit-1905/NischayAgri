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