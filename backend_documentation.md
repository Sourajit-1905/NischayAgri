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


# PHASE 2.3 — Baselines and Uncertainty Stack

## What this phase does

Implements lightweight backbones (MobileNetV3, EfficientNet-B0, a small ViT) behind a common interface, with a hierarchical head (crop → disease), training scripts with multi-seed support, and experiment tracking. Then the actual uncertainty stack: softmax baseline, temperature scaling, MC-dropout, a small deep ensemble, and conformal prediction if time allows. OOD scorers: MSP, energy, Mahalanobis, ODIN. Crop-level rejection logic (abstain on uncertain crop before predicting disease; reject invalid crop-disease combos). Training flags for an augmentation ablation (with/without color augmentation, with/without GAN data). Computes accuracy, macro-F1, ECE, OOD AUROC/FPR95, risk-coverage, and per-class calibration/risk-coverage for rare diseases.

## Done When

One command reproduces a results table (models × methods × target domains) as mean ± std over seeds.

## Things to Flag Before Starting

This is the largest phase so far by a significant margin — it’s the research core of the project. A few things worth flagging before we start:

1. **This phase needs real data and real training.** Unlike Phase 2.2, synthetic-tree scratch checks won’t meaningfully validate this — training even a tiny model needs actual images. We should decide now whether to build and verify this against your real PlantVillage data (once downloaded) or against tiny synthetic datasets purely to prove the *code* is wired correctly, with real training deferred.
2. **PyTorch needs to be added** to `ml/pyproject.toml` now — I deliberately deferred it from Phase 2.2’s step 1 for exactly this moment.
3. **Given your no-budget constraint:** everything here runs locally and free (PyTorch, scikit-learn-based conformal prediction, MLflow self-hosted/local-file-based rather than cloud W&B) — I’ll default to MLflow with local file tracking rather than Weights & Biases, since W&B’s free tier has usage limits and MLflow’s local mode has none.

## Implementation Order, and Why

1. **`ml/pyproject.toml` (update)** — add `torch`, `torchvision`, `mlflow`, and `timm` (for the ViT and EfficientNet backbones without hand-rolling them) to dependencies. Nothing else imports without this.
2. **`configs/model/*.yaml`** — defines each backbone’s hyperparameters (input size, pretrained weights source, dropout rate for MC-dropout later). Written before the model code so the code has a config contract to implement against.
3. **`src/cropml/models/backbone.py`** — the common interface (an abstract base all three backbones implement: `forward`, `embedding_dim`, `enable_mc_dropout`). Everything else in `models/` depends on this.
4. **`src/cropml/models/{mobilenet_v3,efficientnet_b0,vit_small}.py`** — the three concrete backbones against the interface, using `timm` for pretrained weights (free, no API key, downloads once from HuggingFace/timm’s hub).
5. **`src/cropml/models/hierarchical_head.py`** — crop → disease head sitting on top of any backbone’s embedding; depends on step 4’s output shape.
6. **`src/cropml/crop_rejection.py`** — the crop-level abstain/invalid-combo logic; depends on knowing the hierarchical head’s output structure (step 5), independent of training.
7. **`src/cropml/training/dataset.py`** — wraps step 2.2’s `Sample` lists into a `torch.utils.data.Dataset`, the bridge between the data layer and training. Must exist before augmentation (step 8) or the train loop (step 9).
8. **`src/cropml/training/augmentation.py`** — transform pipelines, including the ablation flags (with/without color aug, with/without GAN data) the “Done when” results table needs.
9. **`src/cropml/training/tracking.py`** — the MLflow wrapper; written before the train loop so the loop can log to it from the start rather than being retrofitted.
10. **`src/cropml/training/train_loop.py`** — ties backbone + head + dataset + augmentation + tracking together; the first point where an actual model can be trained end-to-end.
11. **`src/cropml/uncertainty/temperature_scaling.py`**, then **`mc_dropout.py`**, **`deep_ensemble.py`**, **`conformal.py`** — each wraps a *trained* model (step 10’s output), so they come after training exists; temperature scaling first since it’s simplest and other methods can be compared against it.
12. **`src/cropml/ood/{msp,energy,mahalanobis,odin}.py`** — OOD scorers, each operating on model outputs/features; independent of each other, come after a trained model exists to score.
13. **`src/cropml/evaluation/metrics.py`** — accuracy, macro-F1, ECE, OOD AUROC/FPR95, risk-coverage, per-class rare-disease metrics; depends on having uncertainty (step 11) and OOD (step 12) outputs to measure.
14. **`src/cropml/evaluation/results_table.py`** — aggregates metrics (step 13) across models × methods × target domains × seeds into the final table.
15. **`configs/experiment/{default,augmentation_ablation}.yaml`** — experiment configs tying everything together; written once the pieces they configure exist.
16. **`scripts/train.py`** — CLI for a single training run.
17. **`scripts/run_experiment_table.py`** — the actual “one command” the Done-when criterion requires; orchestrates multiple training runs (models × methods × seeds) and calls `results_table.py`.
18. **Tests, last** — `test_backbones.py`, `test_hierarchical_head.py`, `test_uncertainty_methods.py`, `test_ood_scorers.py`, `test_metrics.py`, `test_crop_rejection.py`, each testable with tiny synthetic tensors/datasets rather than real images, so they don’t require real data or real training time.

