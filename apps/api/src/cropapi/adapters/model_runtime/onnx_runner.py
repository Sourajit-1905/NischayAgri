"""ONNX Runtime wrapper: loads a model from the artifacts registry and
runs inference with a timeout guard."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cropapi.core.errors import AppError

_REPO_ROOT = Path(__file__).resolve().parents[6]
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class ModelManifest:
    backbone_name: str
    onnx_path: Path
    input_size: int
    n_crops: int
    n_classes: int
    crops: tuple[str, ...]
    classes: tuple[str, ...]
    weights_sha256: str


@dataclass(frozen=True)
class OnnxOutput:
    crop_logits: np.ndarray      # (batch, n_crops)
    disease_logits: np.ndarray   # (batch, n_classes)
    inference_ms: float


def load_manifest(manifest_path: Path | str | None = None) -> ModelManifest:
    """Loads the manifest for the current model from artifacts/manifest.yaml."""
    if manifest_path is None:
        import yaml
        registry_path = _ARTIFACTS_DIR / "manifest.yaml"
        if not registry_path.exists():
            raise AppError(
                code="model_registry_missing",
                message=f"Model registry not found at {registry_path}",
                status_code=503,
            )
        with registry_path.open(encoding="utf-8") as f:
            registry = yaml.safe_load(f)
        current = registry["current"]
        manifest_path = _REPO_ROOT / registry["models"][current]["manifest_path"]

    p = Path(manifest_path)
    if not p.exists():
        raise AppError(
            code="model_manifest_missing",
            message=f"Model manifest not found at {p}",
            status_code=503,
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    onnx_path = _REPO_ROOT / raw["onnx_path"]
    return ModelManifest(
        backbone_name=raw["backbone_name"],
        onnx_path=onnx_path,
        input_size=raw["input_size"],
        n_crops=raw["n_crops"],
        n_classes=raw["n_classes"],
        crops=tuple(raw["crops"]),
        classes=tuple(raw["classes"]),
        weights_sha256=raw["weights_sha256"],
    )


class OnnxRunner:
    def __init__(self, manifest: ModelManifest, timeout_ms: float = 5000.0) -> None:
        self.manifest = manifest
        self.timeout_ms = timeout_ms
        if not manifest.onnx_path.exists():
            raise AppError(
                code="model_weights_missing",
                message=f"ONNX model not found at {manifest.onnx_path}",
                status_code=503,
            )
        self._session = ort.InferenceSession(
            str(manifest.onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def run(self, images: np.ndarray) -> OnnxOutput:
        """images: (batch, 3, H, W) float32, ImageNet-normalized."""
        t0 = time.perf_counter()
        outputs = self._session.run(
            ["crop_logits", "disease_logits"],
            {"images": images},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > self.timeout_ms:
            raise AppError(
                code="inference_timeout",
                message=(
                    f"Inference took {elapsed_ms:.0f}ms, "
                    f"exceeded {self.timeout_ms:.0f}ms limit"
                ),
                status_code=503,
            )
        return OnnxOutput(
            crop_logits=outputs[0],
            disease_logits=outputs[1],
            inference_ms=elapsed_ms,
        )


