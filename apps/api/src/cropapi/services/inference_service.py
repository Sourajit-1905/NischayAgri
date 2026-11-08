"""
Full inference pipeline: image → PredictionResponse fields.
Orchestrates ONNX runner, softmax/temperature scaling, OOD scoring,
image quality, and crop-rejection gate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cropapi.adapters.model_runtime.onnx_runner import ModelManifest, OnnxRunner
from cropapi.core.errors import AppError
from cropapi.services.quality_service import QualityScore, score_quality


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _msp_ood_score(probs: np.ndarray) -> float:
    """1 - max(softmax probs): higher = more likely OOD."""
    return float(1.0 - probs.max())


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-(probs * np.log(probs + eps)).sum())


def _preprocess(image: Image.Image, input_size: int) -> np.ndarray:
    """Resize, center-crop, normalize with ImageNet stats, return (1,3,H,W) float32."""
    resize_size = int(input_size * 1.14)
    image = image.convert("RGB").resize((resize_size, resize_size), Image.BILINEAR)
    left = (resize_size - input_size) // 2
    image = image.crop((left, left, left + input_size, left + input_size))
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


@dataclass(frozen=True)
class RawInferenceResult:
    request_id: str
    crop_probs: np.ndarray       # (n_crops,)
    disease_probs: np.ndarray    # (n_classes,)
    crop_logits: np.ndarray
    disease_logits: np.ndarray
    ood_score: float
    entropy: float
    quality: QualityScore
    inference_ms: float
    model_version: str


class InferenceService:
    def __init__(self, runner: OnnxRunner, manifest: ModelManifest) -> None:
        self._runner = runner
        self._manifest = manifest

    def run(self, image: Image.Image) -> RawInferenceResult:
        quality = score_quality(image)

        image_arr = _preprocess(image, self._manifest.input_size)
        output = self._runner.run(image_arr)

        crop_probs = _softmax(output.crop_logits[0])
        disease_probs = _softmax(output.disease_logits[0])
        ood_score = _msp_ood_score(disease_probs)
        entropy = _entropy(disease_probs)

        return RawInferenceResult(
            request_id=str(uuid.uuid4()),
            crop_probs=crop_probs,
            disease_probs=disease_probs,
            crop_logits=output.crop_logits[0],
            disease_logits=output.disease_logits[0],
            ood_score=ood_score,
            entropy=entropy,
            quality=quality,
            inference_ms=output.inference_ms,
            model_version=f"{self._manifest.backbone_name}_seed0",
        )

    @property
    def crops(self) -> tuple[str, ...]:
        return self._manifest.crops

    @property
    def classes(self) -> tuple[str, ...]:
        return self._manifest.classes