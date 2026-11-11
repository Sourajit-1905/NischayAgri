"""
Predict service: routes to real inference or stubs based on settings.
"""

from __future__ import annotations

import sys
from itertools import count
from pathlib import Path

from cropapi.core.security import check_image_dimensions, check_upload_size, sniff_image_type
from cropapi.schemas.common import Decision, SeverityLevel
from cropapi.schemas.predict import (
    CropPrediction,
    DecisionInfo,
    DiseaseCandidate,
    DiseasePrediction,
    Explanation,
    ImageQuality,
    OodInfo,
    PredictionResponse,
    SeverityPrediction,
    UncertaintyInfo,
)
from cropapi.services.stub_data import cycle_stub_decision, get_stub_prediction

_ML_SRC = Path(__file__).resolve().parents[5] / "ml" / "src"
if _ML_SRC.exists() and str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

_call_counter = count()


def _build_response_from_inference(result, decision_result, manifest) -> PredictionResponse:
    """
    Builds a contract-conformant PredictionResponse from inference outputs.
    """
    crop_idx = int(result.crop_probs.argmax())
    disease_idx = int(result.disease_probs.argmax())

    # Top-3 disease candidates
    top3_idx = result.disease_probs.argsort()[::-1][:3]
    top_k = [
        DiseaseCandidate(label=manifest.classes[i], confidence=float(result.disease_probs[i]))
        for i in top3_idx
    ]

    # Severity: not yet wired to real severity head; stub at "moderate"
    # until severity_service is implemented (Phase 2.6 extension or later).
    severity = SeverityPrediction(level=SeverityLevel.MODERATE, confidence=0.0)

    # Map cropml Decision -> contract Decision
    from cropml.decision.policy import Decision as MLDecision
    action_map = {
        MLDecision.ACCEPT: Decision.ACCEPT,
        MLDecision.ABSTAIN_RETAKE: Decision.ABSTAIN_RETAKE,
        MLDecision.ABSTAIN_EXPERT: Decision.ABSTAIN_EXPERT,
        MLDecision.REJECT_UNKNOWN: Decision.REJECT_UNKNOWN,
    }

    return PredictionResponse(
        request_id=result.request_id,
        model_version=result.model_version,
        crop=CropPrediction(
            label=manifest.crops[crop_idx],
            confidence=float(result.crop_probs[crop_idx]),
        ),
        disease=DiseasePrediction(
            label=manifest.classes[disease_idx],
            confidence=float(result.disease_probs[disease_idx]),
            top_k=top_k,
        ),
        severity=severity,
        uncertainty=UncertaintyInfo(
            calibrated_confidence=float(result.disease_probs[disease_idx]),
            entropy=result.entropy,
            method="softmax",
        ),
        ood=OodInfo(score=result.ood_score, is_ood=result.ood_score >= 0.7),
        decision=DecisionInfo(
            action=action_map[decision_result.action],
            reason=decision_result.reason,
        ),
        image_quality=ImageQuality(
            blur=result.quality.blur,
            lighting=result.quality.lighting,
            framing=result.quality.framing,
            usable=result.quality.usable,
        ),
        explanation=Explanation(heatmap_png_base64=None),
    )


class PredictService:
    def __init__(
        self,
        max_upload_bytes: int,
        use_stubs: bool = True,
    ) -> None:
        self._max_upload_bytes = max_upload_bytes
        self._use_stubs = use_stubs
        self._runner = None
        self._manifest = None
        self._inf_svc = None
        self._dec_svc = None

        if not use_stubs:
            self._load_real_services()

    def _load_real_services(self) -> None:
        from cropapi.adapters.model_runtime.onnx_runner import OnnxRunner, load_manifest
        from cropapi.services.decision_service import DecisionService
        from cropapi.services.inference_service import InferenceService

        self._manifest = load_manifest()
        self._runner = OnnxRunner(self._manifest)
        self._inf_svc = InferenceService(self._runner, self._manifest)
        self._dec_svc = DecisionService()

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        check_upload_size(image_bytes, self._max_upload_bytes)
        sniff_image_type(image_bytes)
        check_image_dimensions(width=256, height=256)

        if self._use_stubs:
            decision = cycle_stub_decision(next(_call_counter))
            return get_stub_prediction(decision)

        import io

        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            w, h = img.size
        check_image_dimensions(width=w, height=h)

        with Image.open(io.BytesIO(image_bytes)) as img:
            result = self._inf_svc.run(img)

        decision_result = self._dec_svc.apply(result)
        return _build_response_from_inference(result, decision_result, self._manifest)

    