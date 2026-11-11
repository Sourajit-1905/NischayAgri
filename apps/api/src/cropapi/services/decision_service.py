"""
Wraps ml/decision/policy.py decide() for use in the API service layer.
Translates RawInferenceResult into the four Decision values from the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ml/src must be on the path so the API can import cropml.decision.
_ML_SRC = Path(__file__).resolve().parents[5] / "ml" / "src"
if _ML_SRC.exists() and str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from cropml.decision.policy import DecisionConfig, DecisionResult, decide

from cropapi.services.inference_service import RawInferenceResult


class DecisionService:
    def __init__(self, config: DecisionConfig | None = None) -> None:
        self._config = config or DecisionConfig()

    def apply(self, result: RawInferenceResult) -> DecisionResult:
        crop_confidence = float(result.crop_probs.max())
        disease_confidence = float(result.disease_probs.max())
        image_quality = float(1.0 - result.quality.blur)

        return decide(
            crop_confidence=crop_confidence,
            disease_confidence=disease_confidence,
            ood_score=result.ood_score,
            image_quality=image_quality,
            config=self._config,
        )