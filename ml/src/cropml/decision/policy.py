"""Pure decide() function: takes model outputs -> one of four Decision
values from the API contract. No I/O, no model calls - fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    ACCEPT = "ACCEPT"
    ABSTAIN_RETAKE = "ABSTAIN_RETAKE"
    ABSTAIN_EXPERT = "ABSTAIN_EXPERT"
    REJECT_UNKNOWN = "REJECT_UNKNOWN"


@dataclass(frozen=True)
class DecisionConfig:
    crop_confidence_threshold: float = 0.5
    disease_confidence_threshold: float = 0.4
    ood_score_threshold: float = 0.7
    image_quality_threshold: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict) -> DecisionConfig:
        return cls(
            crop_confidence_threshold=float(raw["crop_confidence_threshold"]),
            disease_confidence_threshold=float(raw["disease_confidence_threshold"]),
            ood_score_threshold=float(raw["ood_score_threshold"]),
            image_quality_threshold=float(raw["image_quality_threshold"]),
        )


@dataclass(frozen=True)
class DecisionResult:
    action: Decision
    reason: str


def decide(
    crop_confidence: float,
    disease_confidence: float,
    ood_score: float,
    image_quality: float,
    config: DecisionConfig,
) -> DecisionResult:
    """Pure function: all inputs are floats in [0, 1] (higher = better/more
    confident, EXCEPT ood_score where higher = more likely OOD).

    Decision priority order (first matching rule wins):
    1. Bad image quality -> ABSTAIN_RETAKE (can't trust any model output)
    2. OOD -> REJECT_UNKNOWN (not a supported crop/leaf at all)
    3. Low crop confidence -> ABSTAIN_EXPERT (uncertain what crop this is)
    4. Low disease confidence -> ABSTAIN_EXPERT (uncertain which disease)
    5. Everything passes -> ACCEPT
    """
    if image_quality < config.image_quality_threshold:
        return DecisionResult(Decision.ABSTAIN_RETAKE, "image_quality_insufficient")
    if ood_score >= config.ood_score_threshold:
        return DecisionResult(Decision.REJECT_UNKNOWN, "ood_score_above_threshold")
    if crop_confidence < config.crop_confidence_threshold:
        return DecisionResult(Decision.ABSTAIN_EXPERT, "low_crop_confidence")
    if disease_confidence < config.disease_confidence_threshold:
        return DecisionResult(Decision.ABSTAIN_EXPERT, "low_disease_confidence")
    return DecisionResult(Decision.ACCEPT, "high_confidence_in_distribution")