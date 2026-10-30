from pydantic import BaseModel, Field

from cropapi.schemas.common import Decision, SeverityLevel


class CropPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class DiseaseCandidate(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class DiseasePrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[DiseaseCandidate]


class SeverityPrediction(BaseModel):
    level: SeverityLevel
    confidence: float = Field(ge=0, le=1)


class UncertaintyInfo(BaseModel):
    calibrated_confidence: float = Field(ge=0, le=1)
    entropy: float = Field(ge=0)
    method: str


class OodInfo(BaseModel):
    score: float
    is_ood: bool


class DecisionInfo(BaseModel):
    action: Decision
    reason: str


class ImageQuality(BaseModel):
    blur: float = Field(ge=0, le=1)
    lighting: float = Field(ge=0, le=1)
    framing: float = Field(ge=0, le=1)
    usable: bool


class Explanation(BaseModel):
    heatmap_png_base64: str | None = None


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    crop: CropPrediction
    disease: DiseasePrediction
    severity: SeverityPrediction
    uncertainty: UncertaintyInfo
    ood: OodInfo
    decision: DecisionInfo
    image_quality: ImageQuality
    explanation: Explanation | None = None