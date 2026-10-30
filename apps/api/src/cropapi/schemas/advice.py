from pydantic import BaseModel

from cropapi.schemas.common import AdviceMode, Decision, SeverityLevel


class AdviceRequest(BaseModel):
    request_id: str
    crop: str
    disease: str
    severity: SeverityLevel
    confidence: float
    decision: Decision
    region: str | None = None


class AdviceSection(BaseModel):
    title: str
    body: str


class Citation(BaseModel):
    id: str
    title: str
    source: str
    url: str | None = None
    snippet: str | None = None


class AdviceResponse(BaseModel):
    mode: AdviceMode
    summary: str
    sections: list[AdviceSection]
    citations: list[Citation]
    safety_notice: str

