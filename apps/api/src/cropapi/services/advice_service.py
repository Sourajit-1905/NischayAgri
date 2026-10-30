from cropapi.schemas.advice import AdviceRequest, AdviceResponse
from cropapi.schemas.common import Decision
from cropapi.services.stub_data import get_stub_advice_for_decision


class AdviceService:
    """Stub implementation. Phase 2.7 replaces this with the RAG pipeline
    (retrieval + Gemini adapter) behind the same `get_advice` interface."""

    async def get_advice(self, request: AdviceRequest) -> AdviceResponse:
        decision = Decision(request.decision)
        return get_stub_advice_for_decision(decision)