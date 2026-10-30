from fastapi import APIRouter, Depends

from cropapi.schemas.advice import AdviceRequest, AdviceResponse
from cropapi.services.advice_service import AdviceService

router = APIRouter()


def get_advice_service() -> AdviceService:
    return AdviceService()


@router.post("/advice", response_model=AdviceResponse)
async def post_advice(
    body: AdviceRequest,
    service: AdviceService = Depends(get_advice_service),
) -> AdviceResponse:
    return await service.get_advice(body)