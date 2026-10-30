from fastapi import APIRouter, Depends, UploadFile

from cropapi.config import Settings, get_settings
from cropapi.schemas.predict import PredictionResponse
from cropapi.services.predict_service import PredictService

router = APIRouter()


def get_predict_service(settings: Settings = Depends(get_settings)) -> PredictService:
    return PredictService(max_upload_bytes=settings.max_upload_bytes)


@router.post("/predict", response_model=PredictionResponse)
async def post_predict(
    image: UploadFile,
    service: PredictService = Depends(get_predict_service),
) -> PredictionResponse:
    image_bytes = await image.read()
    return await service.predict(image_bytes)