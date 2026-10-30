from typing import Any

from fastapi import APIRouter

from cropapi.config import get_settings

router = APIRouter()


@router.get("/health")
async def get_health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "model_loaded": settings.use_stubs is False,
        "version": "1.0.0",
    }