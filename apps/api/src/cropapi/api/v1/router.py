from fastapi import APIRouter

from cropapi.api.v1 import advice, health, meta, predict

api_router = APIRouter()
api_router.include_router(health.router, tags=["meta"])
api_router.include_router(meta.router, tags=["meta"])
api_router.include_router(predict.router, tags=["predict"])
api_router.include_router(advice.router, tags=["advice"])