from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cropapi.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for domain errors with a stable error code."""

    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


def _error_body(code: str, message: str, details: dict | None = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id_ctx.get(),
            "details": details or {},
        }
    }


async def app_error_handler(_request: Request, exc: AppError) -> ORJSONResponse:
    logger.warning("app_error", code=exc.code, message=exc.message, details=exc.details)
    return ORJSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.code, exc.message, exc.details),
    )


async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> ORJSONResponse:
    logger.warning("validation_error", errors=exc.errors())
    return ORJSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body("validation_error", "Request validation failed", {"errors": exc.errors()}),
    )


async def http_exception_handler(
    _request: Request, exc: StarletteHTTPException
) -> ORJSONResponse:
    logger.warning("http_exception", status_code=exc.status_code, detail=exc.detail)
    return ORJSONResponse(
        status_code=exc.status_code,
        content=_error_body(f"http_{exc.status_code}", str(exc.detail)),
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> ORJSONResponse:
    logger.error("unhandled_exception", error=str(exc), exc_info=exc)
    return ORJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body("internal_error", "An unexpected error occurred"),
    )