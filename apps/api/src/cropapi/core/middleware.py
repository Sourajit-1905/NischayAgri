import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from cropapi.core.logging import get_logger, new_request_id, request_id_ctx

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Assigns a request ID, logs request/response, and times the request.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming_id = request.headers.get("x-request-id")
        rid = incoming_id or new_request_id()
        token = request_id_ctx.set(rid)

        start = time.perf_counter()
        logger.info("request_started", method=request.method, path=request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            logger.error("request_failed", method=request.method, path=request.url.path)
            request_id_ctx.reset(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = rid
        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        request_id_ctx.reset(token)
        return response