from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status as http_status

from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    details: list[dict] | None = None,
) -> JSONResponse:
    payload: dict[str, object] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = _get_request_id(request)
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    code = "bad_request"
    if exc.status_code == http_status.HTTP_503_SERVICE_UNAVAILABLE:
        code = "service_unavailable"
    elif exc.status_code == http_status.HTTP_504_GATEWAY_TIMEOUT:
        code = "request_timeout"
    elif exc.status_code >= 500:
        code = "internal_error"
    logger.warning(
        "http_exception request_id=%s status_code=%s code=%s detail=%s",
        request_id,
        exc.status_code,
        code,
        detail,
    )
    return error_response(
        status_code=exc.status_code,
        code=code,
        message=detail,
        request_id=request_id,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "validation_exception request_id=%s errors=%s",
        _get_request_id(request),
        exc.errors(),
    )
    return error_response(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="validation_error",
        message="Request validation failed.",
        request_id=_get_request_id(request),
        details=exc.errors(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception request_id=%s error=%s",
        _get_request_id(request),
        exc,
    )
    return error_response(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message=f"Unhandled server error: {exc}",
        request_id=_get_request_id(request),
    )
