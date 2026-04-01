from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import get_logger, setup_logging
from app.engine.manager import initialize_engine_manager
from fastapi import HTTPException

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        started_at = time.perf_counter()
        logger.info(
            "request_started request_id=%s method=%s path=%s client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "request_failed request_id=%s method=%s path=%s latency_ms=%s",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_finished request_id=%s method=%s path=%s status_code=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    engine_manager = initialize_engine_manager(settings)
    engine_manager.load()
    logger.info(
        "model_ready backend=%s model=%s serving_on=%s",
        settings.inference_backend,
        settings.model_name,
        settings.server_url,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(api_router)
    return app


app = create_app()
