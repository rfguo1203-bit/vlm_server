from fastapi import APIRouter, Request
from fastapi import status as http_status

from app.core.config import get_settings
from app.engine.manager import get_engine_manager
from app.schemas.chat import ChatCompletionRequest
from app.schemas.chat import ChatCompletionResponse
from app.schemas.chat import CacheResetRequest
from app.schemas.chat import CacheResetResponse
from app.schemas.chat import ReadinessResponse
from app.services.chat_service import create_chat_completion
from app.services.chat_service import reset_runtime_caches

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz() -> ReadinessResponse:
    status_info = get_engine_manager().status
    return ReadinessResponse(
        status="ready" if status_info.loaded else "not_ready",
        model=status_info.model_name,
        backend=status_info.backend,
        loaded=status_info.loaded,
        error_message=status_info.error_message,
    )


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=http_status.HTTP_200_OK,
)
async def chat_completions(
    http_request: Request,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    settings = get_settings()
    engine_manager = get_engine_manager()
    return await create_chat_completion(
        request,
        engine_manager,
        settings,
        http_request.state.request_id,
    )


@router.get("/internal/engine-status")
async def engine_status() -> dict[str, str | bool | None]:
    status = get_engine_manager().status
    return {
        "backend": status.backend,
        "model_name": status.model_name,
        "model_path": status.model_path,
        "loaded": status.loaded,
        "error_message": status.error_message,
    }


@router.post(
    "/internal/reset-caches",
    response_model=CacheResetResponse,
    status_code=http_status.HTTP_200_OK,
)
async def reset_caches(
    http_request: Request,
    request: CacheResetRequest,
) -> CacheResetResponse:
    engine_manager = get_engine_manager()
    return await reset_runtime_caches(
        engine_manager=engine_manager,
        request_id=http_request.state.request_id,
        reset_prefix_cache=request.reset_prefix_cache,
        reset_mm_cache=request.reset_mm_cache,
        reset_running_requests=request.reset_running_requests,
    )
