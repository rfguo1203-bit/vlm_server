from fastapi import APIRouter
from fastapi import status as http_status

from app.core.config import get_settings
from app.engine.manager import get_engine_manager
from app.schemas.chat import ChatCompletionRequest
from app.schemas.chat import ChatCompletionResponse
from app.schemas.chat import ReadinessResponse
from app.services.chat_service import create_chat_completion

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
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    settings = get_settings()
    engine_manager = get_engine_manager()
    return create_chat_completion(request, engine_manager, settings)


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
