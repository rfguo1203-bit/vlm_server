from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
import uuid

from fastapi import HTTPException, status

from app.core.config import Settings
from app.core.logging import get_logger
from app.engine.manager import EngineManager
from app.schemas.chat import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatImageBase64ContentPart,
    ChatImageContentPart,
    ChatImagePathContentPart,
    ChatMessage,
    ChatMessageResponse,
    ChatTextContentPart,
    UsageInfo,
)
from app.services.image_service import load_image_from_base64
from app.services.image_service import load_image_from_path
from app.services.image_service import normalize_local_image_reference
from app.services.image_service import validate_image_count

logger = get_logger(__name__)


def _build_cache_salt(session_id: str, settings: Settings) -> str:
    digest = hmac.new(
        key=settings.session_cache_secret.encode("utf-8"),
        msg=session_id.encode("utf-8"),
        digestmod=hashlib.sha256,
    )
    return digest.hexdigest()


def _normalize_message_content(
    message: ChatMessage,
    settings: Settings,
) -> tuple[str | list[dict], int]:
    if isinstance(message.content, str):
        content = message.content.strip()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message content must not be empty.",
            )
        return content, 0

    normalized_parts: list[dict] = []
    image_count = 0
    for item in message.content:
        if isinstance(item, ChatTextContentPart):
            text = item.text.strip()
            if text:
                normalized_parts.append({"type": "text", "text": text})
            continue
        if isinstance(item, ChatImagePathContentPart):
            normalized_parts.append(
                {
                    "type": "image_pil",
                    "image_pil": load_image_from_path(item.image_path.path, settings),
                }
            )
            image_count += 1
            continue
        if isinstance(item, ChatImageBase64ContentPart):
            normalized_parts.append(
                {
                    "type": "image_pil",
                    "image_pil": load_image_from_base64(item.image_base64.data, settings),
                }
            )
            image_count += 1
            continue
        if isinstance(item, ChatImageContentPart):
            normalized_parts.append(
                {
                    "type": "image_pil",
                    "image_pil": normalize_local_image_reference(item.image_url.url, settings),
                }
            )
            image_count += 1

    if not normalized_parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No usable content was found in messages.",
        )

    return normalized_parts, image_count


def build_conversation(messages: list[ChatMessage], settings: Settings) -> list[dict]:
    conversation: list[dict] = []
    total_images = 0
    for message in messages:
        content, image_count = _normalize_message_content(message, settings)
        total_images += image_count
        conversation.append({"role": message.role, "content": content})
    validate_image_count(total_images, settings)
    return conversation


def count_images(messages: list[ChatMessage]) -> int:
    total_images = 0
    for message in messages:
        if isinstance(message.content, str):
            continue
        for item in message.content:
            if isinstance(
                item,
                (
                    ChatImagePathContentPart,
                    ChatImageBase64ContentPart,
                    ChatImageContentPart,
                ),
            ):
                total_images += 1
    return total_images


def _extract_generated_text(output) -> str:
    outputs = getattr(output, "outputs", None) or []
    if not outputs:
        return ""
    return (getattr(outputs[0], "text", "") or "").strip()


def _estimate_prompt_tokens(messages: list[ChatMessage]) -> int:
    text_segments: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            text_segments.append(message.content)
            continue
        for item in message.content:
            if isinstance(item, ChatTextContentPart):
                text_segments.append(item.text)
    return len(" ".join(text_segments).split())


def _resolve_max_tokens(
    request: ChatCompletionRequest,
    settings: Settings,
) -> int:
    max_tokens = request.max_tokens or settings.default_max_tokens
    if max_tokens > settings.max_output_tokens_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Requested max_tokens={max_tokens} exceeds the configured limit "
                f"of {settings.max_output_tokens_limit}."
            ),
        )
    return max_tokens


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda out of memory",
        "oom",
        "cublas_status_alloc_failed",
    )
    return any(marker in message for marker in oom_markers)


def _run_inference(
    request: ChatCompletionRequest,
    engine_manager: EngineManager,
    settings: Settings,
) -> ChatCompletionResponse:
    status_info = engine_manager.status
    if not status_info.loaded or engine_manager.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status_info.error_message or "Model is not ready.",
        )

    runtime = engine_manager.engine
    conversation = build_conversation(request.messages, settings)

    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="vLLM is not available in the current environment.",
        ) from exc

    max_tokens = _resolve_max_tokens(request, settings)
    temperature = request.temperature
    if temperature is None:
        temperature = settings.default_temperature

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )
    chat_kwargs = {
        "messages": conversation,
        "sampling_params": sampling_params,
    }
    if request.session_id is not None:
        chat_kwargs["cache_salt"] = _build_cache_salt(request.session_id, settings)

    try:
        results = runtime.engine.chat(**chat_kwargs)
    except TypeError as exc:
        if request.session_id is not None and "cache_salt" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "The current vLLM runtime rejected request-level cache isolation "
                    "(`cache_salt`). Please verify the installed vLLM package/version."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc
    except Exception as exc:
        if _is_oom_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Inference failed due to GPU memory pressure. "
                    "Please reduce image count, image size, or max_tokens and retry."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    generated_text = _extract_generated_text(results[0]) if results else ""
    prompt_tokens = _estimate_prompt_tokens(request.messages)
    completion_tokens = len(generated_text.split())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model or settings.model_name,
        choices=[
            ChatChoice(
                message=ChatMessageResponse(content=generated_text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def create_chat_completion(
    request: ChatCompletionRequest,
    engine_manager: EngineManager,
    settings: Settings,
    request_id: str,
) -> ChatCompletionResponse:
    image_count = count_images(request.messages)
    session_hash_prefix = (
        _build_cache_salt(request.session_id, settings)[:12]
        if request.session_id is not None
        else "none"
    )
    started_at = time.perf_counter()
    logger.info(
        "chat_completion_started request_id=%s model=%s image_count=%s max_tokens=%s temperature=%s session_id_present=%s session_key_hash_prefix=%s message_count=%s cache_reuse_mode=%s",
        request_id,
        request.model or settings.model_name,
        image_count,
        request.max_tokens or settings.default_max_tokens,
        request.temperature if request.temperature is not None else settings.default_temperature,
        request.session_id is not None,
        session_hash_prefix,
        len(request.messages),
        "prefix",
    )
    max_tokens = _resolve_max_tokens(request, settings)
    logger.info(
        "chat_completion_queueing request_id=%s image_count=%s concurrency_limit=%s max_tokens=%s session_id_present=%s session_key_hash_prefix=%s message_count=%s cache_reuse_mode=%s",
        request_id,
        image_count,
        settings.inference_concurrency,
        max_tokens,
        request.session_id is not None,
        session_hash_prefix,
        len(request.messages),
        "prefix",
    )
    queue_started_at = time.perf_counter()
    async with engine_manager.request_semaphore:
        queue_wait_ms = int((time.perf_counter() - queue_started_at) * 1000)
        logger.info(
            "chat_completion_admitted request_id=%s queue_wait_ms=%s concurrency_limit=%s session_id_present=%s session_key_hash_prefix=%s message_count=%s cache_reuse_mode=%s",
            request_id,
            queue_wait_ms,
            settings.inference_concurrency,
            request.session_id is not None,
            session_hash_prefix,
            len(request.messages),
            "prefix",
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_run_inference, request, engine_manager, settings),
                timeout=settings.request_timeout_seconds,
            )
        except TimeoutError as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.warning(
                "chat_completion_timeout request_id=%s timeout_seconds=%s image_count=%s queue_wait_ms=%s latency_ms=%s session_id_present=%s session_key_hash_prefix=%s message_count=%s cache_reuse_mode=%s",
                request_id,
                settings.request_timeout_seconds,
                image_count,
                queue_wait_ms,
                latency_ms,
                request.session_id is not None,
                session_hash_prefix,
                len(request.messages),
                "prefix",
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Inference request timed out. "
                    f"timeout_seconds={settings.request_timeout_seconds}"
                ),
            ) from exc

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "chat_completion_finished request_id=%s model=%s image_count=%s prompt_tokens=%s completion_tokens=%s queue_wait_ms=%s latency_ms=%s session_id_present=%s session_key_hash_prefix=%s message_count=%s cache_reuse_mode=%s",
        request_id,
        response.model,
        image_count,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        queue_wait_ms,
        latency_ms,
        request.session_id is not None,
        session_hash_prefix,
        len(request.messages),
        "prefix",
    )
    return response
