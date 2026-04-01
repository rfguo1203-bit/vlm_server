from __future__ import annotations

import time
import uuid

from fastapi import HTTPException, status

from app.core.config import Settings
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


def create_chat_completion(
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

    max_tokens = request.max_tokens or settings.default_max_tokens
    temperature = request.temperature
    if temperature is None:
        temperature = settings.default_temperature

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    try:
        results = runtime.engine.chat(messages=conversation, sampling_params=sampling_params)
    except Exception as exc:
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
