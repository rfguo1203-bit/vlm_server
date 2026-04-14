from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ImageUrlPayload(BaseModel):
    url: str


class ImagePathPayload(BaseModel):
    path: str


class ImageBase64Payload(BaseModel):
    data: str
    mime_type: str | None = None


class ChatTextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlPayload


class ChatImagePathContentPart(BaseModel):
    type: Literal["image_path"]
    image_path: ImagePathPayload


class ChatImageBase64ContentPart(BaseModel):
    type: Literal["image_base64"]
    image_base64: ImageBase64Payload


ChatContentPart = (
    ChatTextContentPart
    | ChatImageContentPart
    | ChatImagePathContentPart
    | ChatImageBase64ContentPart
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ChatContentPart]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    session_id: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    enable_thinking: bool | None = None

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")
        return self


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessageResponse
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: str
    backend: str
    loaded: bool
    error_message: str | None = None


class CacheResetRequest(BaseModel):
    reset_prefix_cache: bool = True
    reset_mm_cache: bool = True
    reset_running_requests: bool = False


class CacheResetResponse(BaseModel):
    ok: bool = True
    backend: str
    reset_prefix_cache: bool
    reset_mm_cache: bool
    reset_running_requests: bool
