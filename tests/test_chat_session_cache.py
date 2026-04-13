from __future__ import annotations

import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from fastapi import HTTPException
except ModuleNotFoundError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fake_fastapi = types.SimpleNamespace(
        HTTPException=HTTPException,
        status=types.SimpleNamespace(
            HTTP_400_BAD_REQUEST=400,
            HTTP_500_INTERNAL_SERVER_ERROR=500,
            HTTP_503_SERVICE_UNAVAILABLE=503,
            HTTP_504_GATEWAY_TIMEOUT=504,
        ),
    )
    sys.modules["fastapi"] = fake_fastapi

from app.core.config import Settings
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatImagePathContentPart,
    ChatMessage,
    ChatTextContentPart,
    ImagePathPayload,
)
from app.services.chat_service import _build_cache_salt
from app.services.chat_service import _run_inference
from app.services.chat_service import create_chat_completion
from app.services.chat_service import reset_runtime_caches


class FakeSamplingParams:
    def __init__(self, max_tokens: int, temperature: float):
        self.max_tokens = max_tokens
        self.temperature = temperature


class FakeChatBackend:
    def __init__(self, supports_cache_salt: bool = True) -> None:
        self.calls: list[dict] = []
        self.supports_cache_salt = supports_cache_salt

    def chat(self, messages, sampling_params, cache_salt=None):
        if cache_salt is not None and not self.supports_cache_salt:
            raise TypeError("chat() got an unexpected keyword argument 'cache_salt'")
        self.calls.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "cache_salt": cache_salt,
            }
        )
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="assistant reply")],
            )
        ]


class FakeRuntime:
    def __init__(self, supports_cache_salt: bool = True) -> None:
        self.engine = FakeChatBackend(supports_cache_salt=supports_cache_salt)
        self.reset_prefix_cache_calls: list[bool] = []
        self.reset_mm_cache_calls = 0

    def reset_prefix_cache(self, reset_running_requests: bool = False) -> None:
        self.reset_prefix_cache_calls.append(reset_running_requests)

    def reset_mm_cache(self) -> None:
        self.reset_mm_cache_calls += 1


class FakeEngineManager:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.engine = runtime
        self.status = SimpleNamespace(
            loaded=True,
            error_message=None,
            backend="vllm",
            model_name="test-model",
        )
        self.request_semaphore = asyncio.Semaphore(1)

    def resolve_model_name(self, requested_model_name: str | None) -> str:
        return requested_model_name or self.status.model_name

    def ensure_model(self, model_name: str) -> None:
        if model_name != self.status.model_name:
            raise RuntimeError(f"Unsupported model `{model_name}`.")


class ChatSessionCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            model_name="test-model",
            model_path="/tmp/fake-model",
            request_timeout_seconds=5,
            session_cache_secret="unit-test-secret",
        )

    def _patch_vllm(self):
        fake_vllm = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
        return patch.dict(sys.modules, {"vllm": fake_vllm})

    def test_cache_salt_is_stable_per_session(self) -> None:
        salt_a1 = _build_cache_salt("session-a", self.settings)
        salt_a2 = _build_cache_salt("session-a", self.settings)
        salt_b = _build_cache_salt("session-b", self.settings)

        self.assertEqual(salt_a1, salt_a2)
        self.assertNotEqual(salt_a1, salt_b)

    def test_run_inference_without_session_id_keeps_old_behavior(self) -> None:
        runtime = FakeRuntime()
        engine_manager = FakeEngineManager(runtime)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=32,
            temperature=0.2,
        )

        with self._patch_vllm():
            response = _run_inference(request, engine_manager, self.settings)

        self.assertEqual(response.choices[0].message.content, "assistant reply")
        self.assertIsNone(runtime.engine.calls[0]["cache_salt"])
        self.assertEqual(runtime.engine.calls[0]["messages"], [{"role": "user", "content": "hello"}])

    async def test_multiturn_request_passes_full_history_and_cache_salt(self) -> None:
        runtime = FakeRuntime()
        engine_manager = FakeEngineManager(runtime)
        request = ChatCompletionRequest(
            model="test-model",
            session_id="session-a",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="assistant", content="assistant reply"),
                ChatMessage(role="user", content="follow up"),
            ],
            max_tokens=32,
            temperature=0.2,
        )

        with self._patch_vllm():
            response = await create_chat_completion(
                request=request,
                engine_manager=engine_manager,
                settings=self.settings,
                request_id="req-1",
            )

        self.assertEqual(response.choices[0].message.content, "assistant reply")
        self.assertEqual(len(runtime.engine.calls), 1)
        call = runtime.engine.calls[0]
        self.assertEqual(len(call["messages"]), 4)
        self.assertEqual(call["messages"][0]["content"], "You are helpful.")
        self.assertEqual(call["messages"][-1]["content"], "follow up")
        self.assertEqual(call["cache_salt"], _build_cache_salt("session-a", self.settings))

    def test_session_id_falls_back_when_cache_salt_is_unsupported(self) -> None:
        runtime = FakeRuntime(supports_cache_salt=False)
        engine_manager = FakeEngineManager(runtime)
        request = ChatCompletionRequest(
            model="test-model",
            session_id="session-a",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=32,
        )

        with self._patch_vllm():
            response = _run_inference(request, engine_manager, self.settings)

        self.assertEqual(response.choices[0].message.content, "assistant reply")
        self.assertEqual(len(runtime.engine.calls), 1)
        self.assertIsNone(runtime.engine.calls[0]["cache_salt"])

    def test_multimodal_request_passes_cache_salt(self) -> None:
        runtime = FakeRuntime()
        engine_manager = FakeEngineManager(runtime)
        request = ChatCompletionRequest(
            model="test-model",
            session_id="session-image",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        ChatImagePathContentPart(
                            type="image_path",
                            image_path=ImagePathPayload(path="/tmp/example.png"),
                        ),
                        ChatTextContentPart(type="text", text="describe this image"),
                    ],
                )
            ],
            max_tokens=32,
        )

        with self._patch_vllm(), patch(
            "app.services.chat_service.load_image_from_path",
            return_value="fake-image",
        ), patch(
            "app.services.chat_service.validate_image_count",
            return_value=None,
        ):
            response = _run_inference(request, engine_manager, self.settings)

        self.assertEqual(response.choices[0].message.content, "assistant reply")
        call = runtime.engine.calls[0]
        self.assertEqual(call["cache_salt"], _build_cache_salt("session-image", self.settings))
        self.assertEqual(call["messages"][0]["content"][0]["image_pil"], "fake-image")

    def test_unsupported_model_returns_400(self) -> None:
        runtime = FakeRuntime()
        engine_manager = FakeEngineManager(runtime)
        request = ChatCompletionRequest(
            model="gemma-4-26B-A4B-it",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=32,
        )

        with self._patch_vllm():
            with self.assertRaises(HTTPException) as context:
                _run_inference(request, engine_manager, self.settings)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("Unsupported model", context.exception.detail)

    async def test_reset_runtime_caches_calls_prefix_and_mm_reset(self) -> None:
        runtime = FakeRuntime()
        engine_manager = FakeEngineManager(runtime)

        response = await reset_runtime_caches(
            engine_manager=engine_manager,
            request_id="reset-1",
            reset_prefix_cache=True,
            reset_mm_cache=True,
            reset_running_requests=False,
        )

        self.assertTrue(response.ok)
        self.assertEqual(response.backend, "vllm")
        self.assertEqual(runtime.reset_prefix_cache_calls, [False])
        self.assertEqual(runtime.reset_mm_cache_calls, 1)


if __name__ == "__main__":
    unittest.main()
