from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.engine.base import EngineStatus
from app.engine.vllm_engine import VLLMEngine


class EngineManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._request_semaphore: asyncio.Semaphore | None = None
        self._status = EngineStatus(
            backend=settings.inference_backend,
            model_name=settings.model_name,
            model_path=settings.model_path,
            loaded=False,
        )

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def engine(self):
        return self._engine

    @property
    def request_semaphore(self) -> asyncio.Semaphore:
        if self._request_semaphore is None:
            self._request_semaphore = asyncio.Semaphore(
                max(1, self._settings.inference_concurrency)
            )
        return self._request_semaphore

    def load(self) -> None:
        if self._settings.skip_model_load:
            self._status.loaded = False
            self._status.error_message = "Model loading skipped by configuration."
            return

        self.ensure_model(self._settings.model_name)

    def resolve_model_name(self, requested_model_name: str | None) -> str:
        return requested_model_name or self._settings.model_name

    def ensure_model(self, model_name: str) -> None:
        if self._settings.skip_model_load:
            self._status.loaded = False
            self._status.error_message = "Model loading skipped by configuration."
            return

        if self._status.loaded and self._status.model_name == model_name and self._engine is not None:
            return

        backend = self._settings.inference_backend.lower()
        if backend != "vllm":
            raise RuntimeError(
                f"Unsupported inference backend `{self._settings.inference_backend}`. "
                "Only `vllm` is implemented in the current version."
            )

        try:
            model_path = self._settings.resolve_model_path(model_name)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        runtime = VLLMEngine(
            self._settings,
            model_name=model_name,
            model_path=model_path,
        )
        try:
            runtime.load()
        except Exception as exc:
            self._status.loaded = False
            self._status.error_message = str(exc)
            raise RuntimeError(
                "Model load failed during application startup. "
                f"backend={backend}, model={model_name}, "
                f"path={model_path}. Root cause: {exc}"
            ) from exc

        self._engine = runtime
        self._status = runtime.status


_engine_manager: EngineManager | None = None


def initialize_engine_manager(settings: Settings) -> EngineManager:
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager(settings)
    return _engine_manager


def get_engine_manager() -> EngineManager:
    if _engine_manager is None:
        raise RuntimeError("Engine manager has not been initialized.")
    return _engine_manager
