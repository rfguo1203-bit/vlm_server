from __future__ import annotations

from app.core.config import Settings
from app.engine.base import EngineStatus
from app.engine.vllm_engine import VLLMEngine


class EngineManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
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

    def load(self) -> None:
        if self._settings.skip_model_load:
            self._status.loaded = False
            self._status.error_message = "Model loading skipped by configuration."
            return

        backend = self._settings.inference_backend.lower()
        if backend != "vllm":
            raise RuntimeError(
                f"Unsupported inference backend `{self._settings.inference_backend}`. "
                "Only `vllm` is implemented in the current version."
            )

        runtime = VLLMEngine(self._settings)

        try:
            runtime.load()
        except Exception as exc:
            self._status.loaded = False
            self._status.error_message = str(exc)
            raise RuntimeError(
                "Model load failed during application startup. "
                f"backend={backend}, model={self._settings.model_name}, "
                f"path={self._settings.model_path}. Root cause: {exc}"
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
