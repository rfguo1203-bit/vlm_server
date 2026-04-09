from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.engine.base import EngineStatus


class VLLMEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._status = EngineStatus(
            backend="vllm",
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
        model_path = Path(self._settings.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Model path does not exist: {model_path}")

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed in the current environment. "
                "Please activate the server conda env `vllm` before starting the service."
            ) from exc

        try:
            llm_kwargs = dict(
                model=str(model_path),
                trust_remote_code=True,
                dtype="bfloat16",
                tensor_parallel_size=self._settings.tensor_parallel_size,
            )
            if self._settings.enable_prefix_caching:
                llm_kwargs["enable_prefix_caching"] = True
            self._engine = LLM(**llm_kwargs)
        except TypeError as exc:
            if self._settings.enable_prefix_caching and "enable_prefix_caching" in str(exc):
                raise RuntimeError(
                    "The current vLLM runtime rejected `enable_prefix_caching`. "
                    "Please verify the installed vLLM package really matches the expected version."
                ) from exc
            raise RuntimeError(
                f"Failed to initialize vLLM for model `{self._settings.model_name}` "
                f"from `{model_path}` with tensor_parallel_size="
                f"{self._settings.tensor_parallel_size}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize vLLM for model `{self._settings.model_name}` "
                f"from `{model_path}` with tensor_parallel_size="
                f"{self._settings.tensor_parallel_size}: {exc}"
            ) from exc

        self._status.loaded = True
        self._status.error_message = None
