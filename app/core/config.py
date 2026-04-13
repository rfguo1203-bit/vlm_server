import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "vlm-server"
    app_env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8972
    log_level: str = "info"

    model_name: str = "gemma-4-E4B-it"
    model_path: str | None = None
    additional_model_paths_json: str = "{}"
    inference_backend: str = "vllm"
    tensor_parallel_size: int = 2
    skip_model_load: bool = False
    enable_prefix_caching: bool = True
    session_cache_secret: str = "dev-session-cache-secret"

    max_input_images: int = 4
    max_image_bytes: int = 20 * 1024 * 1024
    default_max_tokens: int = 512
    max_output_tokens_limit: int = 10240
    default_temperature: float = 0.1
    inference_concurrency: int = 1

    request_timeout_seconds: int = 120
    allow_file_path_input: bool = True
    allow_base64_input: bool = True

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def supported_model_paths(self) -> dict[str, str]:
        model_paths = {
            "Qwen3.5-27B": "/home/user/g00806422/data/weight/Qwen3.5-27B",
            "gemma-4-26B-A4B-it": "/home/user/g00806422/data/weight/gemma-4-26B-A4B-it",
            "gemma-4-E4B-it": "/home/user/g00806422/data/weight/gemma-4-26B-A4B-it",
        }
        if self.model_path:
            model_paths[self.model_name] = self.model_path
        model_paths.update(self._parse_additional_model_paths())
        return model_paths

    def resolve_model_path(self, model_name: str) -> str:
        model_path = self.supported_model_paths.get(model_name)
        if model_path is None:
            supported_models = ", ".join(sorted(self.supported_model_paths.keys()))
            raise ValueError(
                f"Unsupported model `{model_name}`. Supported models: {supported_models}"
            )
        return model_path

    def _parse_additional_model_paths(self) -> dict[str, str]:
        raw = self.additional_model_paths_json.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid ADDITIONAL_MODEL_PATHS_JSON. Expecting a JSON object."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "Invalid ADDITIONAL_MODEL_PATHS_JSON. Expecting a JSON object."
            )
        normalized: dict[str, str] = {}
        for model_name, model_path in parsed.items():
            if not isinstance(model_name, str) or not isinstance(model_path, str):
                raise ValueError(
                    "Invalid ADDITIONAL_MODEL_PATHS_JSON. "
                    "All model names and model paths must be strings."
                )
            normalized[model_name] = model_path
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
