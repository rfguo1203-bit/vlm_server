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
    port: int = 8000
    log_level: str = "info"

    model_name: str = "Qwen3.5-27B"
    model_path: str = "/home/user/g00806422/data/weight/Qwen3.5-27B"
    inference_backend: str = "vllm"
    skip_model_load: bool = False

    max_input_images: int = 4
    max_image_bytes: int = 20 * 1024 * 1024
    default_max_tokens: int = 512
    default_temperature: float = 0.1
    inference_concurrency: int = 1

    request_timeout_seconds: int = 120
    allow_file_path_input: bool = True
    allow_base64_input: bool = True

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
