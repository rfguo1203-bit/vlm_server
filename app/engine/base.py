from dataclasses import dataclass


@dataclass(slots=True)
class EngineStatus:
    backend: str
    model_name: str
    model_path: str
    loaded: bool = False
    error_message: str | None = None
