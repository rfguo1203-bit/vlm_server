from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.engine.manager import initialize_engine_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    engine_manager = initialize_engine_manager(settings)
    engine_manager.load()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
