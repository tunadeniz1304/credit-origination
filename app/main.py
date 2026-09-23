"""FastAPI application factory for the Akıllı Kredi Operasyon Ajanı."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import get_logger

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    """Build and wire the FastAPI application."""
    settings = get_settings()
    logger = get_logger("app.main")
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Smart Credit Operations Agent — FastAPI + Celery/Redis backend, "
            "RAG document analysis, resilient external API integrations and a "
            "credit decision engine producing BDDK-style allocation reports."
        ),
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        """Serve the operations dashboard."""
        return RedirectResponse("/static/dashboard.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    logger.info("Application initialized (env=%s)", settings.app_env)
    return app


app = create_app()
