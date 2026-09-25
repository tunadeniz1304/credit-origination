"""FastAPI application factory for the Anil2 credit origination platform."""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agents.llm_service import LLMCallRecord, register_recorder, startup_banner
from app.core.config import get_settings, validate_llm_settings
from app.core.limiter import limiter
from app.core.logging import get_logger, request_id_var
from app.core.metrics import HTTP_REQUESTS

_STATIC_DIR = Path(__file__).resolve().parent / "static"
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


_pending_llm_calls: deque[LLMCallRecord] = deque(maxlen=10_000)


def _buffer_llm_call(record: LLMCallRecord) -> None:
    """Never write from inside another unit of work (SQLite lock contention)."""
    _pending_llm_calls.append(record)


def _flush_llm_calls() -> None:
    if not _pending_llm_calls:
        return
    from app.db.models import LLMCall
    from app.db.session import session_scope

    batch = []
    while _pending_llm_calls:
        batch.append(_pending_llm_calls.popleft())
    with session_scope() as session:
        session.add_all(LLMCall(**record.model_dump()) for record in batch)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger = get_logger("app.main")
    validate_llm_settings(settings)
    settings.refuse_unsafe_production("api")
    from app.core.users import ensure_demo_users
    from app.db.session import init_db, session_scope

    init_db(settings)
    from sqlalchemy.exc import IntegrityError

    try:
        with session_scope(settings) as session:
            ensure_demo_users(session, settings)
    except IntegrityError:  # another worker process seeded them concurrently
        logger.info("demo users already created by another worker")
    from app.db.session import register_after_commit
    from app.decisioning.models import get_models

    get_models()  # warm models (and SHAP) before serving traffic
    register_recorder(_buffer_llm_call)
    register_after_commit(_flush_llm_calls)
    logger.info(startup_banner(settings))
    logger.info("Application initialised (env=%s)", settings.app_env)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="2.1.0",
        description=(
            "Explainable, human-in-the-loop hybrid credit origination platform: KYC and fraud "
            "pre-checks, intelligent document processing, open-banking cash-flow analytics, "
            "versioned policy rules + calibrated monotone LightGBM PD + WoE scorecard, SHAP "
            "reason codes and counterfactuals, risk-based pricing, underwriter workbench with "
            "authority matrix and four-eyes, KVKK art. 11 objections and model governance."
        ),
        lifespan=lifespan,
    )
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            {"detail": "çok fazla istek, lütfen daha sonra tekrar deneyin"}, status_code=429
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        get_logger("api.error").exception(
            "unhandled error on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            {"detail": "beklenmeyen bir hata oluştu", "request_id": request_id_var.get()},
            status_code=500,
        )

    @app.middleware("http")
    async def _request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id[:64])
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id[:64]
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = CSP
        if settings.app_env == "prod":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        HTTP_REQUESTS.labels(method=request.method, status=str(response.status_code)).inc()
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    from app.api.routes import applications, auth, pricing, system

    for module in (system, auth, applications, pricing):
        app.include_router(module.router)
    _include_optional_routers(app)

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/static/index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    return app


def _include_optional_routers(app: FastAPI) -> None:
    """Routers of later phases (workbench, governance, agent, demo)."""
    import importlib

    for name in ("workbench", "governance", "agent", "demo"):
        try:
            module = importlib.import_module(f"app.api.routes.{name}")
        except ModuleNotFoundError:
            continue
        app.include_router(module.router)


app = create_app()
