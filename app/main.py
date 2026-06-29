import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.core.exceptions import DomainException
from app.infrastructure.dependencies import init_container
from app.infrastructure.middleware import MetricsMiddleware
from app.infrastructure.routers import (
    clasificador,
    consultar,
    dev,
    diagnostico,
    embeddings,
    generador,
    health,
)
from app.infrastructure.settings import get_settings


def _configurar_logging(nivel: str) -> None:
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configurar_logging(settings.LOG_LEVEL)

    logger = logging.getLogger("app.startup")
    logger.info("Iniciando microservicio de diagnóstico agrícola...")

    container = init_container(settings)
    container.warmup()

    logger.info("Microservicio listo en %s:%s", settings.HOST, settings.PORT)
    yield
    logger.info("Apagando microservicio...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Diagnóstico Agrícola API",
        description=(
            "Microservicio de diagnóstico agrícola con RAG "
            "(CNN + Sentence-BERT + Ollama/Qwen). "
            "Arquitectura hexagonal — consumible desde app móvil."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS (para consumo desde móvil) ─────────────────
    origins = [
        o.strip()
        for o in settings.CORS_ORIGINS.split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Métricas Prometheus ─────────────────────────────
    app.add_middleware(MetricsMiddleware)
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Manejador global de excepciones de dominio ──────
    @app.exception_handler(DomainException)
    async def _domain_exception_handler(
        request: Request, exc: DomainException
    ) -> JSONResponse:
        status_map = {
            "MODELO_NO_DISPONIBLE": 503,
            "OLLAMA_NO_DISPONIBLE": 503,
            "TIMEOUT_INFERENCIA": 504,
            "IMAGEN_INVALIDA": 400,
            "AUTENTICACION_ERROR": 401,
            "AUTORIZACION_ERROR": 403,
        }
        status_code = status_map.get(exc.codigo, 500)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.mensaje,
                "codigo": exc.codigo,
            },
        )

    # ── Routers ─────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(consultar.router)
    app.include_router(diagnostico.router)
    app.include_router(clasificador.router)
    app.include_router(embeddings.router)
    app.include_router(generador.router)

    if settings.DEV_MODE:
        app.include_router(dev.router)

    return app


app = create_app()
