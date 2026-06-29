from fastapi import APIRouter, HTTPException

from app.infrastructure.dependencies import get_container
from app.infrastructure.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])

_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness — el proceso está vivo",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=_VERSION)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness — modelos y Ollama disponibles",
)
async def ready() -> ReadyResponse:
    container = get_container()

    cnn_ok = container.cnn_adapter.esta_disponible()
    bert_ok = container.bert_adapter.esta_disponible()
    ollama_ok = container.ollama_adapter.ollama_responde()
    modulos_ok = container.orquestador_adapter.esta_disponible()

    todo_ok = cnn_ok and bert_ok and ollama_ok and modulos_ok
    status_str = "ready" if todo_ok else "degraded"

    resp = ReadyResponse(
        status=status_str,
        cnn_disponible=cnn_ok,
        bert_disponible=bert_ok,
        ollama_disponible=ollama_ok,
        modulos_cargados=modulos_ok,
    )

    if not todo_ok:
        raise HTTPException(status_code=503, detail=resp.model_dump())

    return resp
