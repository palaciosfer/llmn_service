from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.application.services import EmbeddingsService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import ModeloNoDisponibleError
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_embeddings_service
from app.infrastructure.schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    summary="Generar embeddings con Sentence-BERT (384 dims)",
    responses={503: {"model": ErrorResponse}},
)
async def generar_embeddings(
    body: EmbeddingsRequest,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> EmbeddingsResponse:
    service: EmbeddingsService = get_embeddings_service()

    try:
        embeddings, dimension = await run_in_threadpool(
            service.generar_embeddings, textos=body.textos
        )
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)

    return EmbeddingsResponse(embeddings=embeddings, dimension=dimension)
