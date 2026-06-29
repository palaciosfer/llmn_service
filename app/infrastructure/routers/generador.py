from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.application.services import GeneradorService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import ModeloNoDisponibleError, OllamaNoDisponibleError
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_generador_service
from app.infrastructure.schemas import (
    ErrorResponse,
    GenerarRequest,
    GenerarResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Generador LLM"])


@router.post(
    "/generar",
    response_model=GenerarResponse,
    summary="Generación directa de texto con LLM (sin RAG)",
    responses={
        503: {"model": ErrorResponse},
    },
)
async def generar(
    body: GenerarRequest,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> GenerarResponse:
    service: GeneradorService = get_generador_service()

    try:
        texto = await run_in_threadpool(
            service.generar,
            prompt=body.prompt,
            temperatura=body.temperatura,
        )
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except OllamaNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)

    return GenerarResponse(texto=texto)
