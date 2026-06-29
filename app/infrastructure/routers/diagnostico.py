from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.application.services import DiagnosticoService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import (
    ImagenInvalidaError,
    ModeloNoDisponibleError,
    OllamaNoDisponibleError,
    TimeoutInferenciaError,
)
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_diagnostico_service
from app.infrastructure.schemas import DiagnosticoResponse, ErrorResponse
from app.infrastructure.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["Diagnóstico"])

_TIPOS_IMAGEN = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "/diagnosticar",
    response_model=DiagnosticoResponse,
    summary="Diagnóstico agrícola completo (CNN + RAG + LLM)",
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def diagnosticar(
    imagen: UploadFile = File(..., description="Imagen de la planta"),
    texto: str = Form(default="", description="Síntomas observados"),
    cultivos: str = Form(
        default="", description="Cultivos separados por coma"
    ),
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> DiagnosticoResponse:
    settings = get_settings()

    if imagen.content_type not in _TIPOS_IMAGEN:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo no soportado: {imagen.content_type}. "
            f"Aceptados: {', '.join(sorted(_TIPOS_IMAGEN))}",
        )

    contenido = await imagen.read()

    max_bytes = settings.MAX_IMAGEN_MB * 1024 * 1024
    if len(contenido) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Imagen excede el límite de {settings.MAX_IMAGEN_MB} MB",
        )

    if len(contenido) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La imagen está vacía",
        )

    lista_cultivos = (
        [c.strip() for c in cultivos.split(",") if c.strip()]
        if cultivos
        else None
    )

    service: DiagnosticoService = get_diagnostico_service()

    try:
        resultado = await run_in_threadpool(
            service.diagnosticar,
            imagen_bytes=contenido,
            texto=texto,
            rol=usuario.rol.value,
            cultivos=lista_cultivos,
        )
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except OllamaNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except TimeoutInferenciaError as exc:
        raise HTTPException(status_code=504, detail=exc.mensaje)
    except ImagenInvalidaError as exc:
        raise HTTPException(status_code=400, detail=exc.mensaje)

    return DiagnosticoResponse(
        modo=resultado.modo,
        diagnostico=resultado.diagnostico,
        sintomas=resultado.sintomas,
        avisos=resultado.avisos,
        n_documentos=resultado.n_documentos,
        respuesta=resultado.respuesta,
    )
