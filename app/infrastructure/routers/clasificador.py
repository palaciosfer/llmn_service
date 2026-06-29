from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.application.services import ClasificadorService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import ImagenInvalidaError, ModeloNoDisponibleError
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_clasificador_service
from app.infrastructure.schemas import ClasificacionResponse, ErrorResponse
from app.infrastructure.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["Clasificador CNN"])

_TIPOS_IMAGEN = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


@router.post(
    "/clasificar",
    response_model=ClasificacionResponse,
    summary="Clasificación de imagen con CNN (solo EfficientNet-B4)",
    responses={
        400: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def clasificar(
    imagen: UploadFile = File(..., description="Imagen de la planta"),
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> ClasificacionResponse:
    settings = get_settings()

    if imagen.content_type not in _TIPOS_IMAGEN:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo no soportado: {imagen.content_type}",
        )

    contenido = await imagen.read()

    max_bytes = settings.MAX_IMAGEN_MB * 1024 * 1024
    if len(contenido) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Imagen excede el límite de {settings.MAX_IMAGEN_MB} MB",
        )

    service: ClasificadorService = get_clasificador_service()

    try:
        resultado = await run_in_threadpool(
            service.clasificar, imagen_bytes=contenido
        )
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except ImagenInvalidaError as exc:
        raise HTTPException(status_code=400, detail=exc.mensaje)

    return ClasificacionResponse(
        cultivo=resultado.cultivo,
        enfermedad=resultado.enfermedad,
        confianza=resultado.confianza,
        clase_cnn=resultado.clase_cnn,
        confianza_baja=resultado.confianza_baja,
    )
