"""
Endpoint principal para consumo desde la app móvil.

La CNN corre localmente en el dispositivo; el móvil envía el resultado_cnn
(cultivo, enfermedad, confianza) junto con el texto de síntomas. El servidor
ejecuta: NLP → fusión → búsqueda híbrida (TF-IDF + BERT) → LLM (Ollama/Qwen)
y devuelve diagnóstico, tratamiento, prevención y fuentes.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.application.services import DiagnosticoService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import (
    ModeloNoDisponibleError,
    OllamaNoDisponibleError,
    TimeoutInferenciaError,
)
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_diagnostico_service
from app.infrastructure.schemas import (
    ConsultarRequest,
    ConsultarResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Consulta (móvil)"])


@router.post(
    "/consultar",
    response_model=ConsultarResponse,
    summary="Consulta desde móvil: CNN local + RAG en servidor",
    description=(
        "Recibe el resultado de la CNN ejecutada en el dispositivo móvil "
        "junto con el texto de síntomas del usuario. Ejecuta el pipeline "
        "RAG completo (NLP + fusión + búsqueda + LLM) y devuelve el "
        "diagnóstico con tratamiento y prevención."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Modelo o Ollama no disponible"},
        504: {"model": ErrorResponse, "description": "Timeout en inferencia"},
    },
)
async def consultar(
    body: ConsultarRequest,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> ConsultarResponse:
    service: DiagnosticoService = get_diagnostico_service()

    resultado_cnn = body.resultado_cnn.model_dump()
    cultivos = body.cultivos if body.cultivos else None

    try:
        resultado = await run_in_threadpool(
            service.consultar_con_diagnostico,
            resultado_cnn=resultado_cnn,
            texto=body.texto,
            # El rol de presentación viene del body (elección por pantalla en la
            # app: agricultor=simple, aprendiz=técnica). El JWT sigue
            # autenticando y gateando roles admin en otros endpoints.
            rol=body.rol,
            cultivos=cultivos,
        )
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except OllamaNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    except TimeoutInferenciaError as exc:
        raise HTTPException(status_code=504, detail=exc.mensaje)

    return ConsultarResponse(
        modo=resultado.modo,
        diagnostico=resultado.diagnostico,
        sintomas=resultado.sintomas,
        avisos=resultado.avisos,
        n_documentos=resultado.n_documentos,
        respuesta=resultado.respuesta,
    )
