from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from app.application.services import CampaniasService
from app.core.entities import UsuarioAutenticado
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_campanias_service
from app.infrastructure.schemas import AlertaResponse, MapaCampaniasResponse

router = APIRouter(prefix="/api/v1", tags=["Clustering"])


@router.get(
    "/clustering/mapa-campanias",
    response_model=MapaCampaniasResponse,
    summary="Mapa epidemiológico REAL (campañas SENASICA por estado)",
)
async def mapa_campanias(
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> MapaCampaniasResponse:
    service: CampaniasService = get_campanias_service()
    data = await run_in_threadpool(service.mapa)
    return MapaCampaniasResponse(**data)


@router.get(
    "/alertas",
    response_model=AlertaResponse,
    summary="Alerta epidemiológica real (campaña dominante por estado)",
)
async def alertas(
    estado: str | None = Query(
        None, description="Entidad federativa; si se omite, alerta nacional"
    ),
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> AlertaResponse:
    service: CampaniasService = get_campanias_service()
    data = await run_in_threadpool(service.alerta, estado)
    return AlertaResponse(**data)
