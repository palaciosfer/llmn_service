from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.application.services import OfflineService
from app.core.entities import UsuarioAutenticado
from app.core.exceptions import ModeloNoDisponibleError
from app.infrastructure.auth import obtener_usuario_actual
from app.infrastructure.dependencies import get_offline_service
from app.infrastructure.schemas import (
    CatalogResponse,
    DocumentDownloadResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/api/v1/offline", tags=["Offline"])


@router.get(
    "/catalog",
    response_model=CatalogResponse,
    summary="Catálogo de documentos descargables para RAG on-device",
    responses={503: {"model": ErrorResponse}},
)
async def offline_catalog(
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> CatalogResponse:
    service: OfflineService = get_offline_service()
    try:
        data = await run_in_threadpool(service.catalogo)
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    return CatalogResponse(**data)


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDownloadResponse,
    summary="Descargar un documento con sus chunks y embeddings (384-d)",
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def offline_document(
    doc_id: str,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> DocumentDownloadResponse:
    service: OfflineService = get_offline_service()
    try:
        doc = await run_in_threadpool(service.documento, doc_id)
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=exc.mensaje)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return DocumentDownloadResponse(**doc)
