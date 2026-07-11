from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Diagnóstico ─────────────────────────────────────────────

class DiagnosticoResponse(BaseModel):
    modo: str
    diagnostico: dict[str, Any]
    sintomas: list[str]
    avisos: list[str]
    n_documentos: int
    respuesta: dict[str, Any]


# ── Consultar (flujo principal desde móvil) ─────────────────

class ResultadoCnnSchema(BaseModel):
    cultivo: str = Field(..., description="Cultivo detectado por la CNN")
    enfermedad: str = Field(..., description="Enfermedad detectada")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Confianza [0-1]")
    clase_cnn: str = Field(..., description="Etiqueta cruda del modelo CNN")
    confianza_baja: bool = Field(..., description="True si confianza < umbral")


class ConsultarRequest(BaseModel):
    resultado_cnn: ResultadoCnnSchema = Field(
        ..., description="Resultado de la CNN ejecutada en el dispositivo móvil"
    )
    texto: str = Field(
        default="",
        max_length=2000,
        description="Síntomas observados por el usuario",
    )
    cultivos: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Cultivos para filtrar la búsqueda",
    )


class ConsultarResponse(BaseModel):
    modo: str
    diagnostico: dict[str, Any]
    sintomas: list[str]
    avisos: list[str]
    n_documentos: int
    respuesta: dict[str, Any]


# ── Clasificación CNN ───────────────────────────────────────

class ClasificacionResponse(BaseModel):
    cultivo: str
    enfermedad: str
    confianza: float
    clase_cnn: str
    confianza_baja: bool


# ── Embeddings ──────────────────────────────────────────────

class EmbeddingsRequest(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=100)


class EmbeddingsResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


# ── Generador LLM ──────────────────────────────────────────

class GenerarRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    temperatura: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerarResponse(BaseModel):
    texto: str


# ── Health / Ready ──────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    cnn_disponible: bool
    bert_disponible: bool
    ollama_disponible: bool
    modulos_cargados: bool


# ── Errores ─────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    codigo: str
    detalle: Optional[str] = None


# ── Offline: catálogo y descarga de documentos (RAG on-device) ─────────────

class CatalogItem(BaseModel):
    id: str
    crop_name: str
    disease_name: str
    title: str
    source: str
    size_bytes: int
    version: str


class CatalogResponse(BaseModel):
    documents: list[CatalogItem]


class ChunkModel(BaseModel):
    id: str
    index: int
    text: str
    embedding: list[float]  # 384-d (MiniLM-L12)


class DocumentDownloadResponse(BaseModel):
    id: str
    content: str
    size_bytes: int
    embedding: list[float]  # embedding global (media de los chunks), 384-d
    chunks: list[ChunkModel]


# ── Mapa epidemiológico REAL (campañas fitosanitarias SENASICA) ────────────

class EstadoResumen(BaseModel):
    estado: str
    campanias: int
    superficie_ha: float
    productores: int
    campania_dominante: str
    cultivo_dominante: str


class MapaCampaniasResponse(BaseModel):
    total_campanias: int
    estados: list[EstadoResumen]


class AlertaResponse(BaseModel):
    hay_alerta: bool
    estado: str
    mensaje: str
    campania_dominante: Optional[str] = None
    plaga_dominante: Optional[str] = None
    cultivo_dominante: Optional[str] = None
    campanias: Optional[int] = None
    superficie_ha: Optional[float] = None


# ── Token dev ───────────────────────────────────────────────

class TokenDevRequest(BaseModel):
    sub: str = Field(..., description="ID del usuario")
    rol: str = Field(
        default="agricultor",
        pattern="^(agricultor|aprendiz)$",
        description="Rol: agricultor o aprendiz",
    )
    email: Optional[str] = None


class TokenDevResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    rol: str
    expira_en_horas: int
