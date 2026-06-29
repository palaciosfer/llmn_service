from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Rol(str, Enum):
    AGRICULTOR = "agricultor"
    APRENDIZ = "aprendiz"


@dataclass(frozen=True)
class UsuarioAutenticado:
    id: str
    rol: Rol
    email: Optional[str] = None


@dataclass
class ResultadoClasificacion:
    cultivo: str
    enfermedad: str
    confianza: float
    clase_cnn: str
    confianza_baja: bool


@dataclass
class ResultadoDiagnostico:
    modo: str
    diagnostico: dict[str, Any]
    sintomas: list[str]
    avisos: list[str]
    n_documentos: int
    documentos: list[dict[str, Any]]
    respuesta: dict[str, Any]
