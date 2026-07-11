from abc import ABC, abstractmethod

from app.core.entities import ResultadoClasificacion, ResultadoDiagnostico


class ClasificadorPort(ABC):
    """Puerto de salida: clasificación de imagen con CNN."""

    @abstractmethod
    def predecir(self, imagen_bytes: bytes) -> ResultadoClasificacion: ...

    @abstractmethod
    def esta_disponible(self) -> bool: ...


class BusquedaSemanticaPort(ABC):
    """Puerto de salida: búsqueda semántica y generación de embeddings."""

    @abstractmethod
    def buscar_hibrido(
        self,
        consulta: str,
        cultivos: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]: ...

    @abstractmethod
    def generar_embeddings(self, textos: list[str]) -> tuple[list[list[float]], int]:
        """Retorna (embeddings, dimensión)."""
        ...

    @abstractmethod
    def esta_disponible(self) -> bool: ...


class GeneradorPort(ABC):
    """Puerto de salida: generación de texto via LLM (Ollama)."""

    @abstractmethod
    def generar(self, prompt: str, temperatura: float = 0.2) -> str: ...

    @abstractmethod
    def ollama_responde(self) -> bool: ...

    @abstractmethod
    def esta_disponible(self) -> bool: ...


class OrquestadorPort(ABC):
    """Puerto de salida: orquestación completa del diagnóstico (RAG)."""

    @abstractmethod
    def consultar(
        self,
        imagen_bytes: bytes,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico: ...

    @abstractmethod
    def consultar_con_diagnostico(
        self,
        resultado_cnn: dict,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico:
        """Consulta con resultado CNN ya calculado (desde la app móvil)."""
        ...

    @abstractmethod
    def esta_disponible(self) -> bool: ...


class OfflineRepositoryPort(ABC):
    """Puerto de salida: catálogo y descarga de documentos para RAG on-device."""

    @abstractmethod
    def catalogo(self) -> dict:
        """Lista de documentos disponibles para descarga (uno por cultivo+fuente)."""
        ...

    @abstractmethod
    def documento(self, doc_id: str) -> dict | None:
        """Documento con su contenido, chunks y embeddings (384-d), o None si no existe."""
        ...


class CampaniasRepositoryPort(ABC):
    """Puerto de salida: mapa epidemiológico y alertas (campañas fitosanitarias SENASICA)."""

    @abstractmethod
    def mapa(self) -> dict:
        """Agregado por estado: campañas, superficie, campaña/cultivo dominante."""
        ...

    @abstractmethod
    def alerta(self, estado: str | None = None) -> dict:
        """Campaña/plaga dominante para un estado, o a nivel nacional si se omite."""
        ...
