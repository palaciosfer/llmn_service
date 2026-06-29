import logging

from app.application.services import (
    ClasificadorService,
    DiagnosticoService,
    EmbeddingsService,
    GeneradorService,
)
from app.infrastructure.adapters.busqueda_adapter import BertAdapter
from app.infrastructure.adapters.clasificador_adapter import CnnAdapter
from app.infrastructure.adapters.generador_adapter import OllamaAdapter
from app.infrastructure.adapters.orquestador_adapter import OrquestadorAdapter
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class Container:
    """Contenedor de inyección de dependencias (composition root)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        logger.info("Inicializando adaptadores...")
        self.cnn_adapter = CnnAdapter()
        self.bert_adapter = BertAdapter()
        self.ollama_adapter = OllamaAdapter(settings)
        self.orquestador_adapter = OrquestadorAdapter()

        self.diagnostico_service = DiagnosticoService(self.orquestador_adapter)
        self.clasificador_service = ClasificadorService(self.cnn_adapter)
        self.embeddings_service = EmbeddingsService(self.bert_adapter)
        self.generador_service = GeneradorService(self.ollama_adapter)

        logger.info("Contenedor de dependencias listo")

    def warmup(self) -> None:
        logger.info("Ejecutando warmup de modelos...")
        self.cnn_adapter.warmup()
        self.bert_adapter.warmup()
        logger.info("Warmup completado")


_container: Container | None = None


def init_container(settings: Settings) -> Container:
    global _container
    _container = Container(settings)
    return _container


def get_container() -> Container:
    if _container is None:
        raise RuntimeError(
            "Container no inicializado. Llamar init_container() en el startup."
        )
    return _container


def get_diagnostico_service() -> DiagnosticoService:
    return get_container().diagnostico_service


def get_clasificador_service() -> ClasificadorService:
    return get_container().clasificador_service


def get_embeddings_service() -> EmbeddingsService:
    return get_container().embeddings_service


def get_generador_service() -> GeneradorService:
    return get_container().generador_service
