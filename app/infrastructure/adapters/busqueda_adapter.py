import logging

from app.core.exceptions import ModeloNoDisponibleError
from app.core.ports import BusquedaSemanticaPort

logger = logging.getLogger(__name__)


class BertAdapter(BusquedaSemanticaPort):
    """Adaptador que envuelve modulos.busqueda_semantica (Sentence-BERT)."""

    def __init__(self) -> None:
        self._disponible = False
        self._busqueda = None
        try:
            from modulos import busqueda_semantica  # type: ignore[import-untyped]

            self._busqueda = busqueda_semantica
            self._disponible = True
            logger.info("Módulo búsqueda semántica cargado correctamente")
        except ImportError:
            logger.warning(
                "modulos.busqueda_semantica no encontrado — BERT no disponible"
            )

    def warmup(self) -> None:
        if not self._busqueda:
            return
        try:
            self._busqueda._obtener_modelo()
            logger.info("Warmup BERT completado")
        except Exception as exc:
            logger.warning("Warmup BERT falló: %s", exc)

    def esta_disponible(self) -> bool:
        return self._disponible

    def generar_embeddings(
        self, textos: list[str]
    ) -> tuple[list[list[float]], int]:
        if not self._disponible:
            raise ModeloNoDisponibleError("Modelo BERT (embeddings)")

        modelo = self._busqueda._obtener_modelo()  # type: ignore[union-attr]
        embeddings = modelo.encode(textos)
        return embeddings.tolist(), int(embeddings.shape[1])

    def buscar_hibrido(
        self,
        consulta: str,
        cultivos: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        if not self._disponible:
            raise ModeloNoDisponibleError("Búsqueda semántica BERT")

        return self._busqueda.buscar_hibrido(  # type: ignore[union-attr]
            consulta, cultivos=cultivos, top_k=top_k
        )
