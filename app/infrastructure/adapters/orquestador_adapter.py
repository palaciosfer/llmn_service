import logging
from io import BytesIO

from PIL import Image

from app.core.entities import ResultadoDiagnostico
from app.core.exceptions import (
    ImagenInvalidaError,
    ModeloNoDisponibleError,
    OllamaNoDisponibleError,
    TimeoutInferenciaError,
)
from app.core.ports import OrquestadorPort

logger = logging.getLogger(__name__)


class OrquestadorAdapter(OrquestadorPort):
    """Adaptador que envuelve modulos.asistente.consultar() — pipeline completo."""

    def __init__(self) -> None:
        self._disponible = False
        self._asistente = None
        try:
            from modulos import asistente  # type: ignore[import-untyped]

            self._asistente = asistente
            self._disponible = True
            logger.info("Módulo asistente (orquestador) cargado correctamente")
        except ImportError:
            logger.warning(
                "modulos.asistente no encontrado — orquestador no disponible"
            )

    def esta_disponible(self) -> bool:
        return self._disponible

    def consultar(
        self,
        imagen_bytes: bytes,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico:
        if not self._disponible:
            raise ModeloNoDisponibleError("Orquestador de diagnóstico")

        try:
            imagen = Image.open(BytesIO(imagen_bytes)).convert("RGB")
        except Exception as exc:
            raise ImagenInvalidaError(
                f"No se pudo decodificar la imagen: {exc}"
            )

        try:
            resultado = self._asistente.consultar(  # type: ignore[union-attr]
                imagen=imagen,
                texto=texto,
                rol=rol,
                cultivos=cultivos,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("ollama", "conexión", "connection")):
                raise OllamaNoDisponibleError(str(exc))
            raise
        except TimeoutError as exc:
            raise TimeoutInferenciaError(str(exc))

        return self._mapear_resultado(resultado)

    def consultar_con_diagnostico(
        self,
        resultado_cnn: dict,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico:
        if not self._disponible:
            raise ModeloNoDisponibleError("Orquestador de diagnóstico")

        try:
            resultado = self._asistente.consultar(  # type: ignore[union-attr]
                imagen=None,
                texto=texto,
                rol=rol,
                cultivos=cultivos if cultivos else [],
                resultado_cnn=resultado_cnn,
                forzar_offline=False,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("ollama", "conexión", "connection")):
                raise OllamaNoDisponibleError(str(exc))
            raise
        except TimeoutError as exc:
            raise TimeoutInferenciaError(str(exc))

        return self._mapear_resultado(resultado)

    @staticmethod
    def _mapear_resultado(resultado: dict) -> ResultadoDiagnostico:
        return ResultadoDiagnostico(
            modo=resultado.get("modo", "online"),
            diagnostico=resultado.get("diagnostico", {}),
            sintomas=resultado.get("sintomas", []),
            avisos=resultado.get("avisos", []),
            n_documentos=resultado.get("n_documentos", 0),
            documentos=resultado.get("documentos", []),
            respuesta=resultado.get("respuesta", {}),
        )
