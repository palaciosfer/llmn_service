import logging
from io import BytesIO

from PIL import Image

from app.core.entities import ResultadoClasificacion
from app.core.exceptions import ImagenInvalidaError, ModeloNoDisponibleError
from app.core.ports import ClasificadorPort

logger = logging.getLogger(__name__)


class CnnAdapter(ClasificadorPort):
    """Adaptador que envuelve modulos.clasificador (EfficientNet-B4)."""

    def __init__(self) -> None:
        self._disponible = False
        self._clasificador = None
        try:
            from modulos import clasificador  # type: ignore[import-untyped]

            self._clasificador = clasificador
            self._disponible = True
            logger.info("Módulo clasificador CNN cargado correctamente")
        except ImportError:
            logger.warning(
                "modulos.clasificador no encontrado — CNN no disponible"
            )

    def warmup(self) -> None:
        if not self._clasificador:
            return
        try:
            dummy = Image.new("RGB", (380, 380))
            self._clasificador.predecir(dummy)
            logger.info("Warmup CNN completado")
        except Exception as exc:
            logger.warning("Warmup CNN falló: %s", exc)

    def esta_disponible(self) -> bool:
        return self._disponible

    def predecir(self, imagen_bytes: bytes) -> ResultadoClasificacion:
        if not self._disponible:
            raise ModeloNoDisponibleError("CNN (EfficientNet-B4)")

        try:
            imagen = Image.open(BytesIO(imagen_bytes)).convert("RGB")
        except Exception as exc:
            raise ImagenInvalidaError(
                f"No se pudo decodificar la imagen: {exc}"
            )

        resultado = self._clasificador.predecir(imagen)  # type: ignore[union-attr]

        return ResultadoClasificacion(
            cultivo=resultado.get("cultivo", ""),
            enfermedad=resultado.get("enfermedad", ""),
            confianza=resultado.get("confianza", 0.0),
            clase_cnn=resultado.get("clase_cnn", ""),
            confianza_baja=resultado.get("confianza_baja", True),
        )
