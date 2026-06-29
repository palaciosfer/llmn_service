import logging

import requests

from app.core.exceptions import ModeloNoDisponibleError, OllamaNoDisponibleError
from app.core.ports import GeneradorPort
from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)


class OllamaAdapter(GeneradorPort):
    """Adaptador que envuelve modulos.generador (cliente HTTP a Ollama)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._disponible = False
        self._generador = None
        try:
            from modulos import generador  # type: ignore[import-untyped]

            self._generador = generador
            self._disponible = True
            logger.info("Módulo generador LLM cargado correctamente")
        except ImportError:
            logger.warning(
                "modulos.generador no encontrado — LLM no disponible"
            )

    def _base_url(self) -> str:
        url = self._settings.OLLAMA_URL
        if "/api/generate" in url:
            return url.replace("/api/generate", "")
        return url.rsplit("/api", 1)[0] if "/api" in url else url

    def ollama_responde(self) -> bool:
        try:
            resp = requests.get(
                f"{self._base_url()}/api/tags", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def esta_disponible(self) -> bool:
        return self._disponible

    def generar(self, prompt: str, temperatura: float = 0.2) -> str:
        if not self._disponible:
            raise ModeloNoDisponibleError("Generador LLM (Ollama)")

        try:
            resultado = self._generador._llamar_ollama(prompt)  # type: ignore[union-attr]
            return resultado
        except RuntimeError as exc:
            raise OllamaNoDisponibleError(str(exc))
        except requests.exceptions.Timeout as exc:
            raise OllamaNoDisponibleError(f"Timeout de Ollama: {exc}")
        except requests.exceptions.ConnectionError as exc:
            raise OllamaNoDisponibleError(
                f"No se pudo conectar a Ollama: {exc}"
            )
