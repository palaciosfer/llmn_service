from app.core.entities import ResultadoClasificacion, ResultadoDiagnostico
from app.core.exceptions import ModeloNoDisponibleError
from app.core.ports import (
    BusquedaSemanticaPort,
    ClasificadorPort,
    GeneradorPort,
    OrquestadorPort,
)


class DiagnosticoService:
    """Caso de uso principal: diagnóstico completo (CNN + RAG + LLM)."""

    def __init__(self, orquestador: OrquestadorPort):
        self._orquestador = orquestador

    def diagnosticar(
        self,
        imagen_bytes: bytes,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico:
        if not self._orquestador.esta_disponible():
            raise ModeloNoDisponibleError("Orquestador de diagnóstico")
        return self._orquestador.consultar(imagen_bytes, texto, rol, cultivos)

    def consultar_con_diagnostico(
        self,
        resultado_cnn: dict,
        texto: str,
        rol: str,
        cultivos: list[str] | None = None,
    ) -> ResultadoDiagnostico:
        """Flujo principal desde móvil: CNN ya ejecutada en el dispositivo."""
        if not self._orquestador.esta_disponible():
            raise ModeloNoDisponibleError("Orquestador de diagnóstico")
        return self._orquestador.consultar_con_diagnostico(
            resultado_cnn, texto, rol, cultivos
        )


class ClasificadorService:
    """Caso de uso: clasificación de imagen con CNN."""

    def __init__(self, clasificador: ClasificadorPort):
        self._clasificador = clasificador

    def clasificar(self, imagen_bytes: bytes) -> ResultadoClasificacion:
        if not self._clasificador.esta_disponible():
            raise ModeloNoDisponibleError("Clasificador CNN")
        return self._clasificador.predecir(imagen_bytes)


class EmbeddingsService:
    """Caso de uso: generación de embeddings con BERT."""

    def __init__(self, busqueda: BusquedaSemanticaPort):
        self._busqueda = busqueda

    def generar_embeddings(
        self, textos: list[str]
    ) -> tuple[list[list[float]], int]:
        if not self._busqueda.esta_disponible():
            raise ModeloNoDisponibleError("Modelo de embeddings BERT")
        return self._busqueda.generar_embeddings(textos)


class GeneradorService:
    """Caso de uso: generación directa de texto via LLM."""

    def __init__(self, generador: GeneradorPort):
        self._generador = generador

    def generar(self, prompt: str, temperatura: float = 0.2) -> str:
        if not self._generador.esta_disponible():
            raise ModeloNoDisponibleError("Generador LLM (Ollama)")
        return self._generador.generar(prompt, temperatura)
