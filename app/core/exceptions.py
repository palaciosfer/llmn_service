class DomainException(Exception):
    """Base para todas las excepciones del dominio."""

    def __init__(self, mensaje: str, codigo: str = "ERROR_DOMINIO"):
        self.mensaje = mensaje
        self.codigo = codigo
        super().__init__(mensaje)


class ModeloNoDisponibleError(DomainException):
    def __init__(self, modelo: str):
        super().__init__(
            mensaje=f"Modelo no disponible: {modelo}",
            codigo="MODELO_NO_DISPONIBLE",
        )


class ImagenInvalidaError(DomainException):
    def __init__(self, detalle: str):
        super().__init__(
            mensaje=f"Imagen inválida: {detalle}",
            codigo="IMAGEN_INVALIDA",
        )


class OllamaNoDisponibleError(DomainException):
    def __init__(self, detalle: str = "Ollama no responde"):
        super().__init__(mensaje=detalle, codigo="OLLAMA_NO_DISPONIBLE")


class TimeoutInferenciaError(DomainException):
    def __init__(self, detalle: str = "Timeout en la inferencia"):
        super().__init__(mensaje=detalle, codigo="TIMEOUT_INFERENCIA")


class AutenticacionError(DomainException):
    def __init__(self, detalle: str = "No autenticado"):
        super().__init__(mensaje=detalle, codigo="AUTENTICACION_ERROR")


class AutorizacionError(DomainException):
    def __init__(self, detalle: str = "No autorizado"):
        super().__init__(mensaje=detalle, codigo="AUTORIZACION_ERROR")
