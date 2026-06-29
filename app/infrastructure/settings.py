from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    QWEN_MODELO: str = "qwen3.5:0.8b"
    OLLAMA_TIMEOUT: int = 120

    # ── Servidor ────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    LOG_LEVEL: str = "INFO"
    DEV_MODE: bool = False

    # ── Seguridad / JWT ─────────────────────────────────────
    JWT_SECRET: str = "cambiar-en-produccion-clave-super-secreta"
    JWT_ALGORITHM: str = "HS256"
    CORS_ORIGINS: str = "*"

    # ── Límites ─────────────────────────────────────────────
    MAX_IMAGEN_MB: int = 8

    # ── Device CNN/BERT ─────────────────────────────────────
    DEVICE: str = "cpu"

    # ── Rutas a artefactos del modelo ───────────────────────
    MODELO_CNN_PATH: str = "modelos/best.pth"
    ALMACEN_DB_PATH: str = "datos/almacen.db"
    TFIDF_PATH: str = "datos/tfidf.pkl"
    EMBEDDINGS_PATH: str = "datos/embeddings_bert.pkl"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
