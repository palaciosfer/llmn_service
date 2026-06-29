"""
busqueda_semantica.py
Búsqueda semántica con Sentence-BERT y búsqueda híbrida (TF-IDF + BERT).
Modelo multilingüe: paraphrase-multilingual-MiniLM-L12-v2 (funciona en español).
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from modulos.almacen_documentos import (
    _conectar,
    _cargar_indice,
    _RUTA_BD,
    _RUTA_TFIDF,
)

# --- Rutas por defecto ---
_DIR_BASE = Path(__file__).resolve().parent.parent
_RUTA_EMBEDDINGS = _DIR_BASE / "datos" / "embeddings_bert.pkl"

# Modelo multilingüe ligero, funciona offline una vez descargado
_NOMBRE_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"

# Pesos de la combinación híbrida (deben sumar 1)
_PESO_TFIDF = 0.4
_PESO_BERT = 0.6


# ─────────────────────────────────────────────
# Modelo BERT (singleton para no recargarlo)
# ─────────────────────────────────────────────

_modelo: Optional[SentenceTransformer] = None


def _obtener_modelo() -> SentenceTransformer:
    """Carga el modelo Sentence-BERT una sola vez y lo reutiliza."""
    global _modelo
    if _modelo is None:
        print(f"[bert] Cargando modelo '{_NOMBRE_MODELO}'...")
        _modelo = SentenceTransformer(_NOMBRE_MODELO)
        print("[bert] Modelo listo.")
    return _modelo


# ─────────────────────────────────────────────
# Embeddings de documentos
# ─────────────────────────────────────────────

def construir_embeddings(
    ruta_bd: Path = _RUTA_BD,
    ruta_embeddings: Path = _RUTA_EMBEDDINGS,
) -> None:
    """
    Calcula los embeddings BERT de todos los documentos en SQLite
    y los guarda en disco. Solo necesita ejecutarse cuando cambian los documentos.
    """
    con = _conectar(ruta_bd)
    filas = con.execute("SELECT id, texto FROM documentos").fetchall()
    con.close()

    if not filas:
        print("[bert] No hay documentos en la BD.")
        return

    ids = [f["id"] for f in filas]
    textos = [f["texto"] for f in filas]

    modelo = _obtener_modelo()
    print(f"[bert] Calculando embeddings de {len(textos)} documentos...")
    embeddings = modelo.encode(textos, show_progress_bar=True, convert_to_numpy=True)

    ruta_embeddings.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_embeddings, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings}, f)

    print(f"[bert] Embeddings guardados en {ruta_embeddings}.")


def _cargar_embeddings(ruta_embeddings: Path = _RUTA_EMBEDDINGS) -> Optional[dict]:
    """Carga los embeddings desde disco. Devuelve None si no existen."""
    if not ruta_embeddings.exists():
        return None
    with open(ruta_embeddings, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Búsqueda semántica pura (BERT)
# ─────────────────────────────────────────────

def buscar_semantico(
    consulta: str,
    cultivos: Optional[list] = None,
    top_k: int = 10,
    ruta_bd: Path = _RUTA_BD,
    ruta_embeddings: Path = _RUTA_EMBEDDINGS,
) -> list[dict]:
    """
    Busca documentos por similitud semántica (coseno sobre embeddings BERT).

    Args:
        consulta:        Texto de la consulta.
        cultivos:        Filtro opcional por lista de cultivos.
        top_k:           Número máximo de resultados.
        ruta_bd:         Ruta a la base de datos SQLite.
        ruta_embeddings: Ruta al archivo de embeddings serializado.

    Returns:
        Lista de dicts con claves: id, cultivo, enfermedad, fuente, texto, score_bert.
    """
    datos = _cargar_embeddings(ruta_embeddings)
    if datos is None:
        print("[bert] Embeddings no encontrados. Ejecuta construir_embeddings() primero.")
        return []

    ids_indice: list = datos["ids"]
    matriz_emb: np.ndarray = datos["embeddings"]

    modelo = _obtener_modelo()
    vec_consulta = modelo.encode([consulta], convert_to_numpy=True)
    similitudes = cosine_similarity(vec_consulta, matriz_emb).flatten()

    orden = np.argsort(similitudes)[::-1]

    con = _conectar(ruta_bd)
    resultados = []

    for idx in orden:
        if len(resultados) >= top_k:
            break
        score = float(similitudes[idx])
        if score <= 0.0:
            break

        doc_id = ids_indice[idx]
        fila = con.execute(
            "SELECT id, cultivo, enfermedad, fuente, texto FROM documentos WHERE id = ?",
            (doc_id,),
        ).fetchone()

        if fila is None:
            continue

        if cultivos:
            cultivos_lower = [c.lower() for c in cultivos]
            if fila["cultivo"].lower() not in cultivos_lower:
                continue

        resultados.append({
            "id": fila["id"],
            "cultivo": fila["cultivo"],
            "enfermedad": fila["enfermedad"],
            "fuente": fila["fuente"],
            "texto": fila["texto"],
            "score_bert": round(score, 4),
        })

    con.close()
    return resultados


# ─────────────────────────────────────────────
# Búsqueda híbrida (TF-IDF + BERT)
# ─────────────────────────────────────────────

def buscar_hibrido(
    consulta: str,
    cultivos: Optional[list] = None,
    top_k: int = 10,
    peso_tfidf: float = _PESO_TFIDF,
    peso_bert: float = _PESO_BERT,
    ruta_bd: Path = _RUTA_BD,
    ruta_tfidf: Path = _RUTA_TFIDF,
    ruta_embeddings: Path = _RUTA_EMBEDDINGS,
) -> list[dict]:
    """
    Combina TF-IDF (léxico) y BERT (semántico) en una puntuación híbrida.
    Respeta el filtro por cultivos.

    Args:
        consulta:    Texto de la consulta.
        cultivos:    Filtro opcional por lista de cultivos.
        top_k:       Número máximo de resultados.
        peso_tfidf:  Peso del componente TF-IDF (0-1).
        peso_bert:   Peso del componente BERT (0-1).

    Returns:
        Lista de dicts ordenada por score_hibrido descendente.
        Claves: id, cultivo, enfermedad, fuente, texto,
                score_tfidf, score_bert, score_hibrido.
    """
    # --- Puntuaciones TF-IDF ---
    indice_tfidf = _cargar_indice(ruta_tfidf)
    scores_tfidf: dict[int, float] = {}
    if indice_tfidf:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        vec = indice_tfidf["vectorizador"].transform([consulta])
        sims = cos_sim(vec, indice_tfidf["matriz"]).flatten()
        for i, doc_id in enumerate(indice_tfidf["ids"]):
            scores_tfidf[doc_id] = float(sims[i])
    else:
        print("[hibrido] Índice TF-IDF no disponible; se usa solo BERT.")

    # --- Puntuaciones BERT ---
    datos_bert = _cargar_embeddings(ruta_embeddings)
    scores_bert: dict[int, float] = {}
    if datos_bert:
        modelo = _obtener_modelo()
        vec_consulta = modelo.encode([consulta], convert_to_numpy=True)
        sims_bert = cosine_similarity(vec_consulta, datos_bert["embeddings"]).flatten()
        for i, doc_id in enumerate(datos_bert["ids"]):
            scores_bert[doc_id] = float(sims_bert[i])
    else:
        print("[hibrido] Embeddings BERT no disponibles; se usa solo TF-IDF.")

    # --- Unir todos los doc_ids conocidos ---
    todos_ids = set(scores_tfidf.keys()) | set(scores_bert.keys())
    if not todos_ids:
        return []

    # --- Calcular score híbrido ---
    combinados = []
    for doc_id in todos_ids:
        s_tfidf = scores_tfidf.get(doc_id, 0.0)
        s_bert = scores_bert.get(doc_id, 0.0)
        s_hibrido = peso_tfidf * s_tfidf + peso_bert * s_bert
        combinados.append((doc_id, s_tfidf, s_bert, s_hibrido))

    combinados.sort(key=lambda x: x[3], reverse=True)

    # --- Recuperar documentos de SQLite y aplicar filtro ---
    con = _conectar(ruta_bd)
    resultados = []

    for doc_id, s_tfidf, s_bert, s_hibrido in combinados:
        if len(resultados) >= top_k:
            break
        if s_hibrido <= 0.0:
            break

        fila = con.execute(
            "SELECT id, cultivo, enfermedad, fuente, texto FROM documentos WHERE id = ?",
            (doc_id,),
        ).fetchone()

        if fila is None:
            continue

        if cultivos:
            cultivos_lower = [c.lower() for c in cultivos]
            if fila["cultivo"].lower() not in cultivos_lower:
                continue

        resultados.append({
            "id": fila["id"],
            "cultivo": fila["cultivo"],
            "enfermedad": fila["enfermedad"],
            "fuente": fila["fuente"],
            "texto": fila["texto"],
            "score_tfidf": round(s_tfidf, 4),
            "score_bert": round(s_bert, 4),
            "score_hibrido": round(s_hibrido, 4),
        })

    con.close()
    return resultados
