"""
almacen_documentos.py
Gestiona el almacén de documentos fitosanitarios: SQLite + TF-IDF + caché Top-K.
"""

import os
import json
import sqlite3
import pickle
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Rutas por defecto ---
_DIR_BASE = Path(__file__).resolve().parent.parent
_RUTA_BD = _DIR_BASE / "datos" / "almacen.db"
_RUTA_TFIDF = _DIR_BASE / "datos" / "tfidf.pkl"
_DIR_DOCS = _DIR_BASE / "documentos"


# ─────────────────────────────────────────────
# Conexión y esquema
# ─────────────────────────────────────────────

def _conectar(ruta_bd: Path = _RUTA_BD) -> sqlite3.Connection:
    """Abre (o crea) la base de datos y garantiza que las tablas existen."""
    ruta_bd.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta_bd))
    con.row_factory = sqlite3.Row
    _crear_tablas(con)
    return con


def _crear_tablas(con: sqlite3.Connection) -> None:
    """Crea las tablas si no existen."""
    con.executescript("""
        CREATE TABLE IF NOT EXISTS documentos (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            cultivo   TEXT    NOT NULL,
            enfermedad TEXT   NOT NULL,
            fuente    TEXT    DEFAULT '',
            texto     TEXT    NOT NULL,
            fragmento INTEGER DEFAULT 0,   -- nº de trozo dentro de un documento largo
            UNIQUE(cultivo, enfermedad, fuente, fragmento)
        );

        CREATE TABLE IF NOT EXISTS cache_topk (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            enfermedad  TEXT NOT NULL,
            doc_ids     TEXT NOT NULL,   -- JSON: lista de IDs de documentos
            actualizado TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()


# ─────────────────────────────────────────────
# Carga de documentos desde archivos
# ─────────────────────────────────────────────

def _leer_metadatos_txt(texto: str) -> dict:
    """
    Extrae cultivo, enfermedad y fuente de las primeras líneas del .txt.
    Espera líneas con formato 'CAMPO: valor'.
    """
    meta = {"cultivo": "", "enfermedad": "", "fuente": ""}
    mapa = {"CULTIVO": "cultivo", "ENFERMEDAD": "enfermedad", "FUENTE": "fuente"}
    for linea in texto.splitlines()[:6]:
        partes = linea.split(":", 1)
        if len(partes) == 2:
            clave = partes[0].strip().upper()
            if clave in mapa:
                meta[mapa[clave]] = partes[1].strip()
    return meta


def cargar_desde_directorio(
    directorio: Path = _DIR_DOCS,
    ruta_bd: Path = _RUTA_BD,
) -> int:
    """
    Lee todos los .txt y .pdf del directorio y los inserta en SQLite.
    Devuelve el número de documentos nuevos insertados.
    """
    con = _conectar(ruta_bd)
    insertados = 0

    for archivo in sorted(directorio.glob("**/*")):
        if archivo.suffix.lower() == ".txt":
            texto = archivo.read_text(encoding="utf-8", errors="ignore")
        elif archivo.suffix.lower() == ".pdf":
            texto = _leer_pdf(archivo)
        else:
            continue

        if not texto.strip():
            continue

        meta = _leer_metadatos_txt(texto)
        if not meta["cultivo"] or not meta["enfermedad"]:
            # Sin metadatos mínimos, usar el nombre del archivo como fallback
            meta["cultivo"] = archivo.stem.split("_")[0]
            meta["enfermedad"] = archivo.stem

        try:
            con.execute(
                "INSERT OR IGNORE INTO documentos (cultivo, enfermedad, fuente, texto) "
                "VALUES (?, ?, ?, ?)",
                (meta["cultivo"], meta["enfermedad"], meta["fuente"], texto),
            )
            if con.execute("SELECT changes()").fetchone()[0] > 0:
                insertados += 1
        except sqlite3.Error as e:
            print(f"[almacen] Error al insertar {archivo.name}: {e}")

    con.commit()
    con.close()
    return insertados


def agregar_corpus(registros: list[dict], ruta_bd: Path = _RUTA_BD) -> int:
    """
    Inserta un corpus ya troceado en el almacén. Cada registro es un fragmento.

    Args:
        registros: lista de dicts con claves 'cultivo', 'enfermedad', 'fuente', 'texto'.
                   Los fragmentos de un mismo (cultivo, enfermedad, fuente) se numeran
                   automáticamente, de modo que volver a ejecutar es idempotente.

    Returns:
        Número de fragmentos nuevos insertados.
    """
    con = _conectar(ruta_bd)
    insertados = 0
    contador: dict = {}

    for r in registros:
        cultivo = (r.get("cultivo") or "").strip()
        enfermedad = (r.get("enfermedad") or "").strip()
        fuente = (r.get("fuente") or "").strip()
        texto = (r.get("texto") or "").strip()
        if not cultivo or not texto:
            continue

        clave = (cultivo, enfermedad, fuente)
        frag = contador.get(clave, 0)
        contador[clave] = frag + 1

        try:
            con.execute(
                "INSERT OR IGNORE INTO documentos (cultivo, enfermedad, fuente, texto, fragmento) "
                "VALUES (?, ?, ?, ?, ?)",
                (cultivo, enfermedad, fuente, texto, frag),
            )
            if con.execute("SELECT changes()").fetchone()[0] > 0:
                insertados += 1
        except sqlite3.Error as e:
            print(f"[almacen] Error al insertar fragmento {clave} #{frag}: {e}")

    con.commit()
    con.close()
    return insertados


def _leer_pdf(ruta: Path) -> str:
    """Extrae texto de un PDF con pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(ruta))
        return "\n".join(
            pagina.extract_text() or "" for pagina in reader.pages
        )
    except ImportError:
        print("[almacen] pypdf no instalado; omitiendo PDF:", ruta.name)
        return ""
    except Exception as e:
        print(f"[almacen] Error leyendo PDF {ruta.name}: {e}")
        return ""


# ─────────────────────────────────────────────
# Índice TF-IDF
# ─────────────────────────────────────────────

def construir_indice(ruta_bd: Path = _RUTA_BD, ruta_tfidf: Path = _RUTA_TFIDF) -> None:
    """
    Construye el vectorizador TF-IDF con todos los documentos de la BD
    y lo guarda en disco (pickle).
    """
    con = _conectar(ruta_bd)
    filas = con.execute("SELECT id, texto FROM documentos").fetchall()
    con.close()

    if not filas:
        print("[almacen] No hay documentos para indexar.")
        return

    ids = [f["id"] for f in filas]
    textos = [f["texto"] for f in filas]

    vectorizador = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    matriz = vectorizador.fit_transform(textos)

    ruta_tfidf.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_tfidf, "wb") as f:
        pickle.dump({"ids": ids, "vectorizador": vectorizador, "matriz": matriz}, f)

    print(f"[almacen] Índice TF-IDF construido con {len(ids)} documentos.")


def _cargar_indice(ruta_tfidf: Path = _RUTA_TFIDF) -> Optional[dict]:
    """Carga el índice TF-IDF desde disco. Devuelve None si no existe."""
    if not ruta_tfidf.exists():
        return None
    with open(ruta_tfidf, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Búsqueda
# ─────────────────────────────────────────────

def buscar(
    consulta: str,
    cultivos: Optional[list] = None,
    top_k: int = 10,
    ruta_bd: Path = _RUTA_BD,
    ruta_tfidf: Path = _RUTA_TFIDF,
) -> list[dict]:
    """
    Busca los documentos más relevantes para la consulta usando TF-IDF.

    Args:
        consulta:  Texto de la consulta (síntomas, enfermedad, etc.).
        cultivos:  Lista de cultivos para filtrar. None = sin filtro.
        top_k:     Número máximo de documentos a devolver.
        ruta_bd:   Ruta a la base de datos SQLite.
        ruta_tfidf: Ruta al índice TF-IDF serializado.

    Returns:
        Lista de dicts con claves: id, cultivo, enfermedad, fuente, texto, score.
    """
    indice = _cargar_indice(ruta_tfidf)
    if indice is None:
        print("[almacen] Índice TF-IDF no encontrado. Ejecuta construir_indice() primero.")
        return []

    vectorizador: TfidfVectorizer = indice["vectorizador"]
    matriz = indice["matriz"]
    ids_indice: list = indice["ids"]

    # Vectorizar la consulta
    vec_consulta = vectorizador.transform([consulta])
    similitudes = cosine_similarity(vec_consulta, matriz).flatten()

    # Ordenar por similitud descendente
    orden = np.argsort(similitudes)[::-1]

    # Recuperar documentos de SQLite
    con = _conectar(ruta_bd)
    resultados = []

    for idx in orden:
        if len(resultados) >= top_k:
            break
        score = float(similitudes[idx])
        if score == 0.0:
            break  # los que siguen tampoco son relevantes

        doc_id = ids_indice[idx]
        fila = con.execute(
            "SELECT id, cultivo, enfermedad, fuente, texto FROM documentos WHERE id = ?",
            (doc_id,),
        ).fetchone()

        if fila is None:
            continue

        # Filtrar por cultivos si se especificó
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
            "score": round(score, 4),
        })

    con.close()
    return resultados


# ─────────────────────────────────────────────
# Caché Top-K
# ─────────────────────────────────────────────

def guardar_topk(
    enfermedad: str,
    documentos: list[dict],
    ruta_bd: Path = _RUTA_BD,
) -> None:
    """
    Guarda (o actualiza) el Top-K de documentos para una enfermedad en el caché.

    Args:
        enfermedad:  Nombre de la enfermedad (clave del caché).
        documentos:  Lista de dicts devuelta por buscar() o buscar_hibrido().
    """
    doc_ids = json.dumps([d["id"] for d in documentos])
    con = _conectar(ruta_bd)
    # Upsert: actualizar si existe, insertar si no
    existente = con.execute(
        "SELECT id FROM cache_topk WHERE enfermedad = ?", (enfermedad,)
    ).fetchone()

    if existente:
        con.execute(
            "UPDATE cache_topk SET doc_ids = ?, actualizado = datetime('now') WHERE enfermedad = ?",
            (doc_ids, enfermedad),
        )
    else:
        con.execute(
            "INSERT INTO cache_topk (enfermedad, doc_ids) VALUES (?, ?)",
            (enfermedad, doc_ids),
        )
    con.commit()
    con.close()


def recuperar_topk(
    enfermedad: str,
    ruta_bd: Path = _RUTA_BD,
) -> list[dict]:
    """
    Recupera los documentos del caché Top-K para una enfermedad.
    Devuelve lista vacía si no hay caché para esa enfermedad.
    """
    con = _conectar(ruta_bd)
    fila = con.execute(
        "SELECT doc_ids FROM cache_topk WHERE enfermedad = ?", (enfermedad,)
    ).fetchone()

    if fila is None:
        con.close()
        return []

    doc_ids = json.loads(fila["doc_ids"])
    if not doc_ids:
        con.close()
        return []

    placeholders = ",".join("?" * len(doc_ids))
    filas = con.execute(
        f"SELECT id, cultivo, enfermedad, fuente, texto FROM documentos WHERE id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    con.close()

    return [
        {
            "id": f["id"],
            "cultivo": f["cultivo"],
            "enfermedad": f["enfermedad"],
            "fuente": f["fuente"],
            "texto": f["texto"],
            "score": None,  # el score ya no aplica al recuperar del caché
        }
        for f in filas
    ]


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def listar_documentos(ruta_bd: Path = _RUTA_BD) -> list[dict]:
    """Devuelve todos los documentos (sin el texto completo) para inspección."""
    con = _conectar(ruta_bd)
    filas = con.execute(
        "SELECT id, cultivo, enfermedad, fuente FROM documentos ORDER BY cultivo, enfermedad"
    ).fetchall()
    con.close()
    return [dict(f) for f in filas]
