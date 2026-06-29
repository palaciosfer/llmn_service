"""
mis_cultivos.py
Gestiona la lista de cultivos de la parcela del usuario (tabla SQLite `mis_cultivos`).

Esta lista es configuración local de la app (NO es un sistema de cuentas) y se usa
para dos cosas:
  - filtrar las búsquedas (almacen_documentos.buscar / busqueda_semantica.buscar_hibrido),
  - pre-cargar el caché Top-K (caché semilla) solo de los cultivos del usuario.
"""

import sqlite3
from pathlib import Path

# --- Rutas por defecto (misma BD que el almacén) ---
_DIR_BASE = Path(__file__).resolve().parent.parent
_RUTA_BD = _DIR_BASE / "datos" / "almacen.db"


# ─────────────────────────────────────────────
# Conexión y esquema
# ─────────────────────────────────────────────

def _conectar(ruta_bd: Path = _RUTA_BD) -> sqlite3.Connection:
    """Abre (o crea) la base de datos y garantiza que la tabla existe."""
    ruta_bd.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta_bd))
    con.row_factory = sqlite3.Row
    _crear_tabla(con)
    return con


def _crear_tabla(con: sqlite3.Connection) -> None:
    """Crea la tabla mis_cultivos si no existe."""
    con.executescript("""
        CREATE TABLE IF NOT EXISTS mis_cultivos (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            cultivo   TEXT NOT NULL UNIQUE,
            agregado  TEXT DEFAULT (datetime('now'))
        );
    """)
    con.commit()


def _normalizar(cultivo: str) -> str:
    """Normaliza el nombre del cultivo: sin espacios sobrantes y en minúsculas.

    Se guarda en minúsculas para que coincida con el filtrado por cultivo
    (buscar() compara en minúsculas) y para evitar duplicados tipo 'Maíz'/'maíz'.
    """
    return cultivo.strip().lower()


# ─────────────────────────────────────────────
# Operaciones
# ─────────────────────────────────────────────

def agregar(cultivo: str, ruta_bd: Path = _RUTA_BD) -> bool:
    """
    Agrega un cultivo a la parcela.

    Args:
        cultivo: Nombre del cultivo (ej. 'maíz'). Se normaliza a minúsculas.

    Returns:
        True si se agregó; False si estaba vacío o ya existía.
    """
    nombre = _normalizar(cultivo)
    if not nombre:
        print("[mis_cultivos] Nombre de cultivo vacío; no se agregó.")
        return False

    con = _conectar(ruta_bd)
    try:
        con.execute("INSERT INTO mis_cultivos (cultivo) VALUES (?)", (nombre,))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE: el cultivo ya estaba registrado
        return False
    finally:
        con.close()


def quitar(cultivo: str, ruta_bd: Path = _RUTA_BD) -> bool:
    """
    Quita un cultivo de la parcela.

    Returns:
        True si se eliminó algo; False si no estaba registrado.
    """
    nombre = _normalizar(cultivo)
    con = _conectar(ruta_bd)
    try:
        con.execute("DELETE FROM mis_cultivos WHERE cultivo = ?", (nombre,))
        eliminados = con.execute("SELECT changes()").fetchone()[0]
        con.commit()
        return eliminados > 0
    finally:
        con.close()


def listar(ruta_bd: Path = _RUTA_BD) -> list[str]:
    """Devuelve la lista de cultivos registrados, en orden alfabético."""
    con = _conectar(ruta_bd)
    filas = con.execute(
        "SELECT cultivo FROM mis_cultivos ORDER BY cultivo"
    ).fetchall()
    con.close()
    return [f["cultivo"] for f in filas]


def existe(cultivo: str, ruta_bd: Path = _RUTA_BD) -> bool:
    """Indica si un cultivo ya está registrado en la parcela."""
    nombre = _normalizar(cultivo)
    con = _conectar(ruta_bd)
    fila = con.execute(
        "SELECT 1 FROM mis_cultivos WHERE cultivo = ?", (nombre,)
    ).fetchone()
    con.close()
    return fila is not None


def limpiar_todo(ruta_bd: Path = _RUTA_BD) -> None:
    """Elimina todos los cultivos registrados (reinicio de la parcela)."""
    con = _conectar(ruta_bd)
    con.execute("DELETE FROM mis_cultivos")
    con.commit()
    con.close()
