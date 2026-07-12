"""
Router de Agenda / seguimiento de tratamientos (perfil aprendiz).

Expone la agenda que la app generaba solo localmente (Hive), para que pueda
sincronizarse con el backend. Diseño:

  - La app GENERA la agenda a partir de sus diagnósticos y la SUBE al backend
    con `PUT /aprendiz/agenda` (upsert). El campo `isPendingSync` del modelo de
    la app refleja justamente ese flujo (generado local → sincronizado).
  - `GET /aprendiz/agenda` devuelve la agenda guardada del usuario autenticado.
  - `POST .../activities/{id}/complete` y `.../postpone` mutan una actividad.

Persistencia: SQLite propio (`datos/agenda.db`), separado del almacén de
documentos. La identidad del usuario sale del `sub` del JWT (obtener_usuario_actual);
la agenda es por-usuario. Contrato de datos (camelCase) = el que ya espera la app
(ver AgendaOverviewModel / AgendaActivityModel en el proyecto Flutter).
"""

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.entities import UsuarioAutenticado
from app.infrastructure.auth import obtener_usuario_actual

router = APIRouter(prefix="/api/v1", tags=["Agenda"])

_RUTA_BD = Path(__file__).resolve().parents[3] / "datos" / "agenda.db"


# ─────────────────────────────────────────────
# Esquemas (camelCase = contrato exacto de la app)
# ─────────────────────────────────────────────

class AgendaCropContext(BaseModel):
    cropName: str = ""
    currentStage: str = ""
    currentWeek: int = 0


class AgendaActivity(BaseModel):
    id: str
    title: str
    description: str
    checklist: list[str] = Field(default_factory=list)
    scheduledDate: str  # ISO-8601 (DateTime.parse en la app)
    weekNumber: int = 0
    status: str = "pending"       # pending | completed | postponed
    category: str = "generic"
    isPendingSync: bool = False


class AgendaOverview(BaseModel):
    cropContext: AgendaCropContext
    activities: list[AgendaActivity] = Field(default_factory=list)


class PostponeRequest(BaseModel):
    reason: str = ""


# ─────────────────────────────────────────────
# Persistencia SQLite (por usuario)
# ─────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_RUTA_BD)
    con.execute(
        """CREATE TABLE IF NOT EXISTS agenda_contexto (
               usuario TEXT PRIMARY KEY,
               crop_name TEXT, current_stage TEXT, current_week INTEGER
           )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS agenda_actividad (
               usuario TEXT, id TEXT, title TEXT, description TEXT,
               checklist TEXT, scheduled_date TEXT, week_number INTEGER,
               status TEXT, category TEXT, is_pending_sync INTEGER,
               PRIMARY KEY (usuario, id)
           )"""
    )
    return con


def _guardar_overview(usuario: str, ov: AgendaOverview) -> None:
    con = _conn()
    try:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO agenda_contexto "
                "(usuario, crop_name, current_stage, current_week) VALUES (?,?,?,?)",
                (usuario, ov.cropContext.cropName, ov.cropContext.currentStage,
                 ov.cropContext.currentWeek),
            )
            # La agenda subida reemplaza por completo la del usuario.
            con.execute("DELETE FROM agenda_actividad WHERE usuario = ?", (usuario,))
            con.executemany(
                "INSERT OR REPLACE INTO agenda_actividad "
                "(usuario, id, title, description, checklist, scheduled_date, "
                "week_number, status, category, is_pending_sync) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (usuario, a.id, a.title, a.description, json.dumps(a.checklist),
                     a.scheduledDate, a.weekNumber, a.status, a.category,
                     int(a.isPendingSync))
                    for a in ov.activities
                ],
            )
    finally:
        con.close()


def _fila_a_actividad(r: sqlite3.Row) -> AgendaActivity:
    return AgendaActivity(
        id=r["id"], title=r["title"], description=r["description"],
        checklist=json.loads(r["checklist"] or "[]"),
        scheduledDate=r["scheduled_date"], weekNumber=r["week_number"],
        status=r["status"], category=r["category"],
        isPendingSync=bool(r["is_pending_sync"]),
    )


def _obtener_overview(usuario: str) -> AgendaOverview:
    con = _conn()
    con.row_factory = sqlite3.Row
    try:
        ctx = con.execute(
            "SELECT crop_name, current_stage, current_week FROM agenda_contexto "
            "WHERE usuario = ?", (usuario,)
        ).fetchone()
        acts = con.execute(
            "SELECT * FROM agenda_actividad WHERE usuario = ? ORDER BY week_number, scheduled_date",
            (usuario,)
        ).fetchall()
    finally:
        con.close()

    contexto = AgendaCropContext(
        cropName=ctx["crop_name"] if ctx else "",
        currentStage=ctx["current_stage"] if ctx else "",
        currentWeek=ctx["current_week"] if ctx else 0,
    )
    return AgendaOverview(cropContext=contexto,
                          activities=[_fila_a_actividad(r) for r in acts])


def _cambiar_estado(usuario: str, activity_id: str, status: str) -> AgendaActivity | None:
    con = _conn()
    con.row_factory = sqlite3.Row
    try:
        with con:
            cur = con.execute(
                "UPDATE agenda_actividad SET status = ?, is_pending_sync = 0 "
                "WHERE usuario = ? AND id = ?", (status, usuario, activity_id),
            )
            if cur.rowcount == 0:
                return None
        r = con.execute(
            "SELECT * FROM agenda_actividad WHERE usuario = ? AND id = ?",
            (usuario, activity_id)
        ).fetchone()
        return _fila_a_actividad(r) if r else None
    finally:
        con.close()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.get(
    "/aprendiz/agenda",
    response_model=AgendaOverview,
    summary="Agenda del usuario (contexto de cultivo + actividades)",
)
async def obtener_agenda(
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> AgendaOverview:
    return await run_in_threadpool(_obtener_overview, usuario.id)


@router.put(
    "/aprendiz/agenda",
    response_model=AgendaOverview,
    summary="Sincroniza (upsert) la agenda generada por la app",
)
async def sincronizar_agenda(
    overview: AgendaOverview,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> AgendaOverview:
    await run_in_threadpool(_guardar_overview, usuario.id, overview)
    return await run_in_threadpool(_obtener_overview, usuario.id)


@router.post(
    "/aprendiz/agenda/activities/{activity_id}/complete",
    response_model=AgendaActivity,
    summary="Marca una actividad como completada",
)
async def completar_actividad(
    activity_id: str,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> AgendaActivity:
    act = await run_in_threadpool(_cambiar_estado, usuario.id, activity_id, "completed")
    if act is None:
        raise HTTPException(status_code=404, detail="Actividad no encontrada")
    return act


@router.post(
    "/aprendiz/agenda/activities/{activity_id}/postpone",
    response_model=AgendaActivity,
    summary="Pospone una actividad",
)
async def posponer_actividad(
    activity_id: str,
    body: PostponeRequest | None = None,
    usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
) -> AgendaActivity:
    act = await run_in_threadpool(_cambiar_estado, usuario.id, activity_id, "postponed")
    if act is None:
        raise HTTPException(status_code=404, detail="Actividad no encontrada")
    return act
