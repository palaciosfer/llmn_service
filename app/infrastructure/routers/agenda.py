"""
Router de Agenda / seguimiento de tratamientos.

La agenda se **genera automáticamente** a partir del tratamiento/prevención de un
diagnóstico (opt-in del usuario con "Agregar a la agenda"), y existe para **ambos
roles** con un **plan distinto por rol** (porque el tratamiento ya viene distinto
por rol: agricultor = práctico, aprendiz = pedagógico).

Rutas (por rol, `{rol}` ∈ {agricultor, aprendiz}):
  - GET  /api/v1/{rol}/agenda                          -> overview del usuario+rol
  - POST /api/v1/{rol}/agenda/generar                  -> genera y guarda desde un diagnóstico
  - POST /api/v1/{rol}/agenda/activities/{id}/complete
  - POST /api/v1/{rol}/agenda/activities/{id}/postpone

Generación de fechas: se le pide al LLM (Ollama/Qwen) que proponga, para cada paso,
`offsetDias` (días desde hoy) y `semana`. Como un modelo pequeño no siempre devuelve
JSON válido, hay un **fallback determinista** que reparte los pasos por una regla
fija — así la agenda SIEMPRE se genera.

Persistencia: SQLite propio (`datos/agenda.db`), por (usuario, rol). Identidad del
usuario = `sub` del JWT.
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.entities import UsuarioAutenticado
from app.infrastructure.auth import obtener_usuario_actual

router = APIRouter(prefix="/api/v1", tags=["Agenda"])

_RUTA_BD = Path(__file__).resolve().parents[3] / "datos" / "agenda.db"
_MAX_ACTIVIDADES = 8


# ─────────────────────────────────────────────
# Esquemas (camelCase = contrato de la app)
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
    scheduledDate: str
    weekNumber: int = 0
    status: str = "pending"
    category: str = "generic"
    isPendingSync: bool = False


class AgendaOverview(BaseModel):
    cropContext: AgendaCropContext
    activities: list[AgendaActivity] = Field(default_factory=list)


class GenerarAgendaRequest(BaseModel):
    """Lo que la app ya tiene tras /consultar; con esto se arma la agenda."""
    cultivo: str
    enfermedad: str = ""
    tratamiento: str = ""
    prevencion: str = ""
    currentStage: str = ""


# ─────────────────────────────────────────────
# Persistencia SQLite (por usuario + rol)
# ─────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _RUTA_BD.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_RUTA_BD)
    con.execute(
        """CREATE TABLE IF NOT EXISTS agenda_ctx (
               usuario TEXT, rol TEXT, crop_name TEXT, current_stage TEXT,
               current_week INTEGER, PRIMARY KEY (usuario, rol)
           )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS agenda_act (
               usuario TEXT, rol TEXT, id TEXT, title TEXT, description TEXT,
               checklist TEXT, scheduled_date TEXT, week_number INTEGER,
               status TEXT, category TEXT, is_pending_sync INTEGER,
               PRIMARY KEY (usuario, rol, id)
           )"""
    )
    return con


def _guardar(usuario: str, rol: str, ov: AgendaOverview) -> None:
    con = _conn()
    try:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO agenda_ctx "
                "(usuario, rol, crop_name, current_stage, current_week) VALUES (?,?,?,?,?)",
                (usuario, rol, ov.cropContext.cropName,
                 ov.cropContext.currentStage, ov.cropContext.currentWeek),
            )
            con.execute("DELETE FROM agenda_act WHERE usuario=? AND rol=?", (usuario, rol))
            con.executemany(
                "INSERT OR REPLACE INTO agenda_act "
                "(usuario, rol, id, title, description, checklist, scheduled_date, "
                "week_number, status, category, is_pending_sync) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (usuario, rol, a.id, a.title, a.description, json.dumps(a.checklist),
                     a.scheduledDate, a.weekNumber, a.status, a.category, int(a.isPendingSync))
                    for a in ov.activities
                ],
            )
    finally:
        con.close()


def _fila(r: sqlite3.Row) -> AgendaActivity:
    return AgendaActivity(
        id=r["id"], title=r["title"], description=r["description"],
        checklist=json.loads(r["checklist"] or "[]"),
        scheduledDate=r["scheduled_date"], weekNumber=r["week_number"],
        status=r["status"], category=r["category"],
        isPendingSync=bool(r["is_pending_sync"]),
    )


def _obtener(usuario: str, rol: str) -> AgendaOverview:
    con = _conn()
    con.row_factory = sqlite3.Row
    try:
        ctx = con.execute(
            "SELECT crop_name, current_stage, current_week FROM agenda_ctx "
            "WHERE usuario=? AND rol=?", (usuario, rol)
        ).fetchone()
        acts = con.execute(
            "SELECT * FROM agenda_act WHERE usuario=? AND rol=? "
            "ORDER BY week_number, scheduled_date", (usuario, rol)
        ).fetchall()
    finally:
        con.close()
    contexto = AgendaCropContext(
        cropName=ctx["crop_name"] if ctx else "",
        currentStage=ctx["current_stage"] if ctx else "",
        currentWeek=ctx["current_week"] if ctx else 0,
    )
    return AgendaOverview(cropContext=contexto, activities=[_fila(r) for r in acts])


def _cambiar_estado(usuario: str, rol: str, act_id: str, status: str) -> AgendaActivity | None:
    con = _conn()
    con.row_factory = sqlite3.Row
    try:
        with con:
            cur = con.execute(
                "UPDATE agenda_act SET status=?, is_pending_sync=0 "
                "WHERE usuario=? AND rol=? AND id=?", (status, usuario, rol, act_id),
            )
            if cur.rowcount == 0:
                return None
        r = con.execute(
            "SELECT * FROM agenda_act WHERE usuario=? AND rol=? AND id=?",
            (usuario, rol, act_id)
        ).fetchone()
        return _fila(r) if r else None
    finally:
        con.close()


# ─────────────────────────────────────────────
# Generación de la agenda (LLM propone fechas, con fallback determinista)
# ─────────────────────────────────────────────

def _pasos(req: GenerarAgendaRequest) -> list[tuple[str, str]]:
    """Extrae los pasos REALES (texto, categoría) de tratamiento y prevención.
    El texto de la actividad sale de aquí — NUNCA del LLM — para que siempre sea
    fiel a los documentos y no dependa de que el modelo pequeño redacte bien."""
    pasos: list[tuple[str, str]] = []
    for texto, cat in ((req.tratamiento, "tratamiento"), (req.prevencion, "prevencion")):
        for linea in (texto or "").splitlines():
            s = linea.strip().lstrip("-*•").strip()
            if len(s) >= 4:
                pasos.append((s, cat))
    return pasos[:_MAX_ACTIVIDADES]


def _fecha(offset_dias: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=max(0, int(offset_dias)))) \
        .strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _horario_regla(n: int) -> dict[int, tuple[int, int]]:
    """Fallback determinista: un paso cada +2 días desde hoy. {paso: (offsetDias, semana)}."""
    return {i + 1: (i * 2, (i * 2) // 7 + 1) for i in range(n)}


def _horario_llm(pasos: list[tuple[str, str]], req: GenerarAgendaRequest) -> dict[int, tuple[int, int]]:
    """Pide al LLM SOLO el calendario (offsetDias/semana por número de paso).
    Devuelve {} si el modelo no responde o no da JSON parseable (se usa la regla)."""
    from modulos import generador  # type: ignore[import-untyped]
    lista = "\n".join(f"{i + 1}. {p[0]}" for i, p in enumerate(pasos))
    prompt = (
        f"Eres un planificador agrícola. Para {req.cultivo} con {req.enfermedad}, "
        "propón CUÁNDO hacer cada paso. Devuelve SOLO un arreglo JSON, sin texto extra:\n"
        '[{"paso": 1, "offsetDias": 0, "semana": 1}, {"paso": 2, "offsetDias": 3, "semana": 1}]\n'
        "offsetDias = días desde hoy (0 = hoy). semana = 1, 2, 3...\n\n"
        f"PASOS:\n{lista}\n\nJSON:"
    )
    try:
        texto = generador._llamar_ollama(prompt)
    except Exception:
        return {}
    m = re.search(r"\[.*\]", texto, flags=re.DOTALL)
    if not m:
        return {}
    try:
        datos = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(datos, list):
        return {}
    h: dict[int, tuple[int, int]] = {}
    for d in datos:
        if not isinstance(d, dict):
            continue
        try:
            paso = int(d.get("paso"))
            off = int(d.get("offsetDias", 0) or 0)
            semana = int(d.get("semana", off // 7 + 1) or 1)
        except (TypeError, ValueError):
            continue
        h[paso] = (off, semana)
    return h


def _generar(usuario: str, rol: str, req: GenerarAgendaRequest) -> AgendaOverview:
    pasos = _pasos(req)
    horario = _horario_llm(pasos, req) if pasos else {}
    regla = _horario_regla(len(pasos))
    actividades = []
    for i, (texto, categoria) in enumerate(pasos):
        # El LLM propone la fecha; si falta ese paso, cae a la regla determinista.
        off, semana = horario.get(i + 1, regla[i + 1])
        actividades.append(AgendaActivity(
            id=f"act_{i + 1}", title=texto[:80], description=texto,
            scheduledDate=_fecha(off), weekNumber=semana, category=categoria,
        ))
    ov = AgendaOverview(
        cropContext=AgendaCropContext(
            cropName=req.cultivo, currentStage=req.currentStage, currentWeek=1),
        activities=actividades,
    )
    _guardar(usuario, rol, ov)
    return ov


# ─────────────────────────────────────────────
# Registro de endpoints por rol (agricultor y aprendiz)
# ─────────────────────────────────────────────

def _registrar_rol(rol: str) -> None:
    base = f"/{rol}/agenda"

    @router.get(base, response_model=AgendaOverview, name=f"agenda_{rol}_get",
                summary=f"Agenda del usuario ({rol})")
    async def _get(usuario: UsuarioAutenticado = Depends(obtener_usuario_actual)) -> AgendaOverview:
        return await run_in_threadpool(_obtener, usuario.id, rol)

    @router.post(f"{base}/generar", response_model=AgendaOverview, name=f"agenda_{rol}_generar",
                 summary=f"Genera la agenda ({rol}) desde un diagnóstico (LLM + fallback)")
    async def _generar_ep(
        req: GenerarAgendaRequest,
        usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
    ) -> AgendaOverview:
        return await run_in_threadpool(_generar, usuario.id, rol, req)

    @router.post(f"{base}/activities/{{activity_id}}/complete",
                 response_model=AgendaActivity, name=f"agenda_{rol}_complete",
                 summary=f"Completar actividad ({rol})")
    async def _complete(
        activity_id: str,
        usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
    ) -> AgendaActivity:
        act = await run_in_threadpool(_cambiar_estado, usuario.id, rol, activity_id, "completed")
        if act is None:
            raise HTTPException(status_code=404, detail="Actividad no encontrada")
        return act

    @router.post(f"{base}/activities/{{activity_id}}/postpone",
                 response_model=AgendaActivity, name=f"agenda_{rol}_postpone",
                 summary=f"Posponer actividad ({rol})")
    async def _postpone(
        activity_id: str,
        usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
    ) -> AgendaActivity:
        act = await run_in_threadpool(_cambiar_estado, usuario.id, rol, activity_id, "postponed")
        if act is None:
            raise HTTPException(status_code=404, detail="Actividad no encontrada")
        return act


for _rol in ("agricultor", "aprendiz"):
    _registrar_rol(_rol)
