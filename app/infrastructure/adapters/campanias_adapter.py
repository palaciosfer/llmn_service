"""
Adaptador del mapa epidemiológico REAL desde las campañas fitosanitarias.

Lee datos/campanias/*.csv (datos reales de SENASICA) y agrega por entidad
federativa (estado) para el mapa epidemiológico y las alertas que consume la app.
"""

import csv
import glob
import unicodedata
from collections import Counter
from pathlib import Path

from app.core.ports import CampaniasRepositoryPort

_DIR_BASE = Path(__file__).resolve().parents[3]
_DIR_CSV = _DIR_BASE / "datos" / "campanias"


def _num(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _norm(s: str) -> str:
    """Normaliza para comparar estados (minúsculas, sin acentos ni espacios extra)."""
    s = unicodedata.normalize("NFKD", (s or "").strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


class CampaniasAdapter(CampaniasRepositoryPort):
    """Lee y agrega los CSV de campañas fitosanitarias (SENASICA)."""

    def esta_disponible(self) -> bool:
        return _DIR_CSV.exists()

    def _cargar(self) -> list[dict]:
        """Lee y unifica los CSV de campañas (maneja BOM y espacios en cabeceras)."""
        filas = []
        for ruta in sorted(glob.glob(str(_DIR_CSV / "*.csv"))):
            with open(ruta, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    rr = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
                    filas.append({
                        "campania": rr.get("cf", ""),
                        "plaga": rr.get("pa", ""),
                        "estado": rr.get("ef", ""),
                        "cultivo": rr.get("cha", "").strip().lower(),
                        "superficie": _num(rr.get("sa", 0)),
                        "productores": _num(rr.get("productores_atendidos", 0)),
                    })
        return filas

    def mapa(self) -> dict:
        filas = self._cargar()
        por_estado: dict = {}
        for f in filas:
            e = f["estado"] or "(sin estado)"
            d = por_estado.setdefault(e, {"campania": Counter(), "cultivo": Counter(),
                                          "superficie": 0.0, "productores": 0.0, "n": 0})
            d["campania"][f["campania"]] += 1
            if f["cultivo"]:
                d["cultivo"][f["cultivo"]] += 1
            d["superficie"] += f["superficie"]
            d["productores"] += f["productores"]
            d["n"] += 1

        estados = []
        for e, d in por_estado.items():
            camp = d["campania"].most_common(1)[0][0] if d["campania"] else ""
            cul = d["cultivo"].most_common(1)[0][0] if d["cultivo"] else ""
            estados.append({
                "estado": e,
                "campanias": d["n"],
                "superficie_ha": round(d["superficie"], 1),
                "productores": int(d["productores"]),
                "campania_dominante": camp,
                "cultivo_dominante": cul,
            })
        estados.sort(key=lambda x: x["superficie_ha"], reverse=True)
        return {"total_campanias": len(filas), "estados": estados}

    def alerta(self, estado: str | None = None) -> dict:
        filas = self._cargar()
        if not filas:
            return {"hay_alerta": False, "estado": estado or "",
                    "mensaje": "No existen alertas para tu región."}

        if estado:
            objetivo = _norm(estado)
            filas = [f for f in filas if _norm(f["estado"]) == objetivo]
            if not filas:
                return {"hay_alerta": False, "estado": estado,
                        "mensaje": "No existen alertas para tu región."}

        campania = Counter(f["campania"] for f in filas).most_common(1)[0][0]
        de_camp = [f for f in filas if f["campania"] == campania]
        plaga = Counter(f["plaga"] for f in de_camp if f["plaga"]).most_common(1)
        cultivo = Counter(f["cultivo"] for f in de_camp if f["cultivo"]).most_common(1)
        superficie = round(sum(f["superficie"] for f in de_camp), 1)

        ambito = f"en {estado}" if estado else "a nivel nacional"
        return {
            "hay_alerta": True,
            "estado": estado or "Nacional",
            "campania_dominante": campania,
            "plaga_dominante": plaga[0][0] if plaga else "",
            "cultivo_dominante": cultivo[0][0] if cultivo else "",
            "campanias": len(de_camp),
            "superficie_ha": superficie,
            "mensaje": f"Campaña activa {ambito}: {campania}"
                       + (f" ({cultivo[0][0]})" if cultivo else "") + ".",
        }
