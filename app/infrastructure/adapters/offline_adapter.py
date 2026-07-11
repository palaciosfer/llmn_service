"""
Adaptador de catálogo/descarga de documentos para el modo offline (RAG on-device).

Un "documento offline" agrupa los fragmentos que comparten (cultivo, fuente).
Sus "chunks" son esos fragmentos, cada uno con su embedding (384-d, MiniLM-L12).
"""

import hashlib
import logging
import pickle
from collections import Counter
from typing import Optional

import numpy as np

from app.core.ports import OfflineRepositoryPort

logger = logging.getLogger(__name__)


def _doc_id(cultivo: str, fuente: str) -> str:
    """Id estable y determinista para un documento (una fuente por cultivo)."""
    clave = f"{cultivo}|{fuente}"
    return "doc_" + hashlib.md5(clave.encode("utf-8")).hexdigest()[:12]


class OfflineAdapter(OfflineRepositoryPort):
    """Adaptador que envuelve modulos.almacen_documentos / busqueda_semantica."""

    def __init__(self) -> None:
        self._disponible = False
        self._almacen = None
        self._busqueda = None
        self._emb_cache: Optional[dict] = None
        try:
            from modulos import almacen_documentos, busqueda_semantica  # type: ignore[import-untyped]

            self._almacen = almacen_documentos
            self._busqueda = busqueda_semantica
            self._disponible = True
            logger.info("Módulos de almacén/embeddings cargados (offline adapter)")
        except ImportError:
            logger.warning(
                "modulos.almacen_documentos/busqueda_semantica no encontrados — "
                "modo offline no disponible"
            )

    def esta_disponible(self) -> bool:
        return self._disponible

    def _cargar_grupos(self) -> dict:
        con = self._almacen._conectar(self._almacen._RUTA_BD)  # type: ignore[union-attr]
        filas = con.execute(
            "SELECT id, cultivo, enfermedad, fuente, texto FROM documentos "
            "ORDER BY cultivo, fuente, fragmento"
        ).fetchall()
        con.close()

        grupos: dict = {}
        for f in filas:
            did = _doc_id(f["cultivo"], f["fuente"])
            g = grupos.setdefault(did, {
                "cultivo": f["cultivo"], "fuente": f["fuente"],
                "enf": Counter(), "frag": [],
            })
            g["enf"][f["enfermedad"] or ""] += 1
            g["frag"].append((f["id"], f["texto"] or ""))

        for g in grupos.values():
            g["enfermedad"] = g["enf"].most_common(1)[0][0] if g["enf"] else ""
        return grupos

    def _emb_map(self) -> dict:
        if self._emb_cache is None:
            ruta = self._busqueda._RUTA_EMBEDDINGS  # type: ignore[union-attr]
            if not ruta.exists():
                self._emb_cache = {}
            else:
                with open(ruta, "rb") as fh:
                    d = pickle.load(fh)
                self._emb_cache = {
                    int(i): np.asarray(e, dtype=float)
                    for i, e in zip(d["ids"], d["embeddings"])
                }
        return self._emb_cache

    @staticmethod
    def _tam_bytes(fragmentos) -> int:
        return sum(len(t.encode("utf-8")) for _, t in fragmentos)

    def catalogo(self) -> dict:
        grupos = self._cargar_grupos()
        docs = []
        for did, g in grupos.items():
            title = f"{g['cultivo'].capitalize()} — {g['enfermedad']}"
            docs.append({
                "id": did,
                "crop_name": g["cultivo"],
                "disease_name": g["enfermedad"],
                "title": title,
                "source": g["fuente"] or "sin fuente",
                "size_bytes": self._tam_bytes(g["frag"]),
                "version": "1.0",
            })
        docs.sort(key=lambda d: (d["crop_name"], d["disease_name"], d["source"]))
        return {"documents": docs}

    def documento(self, doc_id: str) -> Optional[dict]:
        grupos = self._cargar_grupos()
        g = grupos.get(doc_id)
        if g is None:
            return None

        emb = self._emb_map()
        chunks = []
        vecs = []
        for idx, (row_id, texto) in enumerate(g["frag"]):
            v = emb.get(int(row_id))
            vec_list = v.tolist() if v is not None else []
            if v is not None:
                vecs.append(v)
            chunks.append({
                "id": f"{doc_id}_c{idx}",
                "index": idx,
                "text": texto,
                "embedding": vec_list,
            })

        global_emb = np.mean(vecs, axis=0).tolist() if vecs else []
        return {
            "id": doc_id,
            "content": "\n\n".join(t for _, t in g["frag"]),
            "size_bytes": self._tam_bytes(g["frag"]),
            "embedding": global_emb,
            "chunks": chunks,
        }
