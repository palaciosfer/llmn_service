"""
preparar_datos.py — Construye el almacén SQLite, índices TF-IDF y embeddings BERT.

Ejecutar UNA VEZ antes de levantar el microservicio:
    python scripts/preparar_datos.py

Lo que hace:
  1. Carga el corpus combinado (datos/corpus_combinado.json) en SQLite.
  2. Carga los documentos curados (.txt) de documentos/.
  3. Construye el índice TF-IDF (datos/tfidf.pkl).
  4. Construye los embeddings BERT (datos/embeddings_bert.pkl).
  5. Registra cultivos por defecto en 'mis_cultivos'.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulos.almacen_documentos import (
    agregar_corpus,
    cargar_desde_directorio,
    construir_indice,
    listar_documentos,
    _RUTA_BD,
)
from modulos.busqueda_semantica import construir_embeddings
from modulos import mis_cultivos

_DIR = Path(__file__).resolve().parent.parent
_RUTA_CORPUS = _DIR / "datos" / "corpus_combinado.json"
_CULTIVOS_DEFECTO = ["maíz", "calabaza", "frijol", "tomate", "papa", "fresa"]


def main() -> None:
    print("=" * 60)
    print("  Preparando datos para el microservicio")
    print("=" * 60)

    # 1) Limpiar BD anterior si existe
    if _RUTA_BD.exists():
        _RUTA_BD.unlink()
        print(f"\n  BD anterior eliminada: {_RUTA_BD.name}")

    # 2) Cargar corpus JSON
    if _RUTA_CORPUS.exists():
        data = json.load(open(_RUTA_CORPUS, encoding="utf-8"))
        registros = [
            {
                "cultivo": r.get("cultivo", "general"),
                "enfermedad": r.get("tema_plaga", ""),
                "fuente": r.get("fuente", ""),
                "texto": r.get("contenido_texto", ""),
            }
            for r in data
            if r.get("contenido_texto", "").strip()
        ]
        n_corpus = agregar_corpus(registros)
        print(f"\n  Corpus JSON: {n_corpus} fragmentos cargados")
    else:
        print(f"\n  AVISO: {_RUTA_CORPUS.name} no encontrado, se omite")

    # 3) Cargar documentos curados (.txt)
    dir_docs = _DIR / "documentos"
    if dir_docs.exists():
        n_txt = cargar_desde_directorio(dir_docs)
        print(f"  Documentos curados (.txt): {n_txt} cargados")
    else:
        print("  AVISO: directorio documentos/ no encontrado")

    # 4) Resumen
    docs = listar_documentos()
    print(f"\n  Total documentos en almacén: {len(docs)}")
    por_cultivo = Counter(d["cultivo"] for d in docs)
    for cultivo, n in sorted(por_cultivo.items()):
        print(f"    {cultivo:20s} : {n:4d}")

    # 5) Construir índice TF-IDF
    print("\n  Construyendo índice TF-IDF...")
    construir_indice()

    # 6) Construir embeddings BERT
    print("\n  Construyendo embeddings BERT (puede tardar ~1 min)...")
    construir_embeddings()

    # 7) Registrar cultivos
    mis_cultivos.limpiar_todo()
    for c in _CULTIVOS_DEFECTO:
        mis_cultivos.agregar(c)
    print(f"\n  Cultivos registrados: {mis_cultivos.listar()}")

    print("\n" + "=" * 60)
    print("  Datos listos. Puedes levantar el microservicio:")
    print("    uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
