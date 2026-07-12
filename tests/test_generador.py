# -*- coding: utf-8 -*-
"""
Pruebas de _extraer_secciones (modulos/generador.py).

El modelo Qwen 0.8B no respeta siempre el formato exacto "DIAGNÓSTICO:";
a menudo devuelve markdown ("### DIAGNÓSTICO", "**DIAGNÓSTICO:**") o el
encabezado sin dos puntos, o "FUENTES RELEVANTES/CONSULTADAS". El parseo debe
tolerar esas variaciones sin confundir menciones a media oración con encabezados.

Ejecutar con:  python3 -m pytest tests/test_generador.py -q
"""

from modulos.generador import _extraer_secciones


def test_extraer_secciones_formato_markdown():
    """Caso real (maíz / tizón norteño foliar): encabezados '### ' sin dos puntos."""
    texto = """Hola agricultor, aquí el análisis.
### DIAGNÓSTICO
El maíz está afectado por tizón foliar.
### TRATAMIENTO
Aplicar poda preventiva en hojas afectadas.
### PREVENCIÓN
1. Mantener campo seco.
2. Riego regular en fechas críticas.
### FUENTES RELEVANTES
* CIMMYT Enfermedades Maiz Guia Campo
* Ficha Técnica CIMMYT 2021
"""
    r = _extraer_secciones(texto)
    assert "tizón foliar" in r["diagnostico"]
    assert "poda preventiva" in r["tratamiento"]
    assert "campo seco" in r["prevencion"]
    assert r["fuentes"] != ""


def test_extraer_secciones_negrita_markdown():
    """Caso real (tomate): encabezados en negrita '**DIAGNÓSTICO:**' + PREGUNTAS."""
    texto = """**DIAGNÓSTICO:**
El cultivo es tomate y la enfermedad es tizón tardío.
**PREGUNTAS:**
1. ¿Etapa fenológica?
**FUENTES:**
Manual Fitosanitario SAGARPA 2019.
"""
    r = _extraer_secciones(texto)
    assert "tizón tardío" in r["diagnostico"]
    assert not r["diagnostico"].startswith("*")  # sin '**' de markdown sobrante
    assert "fenológica" in r["preguntas"]


def test_extraer_secciones_formato_clasico():
    """El formato original con dos puntos sigue funcionando (no-regresión)."""
    texto = """DIAGNÓSTICO:
Tomate con tizón tardío.
TRATAMIENTO:
Aplicar fungicida cúprico.
PREVENCIÓN:
Rotación de cultivos.
FUENTES:
INTA Ediciones.
"""
    r = _extraer_secciones(texto)
    assert "tizón tardío" in r["diagnostico"]
    assert "fungicida" in r["tratamiento"]
    assert "Rotación" in r["prevencion"]


def test_extraer_secciones_no_confunde_menciones():
    """Una mención en medio de una oración NO debe tomarse como encabezado."""
    texto = """DIAGNÓSTICO
El maíz tiene tizón. El tratamiento del cultivo depende del clima.
TRATAMIENTO
Aplicar fungicida.
"""
    r = _extraer_secciones(texto)
    assert r["tratamiento"].strip() == "Aplicar fungicida."
    assert "tratamiento del cultivo" in r["diagnostico"]


def test_extraer_secciones_fuentes_consultadas():
    """'FUENTES CONSULTADAS' con '##' también se reconoce."""
    texto = """## DIAGNÓSTICO
Cultivo sano en general.
## FUENTES CONSULTADAS
Guía MAGyP.
"""
    r = _extraer_secciones(texto)
    assert "sano" in r["diagnostico"]
    assert "MAGyP" in r["fuentes"]
