"""
generador.py
Genera la respuesta final con Qwen (vía Ollama, API HTTP local) a partir de los
documentos recuperados por el IR. Adapta el nivel de la respuesta según el rol.

Reglas de seguridad del dominio (CLAUDE.md):
  - Responder SOLO con base en los documentos entregados.
  - Si la información no está en los documentos, decir claramente que no se tiene.
  - NUNCA inventar tratamientos, dosis ni productos.
  - Respuestas orientativas; recomendar confirmar con un agrónomo.
"""

import re
import os
import json
from typing import Optional

import requests

# ─────────────────────────────────────────────
# Configuración de Ollama / Qwen
# ─────────────────────────────────────────────
# Configurable por entorno para el despliegue nube/dispositivo (ver CLAUDE.md):
#   - Desarrollo / offline en el móvil → Ollama local (valores por defecto).
#   - Online en producción → exportar OLLAMA_URL / QWEN_MODELO al endpoint en la nube
#     (ahí el modelo puede ser más grande, p. ej. qwen3.5:4b).

_URL_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
_MODELO = os.environ.get("QWEN_MODELO", "qwen3.5:0.8b")  # 0.8b: a bordo del móvil (offline)
_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))  # segundos
_MAX_CHARS_DOC = 3500          # recorta cada documento para no saturar el contexto

# Roles válidos y su descripción de estilo
_ROLES = {
    "agricultor": (
        "Eres un asistente agrícola que habla con un AGRICULTOR. "
        "Usa lenguaje simple, directo y práctico, sin tecnicismos. "
        "Frases cortas. Ve al grano: qué tiene la planta y qué hacer."
    ),
    "aprendiz": (
        "Eres un asistente agrícola que habla con un APRENDIZ (estudiante). "
        "Da una explicación técnica y educativa: menciona el agente causal, "
        "las condiciones que favorecen la enfermedad y el porqué de cada medida. "
        "Mantén el rigor pero sé claro."
    ),
}


# ─────────────────────────────────────────────
# Construcción del prompt
# ─────────────────────────────────────────────

def _formatear_documentos(documentos: list[dict]) -> str:
    """
    Convierte la lista de documentos recuperados en un bloque de texto numerado
    para insertarlo en el prompt. Recorta documentos muy largos.
    """
    if not documentos:
        return "(No se recuperó ningún documento.)"

    bloques = []
    for i, doc in enumerate(documentos, start=1):
        texto = (doc.get("texto", "") or "")[:_MAX_CHARS_DOC]
        fuente = doc.get("fuente", "") or "sin fuente"
        cultivo = doc.get("cultivo", "")
        enfermedad = doc.get("enfermedad", "")
        bloques.append(
            f"--- DOCUMENTO {i} "
            f"(cultivo: {cultivo}; enfermedad: {enfermedad}; fuente: {fuente}) ---\n"
            f"{texto}"
        )
    return "\n\n".join(bloques)


def _construir_prompt(
    diagnostico: dict,
    sintomas: list[str],
    documentos: list[dict],
    rol: str,
) -> str:
    """Arma el prompt completo que se envía a Qwen."""
    estilo = _ROLES.get(rol, _ROLES["agricultor"])

    cultivo = diagnostico.get("cultivo", "desconocido")
    enfermedad = diagnostico.get("enfermedad", "desconocida")
    confianza = diagnostico.get("confianza_ajustada",
                                diagnostico.get("confianza_original", 0.0))
    sintomas_txt = ", ".join(sintomas) if sintomas else "no especificados"
    bloque_docs = _formatear_documentos(documentos)

    return f"""{estilo}

REGLAS IMPORTANTES (obligatorias):
1. Responde ÚNICAMENTE con información contenida en los DOCUMENTOS de abajo.
2. Si los documentos SÍ contienen dosis, productos o pasos de tratamiento, DEBES
   incluirlos textualmente en tu respuesta. NO digas "no tengo esa información"
   cuando la información sí aparece en los documentos: léelos con atención.
3. Solo si algo realmente NO está en los documentos, di "No tengo esa información
   en mis documentos". NUNCA inventes dosis, productos ni tratamientos.
   PROHIBIDO añadir un "tratamiento estándar", "tratamiento general" o productos
   "de tu conocimiento": si los documentos no traen dosis ni productos, dilo
   claramente y recomienda acudir a un agrónomo. Más vale no dar dosis que dar una inventada.
4. Usa principalmente los documentos del MISMO cultivo y enfermedad del diagnóstico;
   ignora los documentos que traten de otro cultivo distinto.
5. Recuerda al final que la respuesta es orientativa y conviene confirmar con un agrónomo.
6. Responde en español.

DIAGNÓSTICO DEL SISTEMA (imagen + texto):
- Cultivo: {cultivo}
- Enfermedad detectada: {enfermedad}
- Confianza: {confianza}
- Síntomas del texto del usuario: {sintomas_txt}

DOCUMENTOS DISPONIBLES:
{bloque_docs}

Con base SOLO en los documentos anteriores, redacta la respuesta con estas secciones,
usando exactamente estos encabezados:

DIAGNÓSTICO:
(breve confirmación de la enfermedad y el cultivo)

TRATAMIENTO:
(qué hacer; SI los documentos mencionan dosis y productos concretos, INCLÚYELOS aquí)

PREVENCIÓN:
(cómo evitar que vuelva)

FUENTES:
(lista las fuentes de los documentos que usaste)
"""


# ─────────────────────────────────────────────
# Llamada a Ollama
# ─────────────────────────────────────────────

def _quitar_pensamiento(texto: str) -> str:
    """Elimina bloques <think>...</think> que algunos modelos Qwen emiten."""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()


def _llamar_ollama(
    prompt: str,
    modelo: str = _MODELO,
    url: str = _URL_OLLAMA,
    timeout: int = _TIMEOUT,
) -> str:
    """
    Envía el prompt a Ollama y devuelve el texto generado.

    Lanza RuntimeError con un mensaje claro si Ollama no responde, no está
    corriendo, o devuelve un error.
    """
    cuerpo = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "think": False,        # desactiva el "razonamiento" si el modelo lo soporta
        "options": {
            "temperature": 0.2,  # baja: queremos fidelidad a los documentos
        },
    }

    try:
        resp = requests.post(url, json=cuerpo, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con Ollama en "
            f"{url}. ¿Está corriendo el servidor? (ejecuta 'ollama serve')."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama no respondió en {timeout}s. El modelo puede estar cargando; "
            "reintenta."
        )

    if resp.status_code == 404:
        raise RuntimeError(
            f"El modelo '{modelo}' no existe en Ollama. "
            f"Descárgalo con: ollama pull {modelo}"
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama devolvió error {resp.status_code}: {resp.text[:300]}"
        )

    try:
        datos = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError("Ollama devolvió una respuesta no-JSON inesperada.")

    return _quitar_pensamiento(datos.get("response", "").strip())


# ─────────────────────────────────────────────
# Parseo de la respuesta en secciones
# ─────────────────────────────────────────────

def _extraer_secciones(texto: str) -> dict:
    """
    Separa el texto generado en diagnóstico, tratamiento, prevención y fuentes
    según los encabezados. Best-effort: si falta un encabezado, esa sección
    queda vacía.
    """
    secciones = {"diagnostico": "", "tratamiento": "", "prevencion": "", "fuentes": ""}
    encabezados = {
        "diagnostico": r"DIAGN[ÓO]STICO",
        "tratamiento": r"TRATAMIENTO",
        "prevencion": r"PREVENCI[ÓO]N",
        "fuentes": r"FUENTES",
    }

    # Localiza la posición de cada encabezado
    posiciones = []
    for clave, patron in encabezados.items():
        m = re.search(patron + r"\s*:", texto, flags=re.IGNORECASE)
        if m:
            posiciones.append((m.start(), m.end(), clave))
    posiciones.sort()

    # Extrae el contenido entre encabezados consecutivos
    for i, (_, fin, clave) in enumerate(posiciones):
        inicio_texto = fin
        fin_texto = posiciones[i + 1][0] if i + 1 < len(posiciones) else len(texto)
        secciones[clave] = texto[inicio_texto:fin_texto].strip()

    return secciones


def _fuentes_de_documentos(documentos: list[dict]) -> list[str]:
    """Lista las fuentes únicas de los documentos recuperados (orden estable)."""
    vistas = []
    for doc in documentos:
        fuente = (doc.get("fuente", "") or "").strip()
        if fuente and fuente not in vistas:
            vistas.append(fuente)
    return vistas


# ─────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────

def responder(
    diagnostico: dict,
    sintomas: list[str],
    documentos: list[dict],
    rol: str = "agricultor",
    modelo: str = _MODELO,
) -> dict:
    """
    Genera la respuesta final con Qwen según el rol del usuario.

    Args:
        diagnostico: dict de fusion.combinar() (cultivo, enfermedad, confianza...).
        sintomas:    lista de síntomas del texto (nlp_texto.extraer_sintomas).
        documentos:  documentos recuperados (buscar / buscar_hibrido).
        rol:         'agricultor' (simple) o 'aprendiz' (técnico).
        modelo:      nombre del modelo en Ollama.

    Returns:
        dict con:
          texto        — respuesta completa generada
          diagnostico  — sección de diagnóstico
          tratamiento  — sección de tratamiento
          prevencion   — sección de prevención
          fuentes      — lista de fuentes (derivadas de los documentos)
          rol          — rol usado
          sin_documentos — True si no había documentos en que basarse
    """
    if rol not in _ROLES:
        rol = "agricultor"

    # Sin documentos no podemos responder con seguridad: lo decimos explícitamente.
    if not documentos:
        aviso = ("No tengo documentos sobre este caso, así que no puedo dar un "
                 "tratamiento confiable. Te recomiendo consultar a un agrónomo.")
        return {
            "texto": aviso,
            "diagnostico": "",
            "tratamiento": "",
            "prevencion": "",
            "fuentes": [],
            "rol": rol,
            "sin_documentos": True,
        }

    prompt = _construir_prompt(diagnostico, sintomas, documentos, rol)
    texto = _llamar_ollama(prompt, modelo=modelo)

    secciones = _extraer_secciones(texto)
    fuentes = _fuentes_de_documentos(documentos)

    return {
        "texto": texto,
        "diagnostico": secciones["diagnostico"],
        "tratamiento": secciones["tratamiento"],
        "prevencion": secciones["prevencion"],
        "fuentes": fuentes,
        "rol": rol,
        "sin_documentos": False,
    }
