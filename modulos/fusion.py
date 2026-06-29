"""
fusion.py
Combina el diagnóstico de la imagen (CNN: cultivo+enfermedad+confianza) con los
síntomas extraídos del texto (NLP) para formar:
  - una consulta enriquecida para el recuperador de documentos (TF-IDF + BERT), y
  - una confianza ajustada (sube si el texto refuerza la predicción visual,
    baja si la contradice).

Lógica simple y explicada. No depende de la CNN real (Fase 8): recibe el
resultado de la CNN como un dict, por lo que puede probarse de forma aislada.
"""

from modulos.nlp_texto import limpiar, extraer_sintomas

# ─────────────────────────────────────────────
# Conocimiento del dominio
# ─────────────────────────────────────────────

# Palabras (en español o inglés) que pueden aparecer en el nombre de enfermedad
# devuelto por la CNN → clave canónica de enfermedad. La CNN entrenada usa
# etiquetas estilo PlantVillage en inglés (ej. 'Tomato___Late_blight'); el texto
# del usuario y las pruebas usan español. Cubrimos ambos.
_ALIAS_ENFERMEDAD = {
    # oídio / cenicilla (powdery mildew)
    "oidio": "oidio", "cenicilla": "oidio", "powdery": "oidio",
    # mildiu (downy mildew)
    "mildiu": "mildiu", "downy": "mildiu", "mold": "mildiu",
    # tizón (blight)
    "tizon": "tizon", "blight": "tizon",
    # roya (rust)
    "roya": "roya", "rust": "roya",
    # mancha foliar (leaf spot)
    "mancha": "mancha", "manchas": "mancha", "spot": "mancha", "scab": "mancha",
    # pudrición (rot)
    "pudricion": "pudricion", "podredumbre": "pudricion", "rot": "pudricion",
    # marchitez (wilt)
    "marchitez": "marchitez", "marchitamiento": "marchitez", "wilt": "marchitez",
    # virus / mosaico
    "virus": "virus", "mosaico": "virus", "mosaic": "virus",
    # sano (healthy)
    "sano": "sano", "saludable": "sano", "healthy": "sano",
}

# Para cada enfermedad canónica, los síntomas (términos canónicos que produce
# nlp_texto.extraer_sintomas) que la refuerzan.
_SINTOMAS_POR_ENFERMEDAD = {
    "oidio": {"oidio", "cenicilla", "polvo blanco", "moho blanco"},
    "mildiu": {"mildiu", "humedad", "moho gris", "manchas aceitosas"},
    "tizon": {"tizon", "mildiu", "humedad", "manchas", "manchas marrones", "manchas negras"},
    "roya": {"roya", "pustulas"},
    "mancha": {"manchas", "lesiones", "manchas marrones", "manchas negras"},
    "pudricion": {"pudricion", "podredumbre"},
    "marchitez": {"marchitez", "marchitamiento"},
    "virus": {"amarillamiento", "clorosis", "deformacion", "malformacion",
              "enrollamiento", "hojas enrolladas"},
    "sano": set(),
}

# Ajuste máximo de confianza por refuerzo o contradicción del texto.
_AJUSTE_MAX = 0.15
_AJUSTE_POR_COINCIDENCIA = 0.07


# ─────────────────────────────────────────────
# Apoyos internos
# ─────────────────────────────────────────────

def _detectar_enfermedades(nombre_enfermedad: str) -> set[str]:
    """
    Detecta qué enfermedad(es) canónica(s) aparecen en el nombre devuelto por
    la CNN, buscando alias (español/inglés) dentro del texto normalizado.
    """
    texto = limpiar(nombre_enfermedad)  # minúsculas, sin acentos, solo letras
    tokens = set(texto.split())
    claves = set()
    for alias, clave in _ALIAS_ENFERMEDAD.items():
        if alias in tokens:
            claves.add(clave)
    return claves


def _sintomas_relacionados(claves: set[str]) -> set[str]:
    """Une los síntomas asociados a las enfermedades canónicas detectadas."""
    relacionados = set()
    for clave in claves:
        relacionados |= _SINTOMAS_POR_ENFERMEDAD.get(clave, set())
    return relacionados


def _todos_los_sintomas() -> set[str]:
    """Conjunto de todos los síntomas conocidos (para detectar contradicción)."""
    todos = set()
    for conjunto in _SINTOMAS_POR_ENFERMEDAD.values():
        todos |= conjunto
    return todos


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 0.99) -> float:
    """Acota un valor al rango [minimo, maximo]."""
    return max(minimo, min(maximo, valor))


# ─────────────────────────────────────────────
# Fusión
# ─────────────────────────────────────────────

def combinar(resultado_cnn: dict, sintomas_nlp: list[str]) -> dict:
    """
    Combina la predicción de la CNN con los síntomas del texto.

    Args:
        resultado_cnn: dict con al menos 'enfermedad' y 'confianza'; opcional
                       'cultivo'. Ej: {'cultivo': 'tomate',
                       'enfermedad': 'mildiu', 'confianza': 0.72}.
        sintomas_nlp:  lista de síntomas canónicos (salida de
                       nlp_texto.extraer_sintomas).

    Returns:
        dict con:
          cultivo, enfermedad            — copiados de la CNN
          confianza_original             — la de la CNN
          confianza_ajustada             — tras considerar el texto [0, 0.99]
          estado                         — 'reforzado' | 'posible_contradiccion'
                                           | 'sin_senal_textual'
          sintomas_refuerzo              — síntomas que apoyan el diagnóstico
          sintomas_contradiccion         — síntomas que apuntan a otra enfermedad
          consulta                       — texto enriquecido para el buscador
          explicacion                    — frase legible de lo que pasó
    """
    cultivo = resultado_cnn.get("cultivo", "")
    enfermedad = resultado_cnn.get("enfermedad", "")
    confianza = float(resultado_cnn.get("confianza", 0.0))

    sint = set(sintomas_nlp)
    claves = _detectar_enfermedades(enfermedad)
    relacionados = _sintomas_relacionados(claves)
    otros = _todos_los_sintomas() - relacionados

    refuerzo = sint & relacionados
    contradiccion = sint & otros

    # Ajuste de confianza: el refuerzo manda; si no hay refuerzo pero el texto
    # apunta a otra enfermedad, se baja la confianza.
    if refuerzo:
        delta = min(_AJUSTE_MAX, _AJUSTE_POR_COINCIDENCIA * len(refuerzo))
        estado = "reforzado"
    elif contradiccion:
        delta = -min(_AJUSTE_MAX, _AJUSTE_POR_COINCIDENCIA * len(contradiccion))
        estado = "posible_contradiccion"
    else:
        delta = 0.0
        estado = "sin_senal_textual"

    confianza_ajustada = round(_limitar(confianza + delta), 4)

    consulta = _construir_consulta(cultivo, enfermedad, sintomas_nlp)
    explicacion = _explicar(estado, refuerzo, contradiccion, confianza, confianza_ajustada)

    return {
        "cultivo": cultivo,
        "enfermedad": enfermedad,
        "confianza_original": round(confianza, 4),
        "confianza_ajustada": confianza_ajustada,
        "estado": estado,
        "sintomas_refuerzo": sorted(refuerzo),
        "sintomas_contradiccion": sorted(contradiccion),
        "consulta": consulta,
        "explicacion": explicacion,
    }


def _construir_consulta(cultivo: str, enfermedad: str, sintomas: list[str]) -> str:
    """
    Arma la consulta enriquecida para el IR: cultivo + enfermedad + síntomas,
    sin duplicar palabras y normalizado.
    """
    base = limpiar(f"{cultivo} {enfermedad}")
    vistas = set(base.split())
    partes = [base] if base else []
    for s in sintomas:
        if s not in vistas:
            partes.append(s)
            vistas.update(s.split())
    return " ".join(partes).strip()


def _explicar(estado, refuerzo, contradiccion, antes, despues) -> str:
    """Genera una explicación legible del ajuste."""
    if estado == "reforzado":
        return (f"El texto refuerza el diagnóstico (coincide en: "
                f"{', '.join(sorted(refuerzo))}). "
                f"Confianza {antes:.2f} → {despues:.2f}.")
    if estado == "posible_contradiccion":
        return (f"El texto apunta a otra enfermedad (síntomas: "
                f"{', '.join(sorted(contradiccion))}). "
                f"Confianza {antes:.2f} → {despues:.2f}; conviene revisar.")
    return (f"El texto no aporta señales sobre esta enfermedad. "
            f"Confianza sin cambios ({antes:.2f}).")


def diagnosticar(resultado_cnn: dict, texto: str) -> dict:
    """
    Atajo de conveniencia: extrae los síntomas del texto del usuario con
    nlp_texto y los fusiona con la predicción de la CNN.
    """
    sintomas = extraer_sintomas(texto)
    return combinar(resultado_cnn, sintomas)
