"""
nlp_texto.py
Procesa el texto breve de síntomas que escribe el usuario (ej. 'hojas amarillas
y polvo blanco') y devuelve una lista de síntomas/palabras clave normalizada
para enriquecer la búsqueda (TF-IDF + BERT).

NLP ligero en español: solo biblioteca estándar (re, unicodedata). Sin spaCy/NLTK
ni modelos pesados, según las convenciones del proyecto (dependencias mínimas).
"""

import re
import unicodedata

# ─────────────────────────────────────────────
# Recursos lingüísticos (español, dominio agrícola)
# ─────────────────────────────────────────────

# Palabras vacías frecuentes en español. Se quitan porque no aportan a la búsqueda.
_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "ante", "con", "en", "por", "para", "sin", "sobre", "tras", "y", "e",
    "o", "u", "que", "se", "su", "sus", "mi", "mis", "tu", "tus", "le", "les",
    "lo", "me", "te", "nos", "es", "son", "esta", "este", "estos", "estas",
    "tiene", "tienen", "tengo", "hay", "muy", "mas", "más", "pero", "como",
    "ya", "tambien", "también", "cuando", "donde", "porque", "si", "no",
    "del", "algo", "todo", "toda", "todos", "todas", "alguna", "algunas",
    "creo", "parece", "veo", "tengo", "estan", "están", "esta", "está",
}

# Léxico de síntomas: cada clave es una expresión que el usuario podría escribir
# (frase o palabra) y el valor es el/los término(s) canónico(s) con los que
# conviene enriquecer la búsqueda. Permite que "polvo blanco" recupere "oídio"
# aunque el usuario no use esa palabra.
_LEXICO_SINTOMAS = {
    # Oídio / cenicilla
    "polvo blanco": ["oidio", "cenicilla", "polvo blanco"],
    "polvillo blanco": ["oidio", "cenicilla", "polvo blanco"],
    "ceniza": ["oidio", "cenicilla"],
    "cenicilla": ["oidio", "cenicilla"],
    "moho blanco": ["oidio", "moho blanco"],
    # Mildiu / tizón (humedad)
    "moho gris": ["mildiu", "moho gris"],
    "manchas aceitosas": ["mildiu", "tizon", "manchas aceitosas"],
    "humedad": ["humedad", "mildiu"],
    "tizon": ["tizon", "mildiu"],
    "tizón": ["tizon", "mildiu"],
    # Clorosis / amarillamiento
    "hojas amarillas": ["amarillamiento", "clorosis"],
    "amarillas": ["amarillamiento", "clorosis"],
    "amarillamiento": ["amarillamiento", "clorosis"],
    "amarillentas": ["amarillamiento", "clorosis"],
    # Manchas y lesiones
    "manchas": ["manchas", "lesiones"],
    "mancha": ["manchas", "lesiones"],
    "manchas marrones": ["manchas marrones", "lesiones"],
    "manchas cafe": ["manchas marrones", "lesiones"],
    "manchas negras": ["manchas negras", "lesiones"],
    "puntos negros": ["manchas negras", "lesiones"],
    # Pudrición / marchitez
    "podredumbre": ["pudricion", "podredumbre"],
    "pudricion": ["pudricion", "podredumbre"],
    "se pudre": ["pudricion", "podredumbre"],
    "marchita": ["marchitez", "marchitamiento"],
    "marchitas": ["marchitez", "marchitamiento"],
    "marchitez": ["marchitez", "marchitamiento"],
    "marchitamiento": ["marchitez", "marchitamiento"],
    # Roya / pústulas
    "roya": ["roya", "pustulas"],
    "pustulas": ["roya", "pustulas"],
    "polvo naranja": ["roya", "pustulas"],
    # Otros
    "hojas secas": ["secamiento", "hojas secas"],
    "se secan": ["secamiento", "hojas secas"],
    "enrolladas": ["enrollamiento", "hojas enrolladas"],
    "deformes": ["deformacion", "malformacion"],
}

# Términos del dominio que, si aparecen sueltos, vale la pena conservar como
# palabra clave aunque no estén en el léxico de frases.
_PALABRAS_CLAVE_DOMINIO = {
    "hoja", "hojas", "tallo", "tallos", "fruto", "frutos", "raiz", "raices",
    "flor", "flores", "planta", "plantas", "blanco", "negro", "negra",
    "marron", "cafe", "naranja", "gris", "amarillo", "amarilla", "verde",
    "polvo", "moho", "mancha", "manchas", "lesion", "pudricion", "humedad",
    "seco", "seca", "secas", "marchita", "deforme",
}


# ─────────────────────────────────────────────
# Limpieza y tokenización
# ─────────────────────────────────────────────

def _quitar_acentos(texto: str) -> str:
    """Elimina tildes/diéresis para comparar de forma robusta (interno)."""
    nfkd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def limpiar(texto: str) -> str:
    """
    Normaliza el texto: minúsculas, sin signos de puntuación ni números,
    sin acentos, y con los espacios colapsados.

    Se quitan los acentos para que la coincidencia con el léxico sea robusta
    ('tizón' == 'tizon', 'café' == 'cafe').
    """
    texto = texto.lower()
    texto = _quitar_acentos(texto)
    # Conservar solo letras (a-z, ñ ya quedó como n tras quitar acentos) y espacios
    texto = re.sub(r"[^a-zñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list[str]:
    """
    Tokeniza el texto ya limpio: separa en palabras y quita stopwords y
    tokens de 2 caracteres o menos.
    """
    limpio = limpiar(texto)
    stop_sin_acentos = {_quitar_acentos(s) for s in _STOPWORDS}
    return [
        tok for tok in limpio.split()
        if tok not in stop_sin_acentos and len(tok) > 2
    ]


# ─────────────────────────────────────────────
# Extracción de síntomas
# ─────────────────────────────────────────────

def extraer_sintomas(texto: str) -> list[str]:
    """
    Extrae una lista de síntomas/palabras clave normalizada a partir del texto
    del usuario, lista para enriquecer la búsqueda.

    Combina dos fuentes:
      1. Coincidencias con el léxico de síntomas (frases y palabras → términos
         canónicos del dominio, ej. 'polvo blanco' → 'oidio', 'cenicilla').
      2. Palabras clave del dominio que aparezcan sueltas (ej. 'tallo', 'fruto').

    Devuelve la lista sin duplicados, preservando el orden de aparición.
    """
    if not texto or not texto.strip():
        return []

    limpio = limpiar(texto)
    sintomas: list[str] = []

    def _agregar(termino: str) -> None:
        if termino and termino not in sintomas:
            sintomas.append(termino)

    # 1) Léxico: buscar primero las frases más largas para no fragmentarlas.
    #    (las claves del léxico ya están sin acentos, igual que 'limpio')
    for expresion in sorted(_LEXICO_SINTOMAS, key=len, reverse=True):
        expr_norm = _quitar_acentos(expresion)
        if re.search(rf"\b{re.escape(expr_norm)}\b", limpio):
            for canon in _LEXICO_SINTOMAS[expresion]:
                _agregar(canon)

    # 2) Palabras clave del dominio sueltas (tokenizadas, sin stopwords)
    claves_sin_acentos = {_quitar_acentos(p) for p in _PALABRAS_CLAVE_DOMINIO}
    for tok in tokenizar(texto):
        if tok in claves_sin_acentos:
            _agregar(tok)

    return sintomas


def enriquecer_consulta(texto: str) -> str:
    """
    Devuelve una consulta de texto enriquecida = texto limpio + síntomas
    canónicos detectados. Pensada para pasarse directamente al buscador
    (TF-IDF + BERT) y mejorar el recall.
    """
    base = limpiar(texto)
    sintomas = extraer_sintomas(texto)
    extra = " ".join(s for s in sintomas if s not in base.split())
    return (base + " " + extra).strip()
