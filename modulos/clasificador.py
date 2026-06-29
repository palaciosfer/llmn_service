"""
clasificador.py
CNN de imágenes (Fase 8): foto de hoja → cultivo + enfermedad + confianza.

Modelo: EfficientNet-B4 (torchvision), 50 clases, pesos en best.pth (~97% F1).
El checkpoint best.pth es un dict con:
  - 'model'         : state_dict (claves 'backbone.features...' / 'backbone.classifier.1...')
  - 'class_mapping' : {nombre_clase: indice}
  - 'metricas'      : métricas de entrenamiento

El modelo se carga una sola vez (perezoso) y se reutiliza.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchvision import transforms
from PIL import Image

# --- Rutas / constantes ---
_DIR_BASE = Path(__file__).resolve().parent.parent
_RUTA_PESOS = _DIR_BASE / "modelos" / "best.pth"
_N_CLASES = 50
_TAM_ENTRADA = 380           # EfficientNet-B4 nativo; ajustar si el entrenamiento usó otro
_UMBRAL_CONFIANZA = 0.50     # por debajo: la foto puede no ser válida

# Normalización ImageNet (la usada al entrenar redes de torchvision)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Mapa de token de cultivo (como viene en class_mapping) → nombre en español,
# para que coincida con 'mis_cultivos' (que guarda en español/minúsculas).
_CULTIVO_ES = {
    "Apple": "manzana",
    "Blueberry": "arándano",
    "Calabaza": "calabaza",
    "Cherry_(including_sour)": "cereza",
    "Citrus": "cítrico",
    "Corn_(maize)": "maíz",
    "Frijol": "frijol",
    "Grape": "uva",
    "Orange": "naranja",
    "Peach": "durazno",
    "Pepper,_bell": "pimiento",
    "Potato": "papa",
    "Raspberry": "frambuesa",
    "Soybean": "soya",
    "Squash": "calabaza",
    "Strawberry": "fresa",
    "Tomato": "tomate",
}

# Clases de cítricos que vienen sin prefijo de cultivo en el dataset.
_CLASES_CITRICO = {"Black spot", "Canker", "Greening", "Melanose"}

# Palabras que indican planta sana.
_SANO = {"healthy", "sano", "buenos", "healthy leaf"}


# ─────────────────────────────────────────────
# Arquitectura
# ─────────────────────────────────────────────

class _RedCultivoEnfermedad(nn.Module):
    """EfficientNet-B4 con cabeza de 50 clases, envuelta en 'backbone' para
    coincidir con las claves del state_dict guardado."""

    def __init__(self, n_clases: int = _N_CLASES):
        super().__init__()
        self.backbone = efficientnet_b4(weights=None)
        n_entrada = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier[1] = nn.Linear(n_entrada, n_clases)

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────
# Carga perezosa del modelo
# ─────────────────────────────────────────────

_modelo: Optional[nn.Module] = None
_idx_a_clase: Optional[dict] = None
_transformacion: Optional[transforms.Compose] = None


def _construir_transformacion() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((_TAM_ENTRADA, _TAM_ENTRADA)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _cargar_modelo(ruta_pesos: Path = _RUTA_PESOS):
    """Carga el modelo y el class_mapping una sola vez."""
    global _modelo, _idx_a_clase, _transformacion
    if _modelo is not None:
        return

    if not ruta_pesos.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos en {ruta_pesos}."
        )

    print(f"[cnn] Cargando modelo desde {ruta_pesos.name}...")
    ckpt = torch.load(str(ruta_pesos), map_location="cpu", weights_only=False)

    class_mapping = ckpt["class_mapping"]          # nombre -> idx
    _idx_a_clase = {idx: nombre for nombre, idx in class_mapping.items()}

    modelo = _RedCultivoEnfermedad(n_clases=len(class_mapping))
    modelo.load_state_dict(ckpt["model"])
    modelo.eval()

    _modelo = modelo
    _transformacion = _construir_transformacion()
    print(f"[cnn] Modelo listo ({len(class_mapping)} clases).")


# ─────────────────────────────────────────────
# Parseo de la etiqueta de clase
# ─────────────────────────────────────────────

def _separar_clase(nombre: str) -> tuple[str, str]:
    """
    Convierte el nombre de clase del dataset en (cultivo_es, enfermedad_legible).

    Maneja los tres formatos presentes en class_mapping:
      - 'Tomato___Late_blight'        → ('tomate', 'late blight')
      - 'Calabaza_Downy Mildew'       → ('calabaza', 'downy mildew')
      - 'Black spot' (cítrico sin prefijo) → ('cítrico', 'black spot')
    Las clases sanas devuelven enfermedad 'sano'.
    """
    cultivo_token = ""
    enfermedad = nombre

    if "___" in nombre:                       # formato PlantVillage
        cultivo_token, enfermedad = nombre.split("___", 1)
    elif nombre in _CLASES_CITRICO:           # cítrico sin prefijo
        return "cítrico", nombre.strip().lower()
    elif "_" in nombre:                       # formato 'Calabaza_...' / 'Frijol_...'
        cultivo_token, enfermedad = nombre.split("_", 1)

    cultivo = _CULTIVO_ES.get(cultivo_token, cultivo_token.lower())
    enfermedad = enfermedad.replace("_", " ").strip().lower()

    if enfermedad in _SANO or any(s in enfermedad for s in _SANO):
        enfermedad = "sano"

    return cultivo, enfermedad


# ─────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────

def predecir(ruta_imagen, ruta_pesos: Path = _RUTA_PESOS) -> dict:
    """
    Predice cultivo + enfermedad a partir de una imagen de hoja.

    Args:
        ruta_imagen: ruta a la imagen (str o Path) o un objeto PIL.Image.

    Returns:
        dict con:
          cultivo       — nombre del cultivo en español
          enfermedad    — enfermedad legible (o 'sano')
          confianza     — probabilidad de la clase ganadora [0, 1]
          clase_cnn     — etiqueta cruda del modelo (ej. 'Tomato___Late_blight')
          confianza_baja — True si confianza < umbral (la foto puede no ser válida)
    """
    _cargar_modelo(ruta_pesos)

    # Aceptar ruta o imagen PIL
    if isinstance(ruta_imagen, Image.Image):
        img = ruta_imagen.convert("RGB")
    else:
        img = Image.open(ruta_imagen).convert("RGB")

    tensor = _transformacion(img).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = _modelo(tensor)
        probas = torch.softmax(logits, dim=1)
        confianza, idx = torch.max(probas, dim=1)

    idx = int(idx.item())
    confianza = float(confianza.item())
    clase = _idx_a_clase[idx]
    cultivo, enfermedad = _separar_clase(clase)

    return {
        "cultivo": cultivo,
        "enfermedad": enfermedad,
        "confianza": round(confianza, 4),
        "clase_cnn": clase,
        "confianza_baja": confianza < _UMBRAL_CONFIANZA,
    }
