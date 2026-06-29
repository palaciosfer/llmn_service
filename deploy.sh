#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  deploy.sh — Despliega la API de diagnostico agricola en EC2
#
#  Requisitos previos:
#    1. Una instancia EC2 con Ubuntu 22.04+ (minimo t3.medium, 4GB RAM)
#    2. Puertos 8000 y 22 abiertos en el Security Group
#    3. Archivo .env creado manualmente en la raiz del proyecto
#
#  Uso:
#    chmod +x deploy.sh
#    ./deploy.sh
# ============================================================

MODELO_LLM="${QWEN_MODELO:-qwen3.5:0.8b}"

echo "=================================================="
echo "  Desplegando API de diagnostico agricola"
echo "=================================================="

# ── 1. Verificar que existe .env ────────────────────────────
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: No se encontro el archivo .env"
    echo ""
    echo "Crea el archivo .env con al menos:"
    echo ""
    echo "  JWT_SECRET=tu-clave-secreta-aqui"
    echo "  QWEN_MODELO=qwen3.5:0.8b"
    echo "  DEV_MODE=true"
    echo "  OLLAMA_TIMEOUT=180"
    echo "  LOG_LEVEL=INFO"
    echo "  CORS_ORIGINS=*"
    echo ""
    exit 1
fi

echo ""
echo "[1/6] Instalando Docker..."

if ! command -v docker &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "  Docker instalado. Si es la primera vez, cierra sesion y vuelve a entrar"
    echo "  para que el grupo docker tome efecto, luego re-ejecuta ./deploy.sh"
else
    echo "  Docker ya instalado: $(docker --version)"
fi

# ── 2. Verificar docker compose ─────────────────────────────
echo ""
echo "[2/6] Verificando Docker Compose..."

if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "  Instalando docker-compose-plugin..."
    sudo apt-get install -y -qq docker-compose-plugin
    COMPOSE="docker compose"
fi
echo "  Compose listo: $($COMPOSE version)"

# ── 3. Construir y levantar ─────────────────────────────────
echo ""
echo "[3/6] Construyendo imagen Docker (puede tardar 5-10 min la primera vez)..."

$COMPOSE build --no-cache

echo ""
echo "[4/6] Levantando servicios (API + Ollama)..."

$COMPOSE up -d

# ── 4. Esperar a que Ollama este listo ──────────────────────
echo ""
echo "[5/6] Esperando a que Ollama inicie..."

INTENTOS=0
MAX_INTENTOS=30
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    INTENTOS=$((INTENTOS + 1))
    if [ "$INTENTOS" -ge "$MAX_INTENTOS" ]; then
        echo "  ERROR: Ollama no respondio despues de $MAX_INTENTOS intentos"
        echo "  Revisa los logs: $COMPOSE logs ollama"
        exit 1
    fi
    sleep 2
done
echo "  Ollama listo"

# ── 5. Descargar el modelo LLM ──────────────────────────────
echo ""
echo "[6/6] Descargando modelo LLM: $MODELO_LLM ..."

docker compose exec -T ollama ollama pull "$MODELO_LLM"

echo "  Modelo $MODELO_LLM descargado"

# ── 6. Esperar a que la API este lista ──────────────────────
echo ""
echo "Esperando a que la API cargue los modelos (BERT ~30s)..."

INTENTOS=0
MAX_INTENTOS=60
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    INTENTOS=$((INTENTOS + 1))
    if [ "$INTENTOS" -ge "$MAX_INTENTOS" ]; then
        echo "  ERROR: La API no respondio despues de $MAX_INTENTOS intentos"
        echo "  Revisa los logs: $COMPOSE logs api"
        exit 1
    fi
    sleep 3
done

# ── Resultado ───────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  DESPLIEGUE COMPLETADO"
echo "=================================================="
echo ""

HEALTH=$(curl -s http://localhost:8000/health)
READY=$(curl -s http://localhost:8000/ready 2>/dev/null || echo '{"status":"degraded"}')

echo "  Health:  $HEALTH"
echo "  Ready:   $READY"
echo ""
echo "  API:     http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP_PUBLICA>'):8000"
echo "  Swagger: http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP_PUBLICA>'):8000/docs"
echo ""
echo "  Logs:    $COMPOSE logs -f api"
echo "  Parar:   $COMPOSE down"
echo "  Reiniciar: $COMPOSE restart"
echo ""
echo "=================================================="
