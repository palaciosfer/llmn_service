"""Endpoints de desarrollo — solo disponibles cuando DEV_MODE=true."""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter

from app.infrastructure.schemas import TokenDevRequest, TokenDevResponse
from app.infrastructure.settings import get_settings

router = APIRouter(prefix="/api/v1/dev", tags=["Desarrollo"])


@router.post(
    "/token",
    response_model=TokenDevResponse,
    summary="Generar token JWT de prueba (solo modo desarrollo)",
)
async def generar_token_dev(body: TokenDevRequest) -> TokenDevResponse:
    settings = get_settings()
    expira_horas = 24

    payload = {
        "sub": body.sub,
        "rol": body.rol,
        "email": body.email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=expira_horas),
    }

    token = jwt.encode(
        payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM
    )

    return TokenDevResponse(
        access_token=token,
        rol=body.rol,
        expira_en_horas=expira_horas,
    )
