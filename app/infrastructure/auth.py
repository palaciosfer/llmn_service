from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from app.core.entities import Rol, UsuarioAutenticado
from app.infrastructure.settings import Settings, get_settings

_bearer = HTTPBearer(auto_error=True)


def _decodificar_token(token: str, settings: Settings) -> dict:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
        )


def obtener_usuario_actual(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> UsuarioAutenticado:
    payload = _decodificar_token(credentials.credentials, settings)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token sin identificador de usuario (sub)",
        )

    rol_str = payload.get("rol", "aprendiz")
    try:
        rol = Rol(rol_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Rol no válido: {rol_str}. Use 'agricultor' o 'aprendiz'.",
        )

    return UsuarioAutenticado(
        id=str(user_id),
        rol=rol,
        email=payload.get("email"),
    )


def requerir_rol(*roles_permitidos: Rol):
    """Dependencia de FastAPI que restringe acceso por rol."""

    def _verificar(
        usuario: UsuarioAutenticado = Depends(obtener_usuario_actual),
    ) -> UsuarioAutenticado:
        if usuario.rol not in roles_permitidos:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rol '{usuario.rol.value}' no tiene acceso a este recurso",
            )
        return usuario

    return _verificar
