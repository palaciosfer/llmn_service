"""
Genera un token JWT de prueba para testing local.

Uso:
    python scripts/generar_token.py                          # agricultor por defecto
    python scripts/generar_token.py --rol aprendiz           # aprendiz
    python scripts/generar_token.py --sub user123 --rol agricultor --horas 48
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar JWT de prueba")
    parser.add_argument("--sub", default="dev-user-1", help="ID del usuario")
    parser.add_argument(
        "--rol",
        default="agricultor",
        choices=["agricultor", "aprendiz"],
        help="Rol del usuario",
    )
    parser.add_argument("--email", default="dev@test.com", help="Email")
    parser.add_argument(
        "--secret",
        default="cambiar-en-produccion-clave-super-secreta",
        help="JWT secret (debe coincidir con JWT_SECRET del .env)",
    )
    parser.add_argument(
        "--horas", type=int, default=24, help="Horas de validez"
    )
    args = parser.parse_args()

    ahora = datetime.now(timezone.utc)
    payload = {
        "sub": args.sub,
        "rol": args.rol,
        "email": args.email,
        "iat": ahora,
        "exp": ahora + timedelta(hours=args.horas),
    }

    token = jwt.encode(payload, args.secret, algorithm="HS256")

    print(f"\n{'=' * 60}")
    print(f"  Token JWT generado")
    print(f"{'=' * 60}")
    print(f"  sub:   {args.sub}")
    print(f"  rol:   {args.rol}")
    print(f"  email: {args.email}")
    print(f"  exp:   {ahora + timedelta(hours=args.horas)}")
    print(f"{'=' * 60}")
    print(f"\n{token}\n")
    print(f"  Header para peticiones:")
    print(f"  Authorization: Bearer {token}\n")


if __name__ == "__main__":
    main()
