"""
conexion.py
Detecta si hay conexión a internet, con un timeout corto para no bloquear la app
en el campo (donde la red es intermitente o nula).

Se usa un socket TCP a un servidor DNS público (no descarga nada, es rápido y no
depende de que un sitio web concreto esté arriba).
"""

import socket

# Servidores a probar (host, puerto). DNS de Google y Cloudflare en el puerto 53.
_DESTINOS = [
    ("8.8.8.8", 53),     # Google DNS
    ("1.1.1.1", 53),     # Cloudflare DNS
]

_TIMEOUT_POR_DEFECTO = 2.0  # segundos


def hay_internet(timeout: float = _TIMEOUT_POR_DEFECTO) -> bool:
    """
    Devuelve True si se puede abrir una conexión TCP a algún servidor DNS público.

    Args:
        timeout: Segundos máximos de espera por intento (corto, para el campo).

    Returns:
        True si hay conexión; False si todos los intentos fallan.
    """
    for host, puerto in _DESTINOS:
        try:
            with socket.create_connection((host, puerto), timeout=timeout):
                return True
        except OSError:
            continue  # probar el siguiente destino
    return False


def estado_conexion(timeout: float = _TIMEOUT_POR_DEFECTO) -> str:
    """Devuelve 'online' u 'offline' (útil para mostrar/loggear)."""
    return "online" if hay_internet(timeout) else "offline"
