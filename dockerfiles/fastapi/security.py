"""Autenticacion por API key con roles, rate limiting y proteccion contra fuerza bruta.

Las claves se configuran por variable de entorno (nunca en el codigo):
    API_KEYS="nombre:rol:clave,nombre2:rol:clave2"      roles: predictor | admin
Si no hay claves configuradas, TODO acceso protegido se rechaza (falla cerrada).

Los contadores viven en Valkey para que el limite sea comun a todas las instancias. Si Valkey no
responde se deja pasar el trafico (falla abierta, solo para el rate limit) y se avisa por log:
se prioriza la disponibilidad del servicio clinico sobre el limite.
"""
import hmac
import os
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

import audit
import stream

ROLES = {"predictor": 1, "admin": 2}

# Declara la clave en la documentacion de la API (/docs): habilita el boton "Authorize".
# La validacion real se hace en `requiere`, leyendo la cabecera de la peticion.
ESQUEMA_CLAVE = APIKeyHeader(name="x-api-key", auto_error=False, description="API key (rol predictor o admin)")
LIMITE_POR_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
LIMITE_FALLOS_POR_MIN_Y_ORIGEN = 20     # intentos con clave invalida por origen y minuto


@dataclass(frozen=True)
class Cliente:
    nombre: str
    rol: str


def _cargar_claves() -> list[tuple[str, Cliente]]:
    claves = []
    for item in os.getenv("API_KEYS", "").split(","):
        item = item.strip()
        if not item:
            continue
        partes = item.split(":", 2)
        if len(partes) != 3 or partes[1] not in ROLES or not partes[2]:
            print(f"[seguridad] entrada de API_KEYS ignorada (formato nombre:rol:clave): '{partes[0]}'", flush=True)
            continue
        claves.append((partes[2], Cliente(partes[0], partes[1])))
    if not claves:
        print("[seguridad] API_KEYS vacio: se rechazara todo acceso protegido", flush=True)
    return claves


_CLAVES = _cargar_claves()


def autenticar(clave: str | None) -> Cliente | None:
    """Devuelve el cliente dueno de la clave, o None. Comparacion en tiempo constante."""
    if not clave:
        return None
    encontrado = None
    for valida, cliente in _CLAVES:          # se recorren todas para no filtrar cuál coincidió
        if hmac.compare_digest(clave.encode(), valida.encode()):
            encontrado = cliente
    return encontrado


def _contar(clave: str, ventana_s: int = 60) -> int | None:
    """Incrementa un contador con vencimiento. None si Valkey no responde."""
    try:
        r = stream.cliente()
        n = r.incr(clave)
        if n == 1:
            r.expire(clave, ventana_s + 10)
        return int(n)
    except Exception as exc:
        print(f"[seguridad] contador no disponible ({exc}); se deja pasar", flush=True)
        return None


def excede_limite(cliente: Cliente) -> bool:
    n = _contar(f"rl:{cliente.nombre}:{int(time.time() // 60)}")
    return n is not None and n > LIMITE_POR_MIN


def demasiados_fallos(origen: str) -> bool:
    n = _contar(f"authfail:{origen}:{int(time.time() // 60)}")
    return n is not None and n > LIMITE_FALLOS_POR_MIN_Y_ORIGEN


def requiere(rol_minimo: str):
    """Dependencia de FastAPI: autentica, autoriza por rol, limita la tasa y audita."""
    def dependencia(request: Request, _clave: str | None = Security(ESQUEMA_CLAVE)) -> Cliente:
        origen = request.client.host if request.client else "desconocido"
        recurso = f"{request.method} {request.url.path}"
        cliente = autenticar(request.headers.get("x-api-key"))

        if cliente is None:
            audit.registrar("autenticacion_fallida", "anonimo", recurso, "rest", origen=origen)
            if demasiados_fallos(origen):
                raise HTTPException(429, "Demasiados intentos fallidos", headers={"Retry-After": "60"})
            raise HTTPException(401, "API key ausente o invalida", headers={"WWW-Authenticate": "ApiKey"})
        if ROLES[cliente.rol] < ROLES[rol_minimo]:
            audit.registrar("acceso_denegado", cliente.nombre, recurso, "rest", origen=origen,
                            detalle=f"rol {cliente.rol}, requerido {rol_minimo}")
            raise HTTPException(403, f"Se requiere rol '{rol_minimo}'")
        if excede_limite(cliente):
            audit.registrar("rate_limit", cliente.nombre, recurso, "rest", origen=origen)
            raise HTTPException(429, "Limite de peticiones excedido", headers={"Retry-After": "60"})
        if rol_minimo == "admin":
            audit.registrar("lectura_administrativa", cliente.nombre, recurso, "rest", origen=origen)
        request.state.cliente = cliente
        return cliente
    return dependencia
