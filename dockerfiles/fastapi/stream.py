"""Streaming con Valkey/Redis Streams: nombres, cliente y operaciones comunes.

    stroke:requests  eventos de entrada (pacientes a evaluar)
    stroke:results   predicciones producidas por el worker
    stroke:dlq       eventos invalidos (dead-letter queue), con el motivo del rechazo
"""
import json
import os

import redis

REQUESTS = "stroke:requests"
RESULTS = "stroke:results"
DLQ = "stroke:dlq"
GROUP = "predictores"
MAXLEN = 10_000   # los streams se recortan para no crecer sin limite

_r = None


def cliente() -> redis.Redis:
    global _r
    if _r is None:
        _r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/1"),
                                  decode_responses=True)
    return _r


def publicar(datos: dict, quien: str = "stream-directo") -> str:
    """Publica un evento de entrada y devuelve su id. `quien` identifica al cliente (trazabilidad)."""
    return cliente().xadd(REQUESTS, {"payload": json.dumps(datos), "cliente": quien},
                          maxlen=MAXLEN, approximate=True)


def ultimos(stream: str, count: int = 10) -> list[dict]:
    """Ultimos eventos de un stream (mas recientes primero), con el payload ya decodificado."""
    out = []
    for eid, campos in cliente().xrevrange(stream, count=max(1, min(int(count), 200))):
        item = {"id": eid, **{k: v for k, v in campos.items() if k != "payload"}}
        try:
            item["payload"] = json.loads(campos.get("payload", "null"))
        except json.JSONDecodeError:
            item["payload"] = campos.get("payload")
        out.append(item)
    return out
