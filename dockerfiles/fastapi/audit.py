"""Registro de auditoria de seguridad en el Data Lake (zona audit/).

Guarda QUIEN hizo QUE, CUANDO y con que resultado. Nunca guarda datos clinicos ni claves.
Cada evento es un objeto nuevo (no se sobreescribe nada). Para que un atacante no pueda inflar el
almacenamiento enviando claves invalidas, se escriben como maximo MAX_POR_MIN objetos por tipo de
evento y minuto; el resto solo queda en el log del contenedor.
"""
import json
import time
import uuid
from datetime import datetime, timezone

import lake
import stream

MAX_POR_MIN = 60


def registrar(evento: str, cliente: str, recurso: str, protocolo: str,
              origen: str | None = None, detalle: str | None = None):
    """Registra un evento de seguridad. Nunca propaga errores."""
    ahora = datetime.now(timezone.utc)
    linea = f"[audit] {evento} cliente={cliente} recurso={recurso} protocolo={protocolo}"
    try:
        n = stream.cliente().incr(f"auditcap:{evento}:{int(time.time() // 60)}")
        if n == 1:
            stream.cliente().expire(f"auditcap:{evento}:{int(time.time() // 60)}", 70)
        if n > MAX_POR_MIN:
            print(linea + " (sin objeto: tope por minuto)", flush=True)
            return
    except Exception:
        pass                                     # sin Valkey se audita igual, sin tope
    print(linea, flush=True)
    try:
        lake.asegurar_lake()
        reg = {"id": f"{ahora:%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}",
               "timestamp": ahora.isoformat(timespec="milliseconds"),
               "evento": evento, "cliente": cliente, "recurso": recurso,
               "protocolo": protocolo, "origen": origen, "detalle": detalle}
        lake._cliente().put_object(
            Bucket=lake.BUCKET, Key=f"audit/date={ahora:%Y-%m-%d}/{reg['id']}.json",
            Body=json.dumps(reg).encode("utf-8"), ContentType="application/json")
    except Exception as exc:
        print(f"[audit] no se pudo escribir en el lake: {exc}", flush=True)


def listar(limite: int = 20) -> list[dict]:
    """Ultimos eventos de auditoria (mas recientes primero)."""
    limite = max(1, min(int(limite), 200))
    s3 = lake._cliente()
    fechas = sorted((p["Prefix"] for p in s3.list_objects_v2(
        Bucket=lake.BUCKET, Prefix="audit/", Delimiter="/").get("CommonPrefixes", [])), reverse=True)
    claves = []
    for prefijo in fechas:
        dia = []
        for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=lake.BUCKET, Prefix=prefijo):
            dia += [o["Key"] for o in pag.get("Contents", [])]
        claves += sorted(dia, reverse=True)
        if len(claves) >= limite:
            break
    return [json.loads(s3.get_object(Bucket=lake.BUCKET, Key=k)["Body"].read()) for k in claves[:limite]]
