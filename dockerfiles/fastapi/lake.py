"""Data Lake en MinIO/S3: zonas, registro de predicciones y metadatos del modelo.

Estructura del bucket:
    raw/events/date=YYYY-MM-DD/             eventos de streaming tal como llegaron
    curated/predictions/date=YYYY-MM-DD/    una predicción por objeto JSON (trazabilidad)
    models/v<N>/metadata.json               qué versión del modelo se está sirviendo
    audit/date=YYYY-MM-DD/                  eventos de seguridad (quién, qué, cuándo; sin datos clínicos)
"""
import json
import os
import threading
import uuid
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

BUCKET = os.getenv("LAKE_BUCKET_NAME", "datalake")
ZONAS = ("raw", "curated", "models", "audit")
PREFIJO_PREDICCIONES = "curated/predictions"

_s3 = None
_listo = False
_lock = threading.Lock()


def _cliente():
    """Cliente S3 perezoso; el endpoint sale de AWS_ENDPOINT_URL_S3 (definido en el compose)."""
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}),
        )
    return _s3


def asegurar_lake():
    """Crea el bucket y las zonas si no existen (idempotente)."""
    global _listo
    with _lock:
        if _listo:
            return
        s3 = _cliente()
        try:
            s3.head_bucket(Bucket=BUCKET)          # no requiere permiso de listar todos los buckets
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                raise
            s3.create_bucket(Bucket=BUCKET)
        for zona in ZONAS:
            s3.put_object(Bucket=BUCKET, Key=f"{zona}/.keep", Body=b"")
        _listo = True


def registrar_modelo(info: dict, threshold: float):
    """Guarda en models/v<N>/metadata.json qué modelo se está sirviendo."""
    asegurar_lake()
    meta = {**info, "threshold": threshold,
            "cargado_en": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _cliente().put_object(Bucket=BUCKET, Key=f"models/v{info['version']}/metadata.json",
                          Body=json.dumps(meta, indent=2).encode("utf-8"))


def registrar_prediccion(entrada: dict, resultado: dict, protocolo: str, modelo: dict | None,
                         cliente: str = "anonimo"):
    """Guarda una predicción en la zona curated. Nunca propaga errores: el lake no debe tumbar la API."""
    try:
        asegurar_lake()
        ahora = datetime.now(timezone.utc)
        pid = f"{ahora:%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}"   # ordenable cronológicamente
        registro = {
            "id": pid,
            "timestamp": ahora.isoformat(timespec="milliseconds"),
            "protocolo": protocolo,
            "cliente": cliente,
            "modelo_version": (modelo or {}).get("version", "desconocida"),
            "entrada": entrada,
            "resultado": resultado,
        }
        _cliente().put_object(
            Bucket=BUCKET,
            Key=f"{PREFIJO_PREDICCIONES}/date={ahora:%Y-%m-%d}/{pid}.json",
            Body=json.dumps(registro).encode("utf-8"),
            ContentType="application/json",
        )
        return pid
    except Exception as exc:
        print(f"[lake] no se pudo registrar la prediccion: {exc}", flush=True)
        return None


def registrar_evento_raw(evento_id: str, datos: dict):
    """Aterriza un evento de streaming en la zona raw, sin transformar. No propaga errores."""
    try:
        asegurar_lake()
        ahora = datetime.now(timezone.utc)
        _cliente().put_object(
            Bucket=BUCKET,
            Key=f"raw/events/date={ahora:%Y-%m-%d}/{evento_id}.json",
            Body=json.dumps({"id": evento_id, "recibido": ahora.isoformat(timespec="milliseconds"),
                             "datos": datos}).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        print(f"[lake] no se pudo aterrizar el evento {evento_id}: {exc}", flush=True)


def listar_predicciones(limite: int = 10) -> list[dict]:
    """Devuelve las últimas predicciones (más recientes primero)."""
    limite = max(1, min(int(limite), 200))
    asegurar_lake()
    s3 = _cliente()

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIJO_PREDICCIONES}/", Delimiter="/")
    fechas = sorted((p["Prefix"] for p in resp.get("CommonPrefixes", [])), reverse=True)

    claves = []
    for prefijo in fechas:                      # de la fecha más nueva hacia atrás
        del_dia = []
        for pagina in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefijo):
            del_dia += [o["Key"] for o in pagina.get("Contents", [])]
        claves += sorted(del_dia, reverse=True)
        if len(claves) >= limite:
            break

    registros = []
    for clave in claves[:limite]:
        cuerpo = s3.get_object(Bucket=BUCKET, Key=clave)["Body"].read()
        registros.append(json.loads(cuerpo))
    return registros
