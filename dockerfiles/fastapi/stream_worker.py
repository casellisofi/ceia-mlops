"""Worker de streaming: consume eventos, predice y publica el resultado.

Usa un consumer group (se puede escalar con mas replicas) y entrega "al menos una vez":
un evento solo se confirma (XACK) despues de procesarlo; si el worker cae a mitad de camino,
al reiniciar retoma sus eventos pendientes.
"""
import json
import socket
import time

import redis
from pydantic import ValidationError

import lake
import stream
from schemas import PacienteInput
from service import ModeloNoDisponible, inicializar, predecir

CONSUMER = socket.gethostname()


def _enviar_a_dlq(r, eid, crudo, motivo):
    r.xadd(stream.DLQ, {"request_id": eid, "payload": crudo, "error": str(motivo)[:500]},
           maxlen=stream.MAXLEN, approximate=True)
    r.xack(stream.REQUESTS, stream.GROUP, eid)
    print(f"[dlq] {eid}: {str(motivo)[:120]}", flush=True)


def procesar(r, eid, campos):
    crudo = campos.get("payload", "")
    try:
        datos = json.loads(crudo)
        PacienteInput(**datos)                       # misma validacion que REST/GraphQL/gRPC
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        _enviar_a_dlq(r, eid, crudo, exc)
        return

    lake.registrar_evento_raw(eid, datos)            # el evento aterriza tal cual en raw/
    try:
        resultado = predecir(datos, protocolo="streaming", cliente=campos.get("cliente", "stream-directo"))
    except ModeloNoDisponible:
        print(f"[espera] {eid}: modelo no disponible, queda pendiente", flush=True)
        time.sleep(5)
        return                                       # sin XACK: se reintenta

    r.xadd(stream.RESULTS, {"request_id": eid, "payload": json.dumps(resultado)},
           maxlen=stream.MAXLEN, approximate=True)
    r.xack(stream.REQUESTS, stream.GROUP, eid)
    print(f"[ok] {eid}: {resultado['riesgo']} ({resultado['probabilidad_acv']})", flush=True)


def main():
    inicializar()
    r = stream.cliente()
    try:
        r.xgroup_create(stream.REQUESTS, stream.GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    print(f"Worker '{CONSUMER}' escuchando {stream.REQUESTS} (grupo {stream.GROUP})", flush=True)

    leyendo_pendientes = True    # primero se retoman los eventos propios sin confirmar
    while True:
        try:
            pos = "0" if leyendo_pendientes else ">"
            resp = r.xreadgroup(stream.GROUP, CONSUMER, {stream.REQUESTS: pos}, count=10, block=5000)
        except redis.ConnectionError as exc:
            print(f"[redis] conexion perdida: {exc}; reintentando", flush=True)
            time.sleep(3)
            continue
        mensajes = resp[0][1] if resp else []
        if leyendo_pendientes and not mensajes:
            leyendo_pendientes = False
            continue
        for eid, campos in mensajes:
            procesar(r, eid, campos)


if __name__ == "__main__":
    main()
