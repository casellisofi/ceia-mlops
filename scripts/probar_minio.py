"""Prueba del minimo privilegio en MinIO: que puede y que NO puede hacer el usuario de la API.

Lee API_S3_USER y API_S3_PASSWORD de .env. Requiere boto3 (esta en el entorno del curso).

    python scripts/probar_minio.py
"""
import pathlib
import sys

import boto3
from botocore.exceptions import ClientError

ENDPOINT = "http://localhost:9000"


def leer_env():
    env = {}
    ruta = pathlib.Path(__file__).resolve().parent.parent / ".env"
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if "=" in linea and not linea.startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def cliente(usuario, clave):
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=usuario,
                        aws_secret_access_key=clave, region_name="us-east-1")


def main():
    env = leer_env()
    if not env.get("API_S3_PASSWORD"):
        sys.exit("Falta API_S3_PASSWORD en .env")
    api = cliente(env.get("API_S3_USER", "api_user"), env["API_S3_PASSWORD"])
    root = cliente(env.get("MINIO_ACCESS_KEY", "minio"), env.get("MINIO_SECRET_ACCESS_KEY", "minio123"))
    fallos = []

    def caso(nombre, esperado, accion):
        try:
            accion()
            obtenido = "PERMITIDO"
        except ClientError as e:
            obtenido = f"DENEGADO ({e.response['Error']['Code']})"
        ok = obtenido.startswith(esperado)
        if not ok:
            fallos.append(nombre)
        print(f"  {'OK   ' if ok else 'FALLA'} {nombre:<52s} esperado {esperado:<10s} obtenido {obtenido}")

    print("Usuario de la API (api_user), no es el usuario raiz:")
    caso("leer los modelos del bucket mlflow", "PERMITIDO",
         lambda: api.list_objects_v2(Bucket="mlflow", MaxKeys=1))
    caso("escribir en el bucket mlflow (alterar un modelo)", "DENEGADO",
         lambda: api.put_object(Bucket="mlflow", Key="x.txt", Body=b"x"))
    caso("leer el dataset original (bucket data)", "DENEGADO",
         lambda: api.get_object(Bucket="data", Key="stroke-data.csv"))
    caso("escribir un registro de auditoria", "PERMITIDO",
         lambda: api.put_object(Bucket="datalake", Key="audit/prueba-permisos.json", Body=b"{}"))
    caso("BORRAR un registro de auditoria", "DENEGADO",
         lambda: api.delete_object(Bucket="datalake", Key="audit/prueba-permisos.json"))

    visibles = sorted(b["Name"] for b in api.list_buckets()["Buckets"])
    todos = sorted(b["Name"] for b in root.list_buckets()["Buckets"])
    ok = "data" in todos and "data" not in visibles
    if not ok:
        fallos.append("visibilidad de buckets")
    print(f"  {'OK   ' if ok else 'FALLA'} buckets que existen: {todos} | los que ve api_user: {visibles}")

    # limpieza: el objeto de prueba solo lo puede borrar el usuario raiz
    root.delete_object(Bucket="datalake", Key="audit/prueba-permisos.json")
    print()
    if fallos:
        print(f"{len(fallos)} control(es) no se comportaron como se esperaba: {fallos}")
        sys.exit(1)
    print("Todos los permisos se comportan como se esperaba.")


if __name__ == "__main__":
    main()
