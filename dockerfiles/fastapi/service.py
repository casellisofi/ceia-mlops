"""Logica de prediccion compartida por los tres protocolos (REST, GraphQL y gRPC)."""
import pandas as pd

import lake
from model import get_model, get_model_info, get_threshold, load_model
from preprocessing import apply_feature_engineering


class ModeloNoDisponible(RuntimeError):
    """El modelo todavia no esta registrado en MLflow o no pudo cargarse."""


def inicializar():
    """Carga el modelo y prepara el Data Lake (se llama al iniciar REST/GraphQL y gRPC)."""
    load_model()
    try:
        lake.asegurar_lake()
        info = get_model_info()
        if info:
            lake.registrar_modelo(info, get_threshold())
    except Exception as exc:
        print(f"[lake] no disponible al iniciar: {exc}", flush=True)


def nivel_riesgo(proba: float) -> str:
    if proba < 0.3:
        return "Bajo"
    if proba < 0.6:
        return "Medio"
    return "Alto"


def predecir(datos: dict, protocolo: str = "rest", cliente: str = "anonimo") -> dict:
    """Aplica el feature engineering y devuelve probabilidad, etiqueta y nivel de riesgo.

    Cada prediccion queda registrada en el Data Lake (zona curated) con el protocolo
    de origen y la version del modelo que la produjo.
    """
    model = get_model()
    if model is None:
        load_model()  # reintenta: el modelo puede haberse registrado despues de iniciar el servicio
        model = get_model()
    if model is None:
        raise ModeloNoDisponible("Modelo no disponible")

    df = apply_feature_engineering(pd.DataFrame([datos]))
    proba = float(model.predict_proba(df)[:, 1][0])
    umbral = get_threshold()
    resultado = {
        "probabilidad_acv": round(proba, 4),
        "prediccion": int(proba >= umbral),
        "riesgo": nivel_riesgo(proba),
        "threshold_usado": umbral,
    }
    lake.registrar_prediccion(datos, resultado, protocolo, get_model_info(), cliente)
    return resultado
