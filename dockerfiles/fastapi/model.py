"""Carga y acceso al modelo de prediccion de ACV registrado en MLflow."""
import os
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Modelo a servir: "stroke-predictor" (centralizado) o "stroke-predictor-federated"
MODEL_NAME = os.getenv("MODEL_NAME", "stroke-predictor")
# Umbral de respaldo; normalmente se usa el que el entrenamiento registro en MLflow
DEFAULT_THRESHOLD = float(os.getenv("STROKE_THRESHOLD", "0.17847"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None
_info = None
_threshold = DEFAULT_THRESHOLD


def load_model():
    """Carga la version mas reciente del modelo desde MLflow al iniciar la API."""
    global _model, _info, _threshold
    try:
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return
    try:
        versiones = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
        ultima = max(versiones, key=lambda v: int(v.version))
        _info = {"name": MODEL_NAME, "version": str(ultima.version), "run_id": ultima.run_id}
        _threshold = float(mlflow.get_run(ultima.run_id).data.params["threshold"])
    except Exception as e:
        print(f"No se pudo leer la version del modelo: {e}")
        _info = {"name": MODEL_NAME, "version": "desconocida", "run_id": None}
        _threshold = DEFAULT_THRESHOLD


def get_model():
    """Retorna el modelo cargado, o None si aun no fue inicializado."""
    return _model


def get_threshold() -> float:
    """Umbral de decision del modelo cargado (el que registro su entrenamiento en MLflow)."""
    return _threshold


def get_model_info():
    """Retorna {name, version, run_id} del modelo cargado, o None."""
    return _info
