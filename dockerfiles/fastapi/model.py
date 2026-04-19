"""Carga y acceso al modelo de prediccion de ACV registrado en MLflow."""
import os
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "stroke-predictor"
THRESHOLD = float(os.getenv("STROKE_THRESHOLD", "0.17847"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None


def load_model():
    """Carga la version mas reciente del modelo desde MLflow al iniciar la API."""
    global _model
    try:
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando modelo: {e}")


def get_model():
    """Retorna el modelo cargado, o None si aun no fue inicializado."""
    return _model
