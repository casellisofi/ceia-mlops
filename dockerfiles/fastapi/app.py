import pandas as pd
from fastapi import FastAPI, HTTPException

from model import THRESHOLD, load_model, get_model
from preprocessing import apply_feature_engineering
from schemas import PacienteInput, PredictionOutput

app = FastAPI(title="Stroke Prediction API")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def root():
    """Healthcheck: verifica que la API este en linea."""
    return {"status": "ok", "model": "stroke-predictor"}


@app.post("/predict", response_model=PredictionOutput)
def predict(paciente: PacienteInput):
    """Recibe datos clinicos, aplica feature engineering y retorna la prediccion de ACV."""
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    df = pd.DataFrame([paciente.dict()])
    df = apply_feature_engineering(df)

    proba = float(model.predict_proba(df)[:, 1][0])
    prediccion = int(proba >= THRESHOLD)

    if proba < 0.3:
        riesgo = "Bajo"
    elif proba < 0.6:
        riesgo = "Medio"
    else:
        riesgo = "Alto"

    return PredictionOutput(
        probabilidad_acv=round(proba, 4),
        prediccion=prediccion,
        riesgo=riesgo,
        threshold_usado=THRESHOLD,
    )
