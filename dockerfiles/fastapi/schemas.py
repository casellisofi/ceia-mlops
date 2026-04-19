from pydantic import BaseModel


class PacienteInput(BaseModel):
    """Datos clinicos del paciente para la prediccion de ACV."""
    age: float
    hypertension: int
    heart_disease: int
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class PredictionOutput(BaseModel):
    """Resultado de la prediccion: probabilidad, etiqueta binaria y nivel de riesgo."""
    probabilidad_acv: float
    prediccion: int
    riesgo: str
    threshold_usado: float
