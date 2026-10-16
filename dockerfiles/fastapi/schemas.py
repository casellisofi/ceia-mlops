from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PacienteInput(BaseModel):
    """Datos clinicos del paciente para la prediccion de ACV.

    La validacion es estricta: rangos plausibles, categorias cerradas y sin campos extra.
    Un valor fuera de rango o desconocido se rechaza en lugar de llegar al modelo.
    """
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    gender: Literal["Male", "Female", "Other"]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(ge=20, le=600)
    bmi: float = Field(ge=10, le=100)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


class PredictionOutput(BaseModel):
    """Resultado de la prediccion: probabilidad, etiqueta binaria y nivel de riesgo."""
    probabilidad_acv: float
    prediccion: int
    riesgo: str
    threshold_usado: float
