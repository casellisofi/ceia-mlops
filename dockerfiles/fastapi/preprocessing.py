import numpy as np
import pandas as pd


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Replica las transformaciones del pipeline de entrenamiento."""
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 50, 70, 120],
        labels=["Joven", "Adulto", "Mayor", "Anciano"],
    ).astype(str)
    df["avg_glucose_level"] = np.clip(df["avg_glucose_level"], 50, 300)
    df["has_risk_factors"] = (
        (df["hypertension"] == 1) | (df["heart_disease"] == 1)
    ).astype(int)
    df["is_female"] = (df["gender"] == "Female").astype(int)
    df = df.drop(columns=["gender"])
    df["work_type"] = df["work_type"].replace({"Never_worked": "children"})
    return df
