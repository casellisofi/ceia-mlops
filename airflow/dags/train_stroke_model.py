from airflow.decorators import dag, task
from datetime import datetime
import os

from config import (
    MLFLOW_TRACKING_URI,
    MINIO_ENDPOINT,
    DATA_BUCKET,
    DATA_KEY,
    RANDOM_STATE,
    MODEL_NAME,
    EXPERIMENT_NAME,
)


@dag(
    dag_id="train_stroke_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["stroke", "mlops"],
)
def train_stroke_model():

    @task
    def check_dataset():
        """Verifica que el CSV exista en MinIO antes de iniciar el pipeline."""
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        try:
            s3.head_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
            print("Dataset encontrado en MinIO.")
        except Exception:
            raise FileNotFoundError(
                f"El archivo '{DATA_KEY}' no esta en el bucket '{DATA_BUCKET}'. "
                "Subilo a MinIO antes de ejecutar el DAG (ver README)."
            )

    @task.virtualenv(
        requirements=["boto3", "pandas==2.2.2"],
        system_site_packages=False,
    )
    def load_data():
        """Descarga el CSV desde MinIO y lo retorna como JSON."""
        import boto3
        import pandas as pd
        from io import StringIO
        import os

        s3 = boto3.client(
            "s3",
            endpoint_url="http://s3:9000",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        obj = s3.get_object(Bucket="data", Key="stroke-data.csv")
        df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
        return df.to_json()

    @task.virtualenv(
        requirements=["pandas==2.2.2", "numpy==1.26.4"],
        system_site_packages=False,
    )
    def feature_engineering(df_json: str):
        """Genera variables derivadas y limpia el dataset."""
        import pandas as pd
        import numpy as np

        df = pd.read_json(df_json)
        df = df.drop(columns=["id"])
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 50, 70, 120],
            labels=["Joven", "Adulto", "Mayor", "Anciano"],
        ).astype(str)
        df["avg_glucose_level"] = np.clip(df["avg_glucose_level"], 50, 300)
        df["has_risk_factors"] = np.where(
            (df["hypertension"] == 1) | (df["heart_disease"] == 1), 1, 0
        )
        df["is_female"] = np.where(df["gender"] == "Female", 1, 0)
        df = df.drop(columns=["gender"])
        df["work_type"] = df["work_type"].replace({"Never_worked": "children"})
        return df.to_json()

    @task.virtualenv(
        requirements=[
            "pandas==2.2.2",
            "numpy==1.26.4",
            "scikit-learn==1.4.2",
            "imbalanced-learn==0.12.3",
            "mlflow==2.19.0",
            "boto3",
        ],
        system_site_packages=True,
    )
    def train_and_register(df_json: str):
        """Entrena Regresion Logistica con SMOTE y registra el modelo en MLflow."""
        import os
        import pandas as pd
        import mlflow
        import mlflow.sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score, recall_score, precision_score,
            f1_score, precision_recall_curve,
        )
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        MLFLOW_URI = "http://mlflow:5000"
        RANDOM_STATE = 42

        df = pd.read_json(df_json)
        X = df.drop("stroke", axis=1)
        y = df["stroke"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        bmi_median = X_train["bmi"].median()
        X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

        cols_drop = ["heart_disease", "bmi", "smoking_status", "age_group"]
        num_features = (
            X.select_dtypes(exclude="object")
            .columns.drop(cols_drop, errors="ignore")
            .tolist()
        )

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), []),
        ])

        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        thr_candidates = thresholds[recall[:-1] >= 0.90]
        best_threshold = float(thr_candidates.max()) if len(thr_candidates) > 0 else 0.5
        y_pred = (y_proba >= best_threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("stroke-prediction")

        with mlflow.start_run(run_name="logistic_regression_smote"):
            mlflow.log_params({
                "C": 0.1,
                "penalty": "l1",
                "solver": "liblinear",
                "threshold": best_threshold,
                "bmi_median": bmi_median,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name="stroke-predictor",
            )
            print(
                f"AUC: {metrics['roc_auc']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"Threshold: {best_threshold:.5f}"
            )

    df_raw = load_data()
    df_features = feature_engineering(df_raw)
    check_dataset() >> df_raw
    train_and_register(df_features)


train_stroke_model()
