"""DAG de aprendizaje federado: hospitales simulados, FedAvg y comparacion con el modelo centralizado.

Tareas:
    check_dataset -> experimento_federado -> publicar_en_lake

El experimento reparte el dataset de ACV entre K hospitales (IID y non-IID por edad), entrena con
FedAvg, mide el efecto de la heterogeneidad y de la privacidad diferencial, registra todo en MLflow
y deja el modelo federado en el Model Registry como `stroke-predictor-federated`.
"""
from datetime import datetime
import os

from airflow.decorators import dag, task

from config import DATA_BUCKET, DATA_KEY, MINIO_ENDPOINT, MLFLOW_TRACKING_URI

EXPERIMENT_NAME = "stroke-federated"
FEDERATED_MODEL_NAME = "stroke-predictor-federated"
LAKE_BUCKET = os.getenv("LAKE_BUCKET_NAME", "datalake")
SEEDS = 5


def _s3():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


@dag(
    dag_id="federated_stroke_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["stroke", "mlops", "federated"],
)
def federated_stroke_model():

    @task
    def check_dataset():
        """Verifica que el CSV exista en MinIO antes de iniciar el experimento."""
        try:
            _s3().head_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
        except Exception:
            raise FileNotFoundError(
                f"El archivo '{DATA_KEY}' no esta en el bucket '{DATA_BUCKET}'. Subilo a MinIO (ver README)."
            )
        print("Dataset encontrado en MinIO.")

    @task
    def experimento_federado() -> dict:
        """Corre FedAvg (IID, non-IID, DP, barrido de epocas), lo registra en MLflow y devuelve el resumen."""
        import json
        import tempfile
        from io import BytesIO

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow
        import mlflow.sklearn
        import pandas as pd

        import federated_lib as fl

        df = pd.read_csv(BytesIO(_s3().get_object(Bucket=DATA_BUCKET, Key=DATA_KEY)["Body"].read()))
        resumen, historia, modelos, X_df, y = fl.correr_experimento(df, seeds=SEEDS)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="fedavg_hospitales") as run, tempfile.TemporaryDirectory() as tmp:
            cfg = fl.CONFIG
            mlflow.log_params({
                "hospitales": cfg["k"], "rondas": cfg["rondas"], "epocas_locales": cfg["epocas"],
                "lr": cfg["lr"], "l2": cfg["l2"], "dp_clip": cfg["clip"], "semillas": SEEDS,
                "features": ",".join(fl.FEATURES),
            })
            for conf, mets in resumen.items():
                if conf.startswith("_"):      # entradas auxiliares (p. ej. diferencias pareadas), no son configuraciones
                    continue
                for m in ("roc_auc", "pr_auc", "recall", "precision", "f1"):
                    mlflow.log_metric(f"{conf}.{m}", mets[m]["media"])
                    mlflow.log_metric(f"{conf}.{m}_std", mets[m]["desvio"])

            # Grafico 1: AUC de validacion por ronda (IID vs non-IID)
            fig, ax = plt.subplots(figsize=(7, 4))
            for nombre, h in historia.items():
                if nombre.startswith("_"):
                    ax.plot(range(1, len(h) + 1), h, ls="--", c="gray", label="centralizado (validación)")
                else:
                    ax.plot(range(1, len(h) + 1), h, label=nombre.replace("federado_", "").replace("_", "-"))
            ax.set_xlabel("ronda"); ax.set_ylabel("AUC (validación)"); ax.grid(alpha=.3); ax.legend()
            fig.tight_layout(); fig.savefig(f"{tmp}/auc_por_ronda.png", dpi=130); plt.close(fig)

            # Grafico 2: efecto de las epocas locales (client drift)
            es = sorted(cfg["sens_epocas"])
            pares = resumen["_pareadas"]       # AUC IID - AUC non-IID, por particion train/test
            fig, ax = plt.subplots(figsize=(7, 4))
            for e in es:
                ax.scatter([e] * len(pares[str(e)]), pares[str(e)], c="#C77D0A", alpha=.75, zorder=3)
            ax.scatter([], [], c="#C77D0A", label="cada partición")
            ax.plot(es, [sum(pares[str(e)]) / len(pares[str(e)]) for e in es], c="#12263A", marker="o",
                    label="promedio")
            ax.axhline(0, ls="--", c="gray")
            ax.set_xscale("log"); ax.set_xlabel("épocas locales por ronda (mismo cómputo total)")
            ax.set_ylabel("AUC IID − AUC non-IID")
            ax.set_title("Cuánto pierde el non-IID respecto del IID", fontsize=11)
            ax.grid(alpha=.3); ax.legend()
            fig.tight_layout(); fig.savefig(f"{tmp}/client_drift.png", dpi=130); plt.close(fig)

            with open(f"{tmp}/resultados.json", "w", encoding="utf-8") as f:
                json.dump(resumen, f, indent=2)
            mlflow.log_artifacts(tmp, artifact_path="federated")

            # Modelo federado (escenario non-IID, semilla 0) listo para servirse desde la API
            w, b, media, desvio, umbral = modelos["federado_non_iid"]
            pipeline = fl.construir_pipeline_sklearn(X_df, y, w, b, media, desvio)
            mlflow.log_param("threshold", umbral)       # la API lee el umbral de este parametro
            mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                     registered_model_name=FEDERATED_MODEL_NAME)

            central = resumen["centralizado"]["roc_auc"]["media"]
            non_iid = resumen["federado_non_iid"]["roc_auc"]["media"]
            print(f"AUC centralizado={central:.4f} | federado IID={resumen['federado_iid']['roc_auc']['media']:.4f} "
                  f"| federado non-IID={non_iid:.4f} | umbral={umbral:.4f}")
            return {"run_id": run.info.run_id, "resumen": resumen}

    @task
    def publicar_en_lake(salida: dict):
        """Guarda los resultados del experimento en la zona curated del Data Lake."""
        import json
        key = f"curated/federated/{salida['run_id']}/resultados.json"
        _s3().put_object(Bucket=LAKE_BUCKET, Key=key,
                         Body=json.dumps(salida["resumen"], indent=2).encode("utf-8"))
        print(f"Resultados publicados en s3://{LAKE_BUCKET}/{key}")

    check = check_dataset()
    salida = experimento_federado()
    check >> salida
    publicar_en_lake(salida)


federated_stroke_model()
