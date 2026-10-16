"""Productor de prueba: publica pacientes aleatorios en el stream de entrada.

Uso (dentro del contenedor):
    docker compose exec fastapi python stream_producer.py -n 20
    docker compose exec fastapi python stream_producer.py -n 5 --invalidos 2
"""
import argparse
import random

import stream

GENEROS = ["Male", "Female"]
TRABAJOS = ["Private", "Self-employed", "Govt_job", "children"]
FUMA = ["formerly smoked", "never smoked", "smokes", "Unknown"]


def paciente_aleatorio(rng: random.Random) -> dict:
    return {
        "age": round(rng.uniform(18, 90), 1),
        "hypertension": rng.choice([0, 0, 0, 1]),
        "heart_disease": rng.choice([0, 0, 0, 1]),
        "gender": rng.choice(GENEROS),
        "ever_married": rng.choice(["Yes", "No"]),
        "work_type": rng.choice(TRABAJOS),
        "Residence_type": rng.choice(["Urban", "Rural"]),
        "avg_glucose_level": round(rng.uniform(60, 280), 2),
        "bmi": round(rng.uniform(16, 45), 1),
        "smoking_status": rng.choice(FUMA),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="cantidad de eventos validos")
    ap.add_argument("--invalidos", type=int, default=0, help="eventos invalidos (iran a la DLQ)")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.semilla)
    for _ in range(args.n):
        stream.publicar(paciente_aleatorio(rng))
    for _ in range(args.invalidos):
        malo = paciente_aleatorio(rng)
        malo["age"] = "no-es-un-numero"
        stream.publicar(malo)
    print(f"Publicados {args.n} eventos validos y {args.invalidos} invalidos en {stream.REQUESTS}")


if __name__ == "__main__":
    main()
