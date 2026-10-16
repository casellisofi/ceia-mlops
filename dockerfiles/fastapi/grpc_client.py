"""Cliente gRPC de prueba. Uso (dentro del contenedor):
    docker compose exec -e GRPC_API_KEY=<clave> grpc_server python grpc_client.py
    docker compose exec grpc_server python grpc_client.py                    # sin clave: UNAUTHENTICATED
    docker compose exec -e GRPC_API_KEY=<clave> grpc_server python grpc_client.py --relleno 40000
                                                                             # mensaje de 40 KB: RESOURCE_EXHAUSTED
"""
import argparse
import os
import sys

import grpc

import stroke_pb2
import stroke_pb2_grpc

TARGET = os.getenv("GRPC_TARGET", "localhost:50051")
API_KEY = os.getenv("GRPC_API_KEY", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relleno", type=int, default=0, help="caracteres extra en un campo, para probar el limite de tamano")
    args = ap.parse_args()

    with grpc.insecure_channel(TARGET) as channel:
        stub = stroke_pb2_grpc.StrokePredictorStub(channel)
        print("Health (publico):", stub.Health(stroke_pb2.HealthRequest()).status)

        paciente = stroke_pb2.PacienteRequest(
            age=67, hypertension=0, heart_disease=1, gender="Male" + "x" * args.relleno,
            ever_married="Yes", work_type="Private", residence_type="Urban",
            avg_glucose_level=228.69, bmi=36.6, smoking_status="formerly smoked",
        )
        metadata = (("x-api-key", API_KEY),) if API_KEY else ()
        try:
            r = stub.Predict(paciente, metadata=metadata)
        except grpc.RpcError as exc:
            print(f"Predict rechazado: {exc.code().name} - {exc.details()}")
            sys.exit(1)
        print(f"Predict: probabilidad_acv={r.probabilidad_acv} prediccion={r.prediccion} "
              f"riesgo={r.riesgo} threshold_usado={r.threshold_usado}")


if __name__ == "__main__":
    main()
