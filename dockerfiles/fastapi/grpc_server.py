"""Servidor gRPC para la prediccion de ACV (puerto 50051)."""
import os
from concurrent import futures

import grpc
from pydantic import ValidationError

import audit
import security
import stroke_pb2
import stroke_pb2_grpc
from schemas import PacienteInput
from service import ModeloNoDisponible, inicializar, predecir

PORT = os.getenv("GRPC_PORT", "50051")


def _autorizar(context, metodo):
    """Exige la API key en el metadata `x-api-key`. Aborta con el codigo gRPC que corresponda."""
    origen = context.peer()
    clave = dict(context.invocation_metadata()).get("x-api-key")
    cliente = security.autenticar(clave)
    if cliente is None:
        audit.registrar("autenticacion_fallida", "anonimo", metodo, "grpc", origen=origen)
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "API key ausente o invalida")
    if security.excede_limite(cliente):
        audit.registrar("rate_limit", cliente.nombre, metodo, "grpc", origen=origen)
        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Limite de peticiones excedido")
    return cliente


class StrokePredictorServicer(stroke_pb2_grpc.StrokePredictorServicer):
    def Predict(self, request, context):
        cliente = _autorizar(context, "/stroke.StrokePredictor/Predict")
        datos = {
            "age": request.age,
            "hypertension": request.hypertension,
            "heart_disease": request.heart_disease,
            "gender": request.gender,
            "ever_married": request.ever_married,
            "work_type": request.work_type,
            "Residence_type": request.residence_type,  # nombre que espera el modelo
            "avg_glucose_level": request.avg_glucose_level,
            "bmi": request.bmi,
            "smoking_status": request.smoking_status,
        }
        try:
            PacienteInput(**datos)  # misma validacion que REST
            resultado = predecir(datos, protocolo="grpc", cliente=cliente.nombre)
        except ValidationError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except ModeloNoDisponible:
            context.abort(grpc.StatusCode.UNAVAILABLE, "Modelo no disponible")
        return stroke_pb2.PredictionResponse(**resultado)

    def Health(self, request, context):
        return stroke_pb2.HealthResponse(status="ok")


def serve():
    inicializar()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[("grpc.max_receive_message_length", 16 * 1024)])   # un paciente pesa < 1 KB
    stroke_pb2_grpc.add_StrokePredictorServicer_to_server(StrokePredictorServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"Servidor gRPC escuchando en el puerto {PORT}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
