from fastapi import Depends, FastAPI, HTTPException, Request
from strawberry.fastapi import GraphQLRouter

import audit
import lake
import stream
from graphql_schema import schema
from schemas import PacienteInput, PredictionOutput
from security import Cliente, requiere
from service import ModeloNoDisponible, inicializar, predecir

app = FastAPI(title="Stroke Prediction API")


def graphql_auth(request: Request):
    """Autentica las consultas GraphQL. Solo la pagina del playground (GET con Accept: text/html)
    se sirve sin clave; ejecutar cualquier consulta exige API key (cabecera x-api-key)."""
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        return None
    return requiere("predictor")(request)


async def graphql_context(cliente=Depends(graphql_auth)):
    return {"cliente": cliente}


# GraphQL en /graphql (con GraphiQL como playground al abrirlo en el navegador)
app.include_router(GraphQLRouter(schema, context_getter=graphql_context), prefix="/graphql")


@app.on_event("startup")
def startup():
    inicializar()


@app.get("/")
def root():
    """Healthcheck: verifica que la API este en linea (sin autenticacion, para los healthchecks)."""
    return {"status": "ok", "model": "stroke-predictor"}


@app.post("/predict", response_model=PredictionOutput)
def predict(paciente: PacienteInput, cliente: Cliente = Depends(requiere("predictor"))):
    """Recibe datos clinicos, aplica feature engineering y retorna la prediccion de ACV."""
    try:
        return PredictionOutput(**predecir(paciente.dict(), protocolo="rest", cliente=cliente.nombre))
    except ModeloNoDisponible:
        raise HTTPException(status_code=503, detail="Modelo no disponible")


@app.post("/stream/predict", status_code=202)
def stream_predict(paciente: PacienteInput, cliente: Cliente = Depends(requiere("predictor"))):
    """Publica el paciente en el stream de entrada; un worker lo procesa de forma asincrona."""
    try:
        return {"event_id": stream.publicar(paciente.dict(), cliente.nombre), "stream": stream.REQUESTS}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Streaming no disponible: {exc}")


@app.get("/stream/results")
def stream_results(count: int = 10, _: Cliente = Depends(requiere("admin"))):
    """Ultimas predicciones producidas por el worker de streaming (solo admin: contienen datos de salud)."""
    try:
        return stream.ultimos(stream.RESULTS, count)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Streaming no disponible: {exc}")


@app.get("/stream/dlq")
def stream_dlq(count: int = 10, _: Cliente = Depends(requiere("admin"))):
    """Eventos rechazados por invalidos (dead-letter queue), con el motivo (solo admin)."""
    try:
        return stream.ultimos(stream.DLQ, count)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Streaming no disponible: {exc}")


@app.get("/predictions")
def predictions(limit: int = 10, _: Cliente = Depends(requiere("admin"))):
    """Historial: ultimas predicciones guardadas en el Data Lake (solo admin: incluye datos clinicos)."""
    try:
        return lake.listar_predicciones(limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data Lake no disponible: {exc}")


@app.get("/audit")
def auditoria(limit: int = 20, _: Cliente = Depends(requiere("admin"))):
    """Eventos de seguridad registrados (autenticaciones fallidas, accesos denegados, rate limit, lecturas admin)."""
    try:
        return audit.listar(limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data Lake no disponible: {exc}")
