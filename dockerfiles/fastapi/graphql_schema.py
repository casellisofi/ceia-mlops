"""Esquema GraphQL (Strawberry) del servicio de prediccion de ACV."""
import strawberry
from strawberry.extensions import QueryDepthLimiter
from strawberry.permission import BasePermission
from strawberry.scalars import JSON

import audit
import lake
from schemas import PacienteInput
from service import predecir


class SoloAdmin(BasePermission):
    """Autorizacion por campo: el historial contiene datos de salud, solo lo ve un admin."""
    message = "Se requiere rol admin"

    def has_permission(self, source, info, **kwargs) -> bool:
        cliente = (info.context or {}).get("cliente")
        if cliente is not None and cliente.rol == "admin":
            audit.registrar("lectura_administrativa", cliente.nombre, f"graphql:{info.field_name}", "graphql")
            return True
        audit.registrar("acceso_denegado", getattr(cliente, "nombre", "anonimo"),
                        f"graphql:{info.field_name}", "graphql", detalle="rol insuficiente")
        return False


@strawberry.input
class PacienteGQL:
    """Datos clinicos del paciente."""
    age: float
    hypertension: int
    heart_disease: int
    gender: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


@strawberry.type
class Prediccion:
    """Resultado de la prediccion de ACV."""
    probabilidad_acv: float
    prediccion: int
    riesgo: str
    threshold_usado: float


@strawberry.type
class RegistroPrediccion:
    """Una prediccion guardada en el Data Lake, con su trazabilidad."""
    id: str
    timestamp: str
    protocolo: str
    modelo_version: str
    entrada: JSON
    resultado: Prediccion


@strawberry.type
class Query:
    @strawberry.field
    def estado(self) -> str:
        """Healthcheck del servicio GraphQL."""
        return "ok"

    @strawberry.field
    def predecir(self, info: strawberry.Info, paciente: PacienteGQL) -> Prediccion:
        """Predice el riesgo de ACV; pide solo los campos que necesites del resultado."""
        datos = strawberry.asdict(paciente)
        datos["Residence_type"] = datos.pop("residence_type")  # nombre que espera el modelo
        PacienteInput(**datos)  # misma validacion que REST
        cliente = (info.context or {}).get("cliente")
        return Prediccion(**predecir(datos, protocolo="graphql", cliente=getattr(cliente, "nombre", "anonimo")))

    @strawberry.field(permission_classes=[SoloAdmin])
    def historial(self, limite: int = 10) -> list[RegistroPrediccion]:
        """Ultimas predicciones guardadas en el Data Lake (mas recientes primero)."""
        return [
            RegistroPrediccion(
                id=r["id"], timestamp=r["timestamp"], protocolo=r["protocolo"],
                modelo_version=r["modelo_version"], entrada=r["entrada"],
                resultado=Prediccion(**r["resultado"]),
            )
            for r in lake.listar_predicciones(limite)
        ]


schema = strawberry.Schema(query=Query, extensions=[QueryDepthLimiter(max_depth=6)])
