FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    mlflow==2.19.0 \
    "sqlalchemy<2.1" \
    boto3 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4  \
    imbalanced-learn==0.12.3 \
    pydantic==2.9.2 \
    strawberry-graphql[fastapi]==0.259.1 \
    grpcio==1.69.0 \
    grpcio-tools==1.69.0 \
    redis==5.0.8

COPY app.py schemas.py preprocessing.py model.py service.py lake.py graphql_schema.py \
     grpc_server.py grpc_client.py stroke.proto \
     stream.py stream_worker.py stream_producer.py security.py audit.py ./

# Genera los stubs gRPC (stroke_pb2.py y stroke_pb2_grpc.py) a partir del .proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. stroke.proto

EXPOSE 8800 50051
