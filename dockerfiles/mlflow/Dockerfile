FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    "sqlalchemy<2.1" \
    boto3 \
    psycopg2-binary

EXPOSE 5000