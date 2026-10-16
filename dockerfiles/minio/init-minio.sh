#!/bin/sh
# Inicializacion de MinIO: buckets y un usuario de MINIMO PRIVILEGIO para la API.
# El usuario raiz (MINIO_ACCESS_KEY) queda reservado para Airflow y MLflow; la API, el servidor
# gRPC y el worker usan API_S3_USER, que no puede borrar objetos ni escribir en el bucket de MLflow.
set -e

for i in $(seq 1 30); do
  mc alias set s3 http://s3:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_ACCESS_KEY" >/dev/null 2>&1 && break
  echo "esperando a MinIO ($i/30)..."; sleep 2
done

mc mb --ignore-existing "s3/$MLFLOW_BUCKET_NAME"
mc mb --ignore-existing "s3/$DATA_REPO_BUCKET_NAME"
mc mb --ignore-existing "s3/$LAKE_BUCKET_NAME"

cat > /tmp/api-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LeerModelosDeMLflow",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": ["arn:aws:s3:::$MLFLOW_BUCKET_NAME", "arn:aws:s3:::$MLFLOW_BUCKET_NAME/*"]
    },
    {
      "Sid": "LeerYEscribirEnElLake",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": ["arn:aws:s3:::$LAKE_BUCKET_NAME", "arn:aws:s3:::$LAKE_BUCKET_NAME/*"]
    }
  ]
}
EOF

mc admin policy create s3 api-policy /tmp/api-policy.json
mc admin user add s3 "$API_S3_USER" "$API_S3_PASSWORD"
mc admin policy attach s3 api-policy --user "$API_S3_USER"
echo "MinIO listo: buckets y usuario '$API_S3_USER' con politica api-policy."
