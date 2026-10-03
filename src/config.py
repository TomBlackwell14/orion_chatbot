import os
from dotenv import load_dotenv

load_dotenv()

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "amazon.nova-lite-v1:0"

# Base de Datos PostgreSQL — Datos reales de ORION
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "3.18.117.177"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "orion_db"),
    "user": os.getenv("DB_USER", "orion_user"),
    "password": os.getenv("DB_PASSWORD", "")
}

# Parámetros de procesamiento
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 2