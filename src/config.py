import os
from dotenv import load_dotenv

load_dotenv()

# Seguridad interna (backend -> chatbot). Debe ser el MISMO valor en ambos contenedores.
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

# Modo de respuesta: "rag" (Bedrock + pgvector) o "eco" (solo devuelve el último mensaje, para probar el contrato)
CHATBOT_MODO = os.getenv("CHATBOT_MODO", "rag").lower()

# AWS Bedrock (en la EC2 las credenciales vienen del rol IAM; no poner access keys)
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
# En us-east-2 Nova se invoca vía perfil de inferencia: "us.amazon.nova-lite-v1:0"
LLM_MODEL = os.getenv("LLM_MODEL", "us.amazon.nova-lite-v1:0")
EMBEDDING_DIM = 1024

# Fuente de protocolos para index.py: ruta local o s3://bucket/clave.txt
PROTOCOLOS_S3_URI = os.getenv("PROTOCOLOS_S3_URI", "")

# Base de Datos PostgreSQL — Datos reales de ORION
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "3.18.117.177"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "orion_db"),
    "user": os.getenv("DB_USER", "orion_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "connect_timeout": 5,
}

# Parámetros de recuperación / generación
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
MAX_HISTORIAL = int(os.getenv("MAX_HISTORIAL", "20"))  # últimos N mensajes que se envían al LLM
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
