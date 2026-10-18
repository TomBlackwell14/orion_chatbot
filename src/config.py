import os
from dotenv import load_dotenv

load_dotenv()

# Seguridad interna (backend -> chatbot). Debe ser el MISMO valor que CHATBOT_INTERNAL_TOKEN en NestJS.
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

# Modo de respuesta: "rag" (Bedrock + índice local) o "eco" (solo devuelve el mensaje, para probar el contrato)
CHATBOT_MODO = os.getenv("CHATBOT_MODO", "rag").lower()

# AWS Bedrock (en la EC2 las credenciales vienen del rol IAM; no poner access keys)
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
# En us-east-2 Nova se invoca vía perfil de inferencia: "us.amazon.nova-lite-v1:0"
LLM_MODEL = os.getenv("LLM_MODEL", "us.amazon.nova-lite-v1:0")
EMBEDDING_DIM = 1024

# Protocolos: archivo dentro de la imagen (docs/). El índice se arma en memoria al primer uso.
PROTOCOLOS_PATH = os.getenv("PROTOCOLOS_PATH", "")

# Parámetros de recuperación / generación
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
MAX_HISTORIAL = int(os.getenv("MAX_HISTORIAL", "20"))  # últimos N mensajes que se envían al LLM
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Contactos de la sede que el bot puede entregar (validar con la institución antes de producción)
CONTACTOS_SEDE = os.getenv(
    "CONTACTOS_SEDE",
    "Seguridad: +56 32 227 7999 (anexo 999) · Salud (policlínico sede): +56 32 227 7888",
)
