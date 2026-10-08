"""Lógica RAG de ORION: embeddings (Titan) + búsqueda en pgvector + respuesta con Bedrock (Converse).

Sin estado: todo lo que necesita llega en el contexto y el historial de cada petición.
"""
import json
import re

import boto3
import psycopg2
from botocore.config import Config

from config import (
    AWS_REGION, DB_CONFIG, EMBEDDING_MODEL, LLM_MODEL,
    MAX_HISTORIAL, MAX_TOKENS, TEMPERATURE, TOP_K_RESULTS,
)

# El backend espera como máximo 25 s; dejamos margen
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(connect_timeout=3, read_timeout=20, retries={"max_attempts": 1}),
)

# Enum cerrado acordado con backend/app
ACCIONES = {
    "NINGUNA": "respuesta informativa",
    "PEDIR_MAS_INFO": "necesitas más datos del usuario",
    "EVACUAR": "el usuario debe salir del lugar",
    "LLAMAR_EMERGENCIAS": "se requiere servicio externo (131 SAMU, 132 Bomberos, 133 Carabineros)",
    "PRIMEROS_AUXILIOS": "entregas instrucciones de primeros auxilios",
    "ESPERAR_RESPONDEDOR": "ya se derivó a una persona del equipo, el usuario debe esperar",
}

AVISO = "ORION es una guía de apoyo. No sustituye a personal de salud ni a servicios de emergencia."

SYSTEM_PROMPT = f"""Eres ORION, asistente de emergencias de la UTFSM. Conversas con una persona que ya reportó un incidente;
el equipo de emergencias de la universidad ya fue notificado. No le pidas que repita la ubicación ni el tipo de emergencia.

REGLAS OBLIGATORIAS:
1. Si es tu primera respuesta tras el mensaje de bienvenida, incluye: "{AVISO}"
2. Responde ÚNICAMENTE con información de los PROTOCOLOS entregados. No inventes procedimientos.
3. Si la situación no está cubierta por los protocolos, dilo ("No cuento con información suficiente para esa situación.") y deriva a humano.
4. Lenguaje claro, directo y breve (máximo 4 oraciones). Pide como máximo 1-2 datos por mensaje.
5. Usa los "Criterios de escalamiento automático" del protocolo para decidir riesgo_vital.
6. Ante duda sobre riesgo para la vida: riesgo_vital=true y derivar_a_humano=true.
7. Si el mensaje no tiene relación con la emergencia, redirige amablemente al tema (accion NINGUNA).

accion_sugerida debe ser EXACTAMENTE uno de:
{chr(10).join(f"- {k}: {v}" for k, v in ACCIONES.items())}

FORMATO DE SALIDA: responde SOLO con un objeto JSON válido, sin texto adicional ni bloques de código:
{{
  "respuesta_texto": "texto para la persona",
  "accion_sugerida": "PEDIR_MAS_INFO",
  "riesgo_vital": false,
  "derivar_a_humano": false,
  "protocolo": "código del protocolo aplicado, ej. SAL-001",
  "datos_clave": {{"heridos": 1, "consciente": false}}
}}"""


# ---------- Embeddings y búsqueda ----------

def generar_vector(texto: str) -> list[float]:
    """Convierte texto a vector semántico con Titan Embeddings v2 (1024 dims)."""
    respuesta = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": texto.strip()[:40000], "dimensions": 1024, "normalize": True}),
    )
    return json.loads(respuesta["body"].read())["embedding"]


def vector_sql(vector: list[float]) -> str:
    """Formato literal de pgvector: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


def buscar_contexto(consulta: str, k: int = TOP_K_RESULTS):
    vector = vector_sql(generar_vector(consulta))
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT codigo_protocolo, texto_completo, embedding <=> %s::vector AS distancia
                FROM protocolos_vectores
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, vector, k),
            )
            return cur.fetchall()
    finally:
        conn.close()


def ping_db() -> bool:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        return True
    except Exception:
        return False


# ---------- Historial ----------

def normalizar_historial(historial: list[dict]) -> list[dict]:
    """Convierte el historial al formato Converse: empieza con user, roles alternados."""
    mensajes: list[dict] = []
    for m in historial[-MAX_HISTORIAL:]:
        texto = (m.get("contenido") or "").strip()
        if not texto:
            continue
        rol = "assistant" if str(m.get("emisor", "")).upper() in ("BOT", "ASISTENTE", "ASSISTANT") else "user"
        if mensajes and mensajes[-1]["role"] == rol:
            mensajes[-1]["content"][0]["text"] += "\n" + texto
        else:
            mensajes.append({"role": rol, "content": [{"text": texto}]})
    while mensajes and mensajes[0]["role"] != "user":
        mensajes.pop(0)  # Converse exige empezar con user (el saludo del bot va como contexto)
    return mensajes


def consulta_para_busqueda(mensajes: list[dict], contexto: dict) -> str:
    """Subcategoría/categoría del incidente + últimos mensajes del usuario."""
    del_usuario = [m["content"][0]["text"] for m in mensajes if m["role"] == "user"]
    partes = [contexto.get("subcategoria"), contexto.get("categoria"), *del_usuario[-3:]]
    return "\n".join(p for p in partes if p)


def texto_contexto(contexto: dict) -> str:
    etiquetas = {"categoria": "Categoría", "subcategoria": "Subcategoría",
                 "ubicacion": "Ubicación", "estado_incidente": "Estado del incidente"}
    lineas = [f"- {etiquetas[k]}: {v}" for k, v in contexto.items() if v and k in etiquetas]
    return "\n".join(lineas) or "- (sin datos)"


# ---------- Respuesta ----------

def _parsear_json(texto: str) -> dict | None:
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin == -1:
        return None
    try:
        return json.loads(texto[inicio:fin + 1])
    except json.JSONDecodeError:
        return None


def _a_bool(valor) -> bool:
    if isinstance(valor, str):
        return valor.strip().lower() in ("true", "si", "sí", "1")
    return bool(valor)


def responder(historial: list[dict], contexto: dict | None = None) -> dict:
    """Devuelve respuesta_texto, accion_sugerida, riesgo_vital, derivar_a_humano y metadata.
    Lanza excepción si Bedrock/BD fallan o el modelo no entrega JSON (el backend aplica su fallback)."""
    contexto = contexto or {}
    mensajes = normalizar_historial(historial)
    if not mensajes or mensajes[-1]["role"] != "user":
        raise ValueError("El último mensaje del historial debe ser del usuario")

    protocolos = buscar_contexto(consulta_para_busqueda(mensajes, contexto))
    codigos = [codigo for codigo, _, _ in protocolos]
    textos = "\n\n---\n\n".join(f"[{codigo}]\n{texto}" for codigo, texto, _ in protocolos) or "(ninguno)"
    system = f"{SYSTEM_PROMPT}\n\nINCIDENTE REPORTADO:\n{texto_contexto(contexto)}\n\nPROTOCOLOS DISPONIBLES:\n{textos}"

    resp = bedrock.converse(
        modelId=LLM_MODEL,
        system=[{"text": system}],
        messages=mensajes,
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
    )
    salida = resp["output"]["message"]["content"][0]["text"]
    datos = _parsear_json(salida)
    if not datos or not str(datos.get("respuesta_texto", "")).strip():
        raise ValueError(f"Respuesta del modelo no es JSON válido: {salida[:200]!r}")

    accion = str(datos.get("accion_sugerida", "")).strip().upper()
    riesgo = _a_bool(datos.get("riesgo_vital", False))
    derivar = _a_bool(datos.get("derivar_a_humano", False)) or riesgo
    if accion not in ACCIONES:
        accion = "ESPERAR_RESPONDEDOR" if derivar else "NINGUNA"

    return {
        "respuesta_texto": str(datos["respuesta_texto"]).strip(),
        "accion_sugerida": accion,
        "riesgo_vital": riesgo,
        "derivar_a_humano": derivar,
        "metadata": {
            "modelo": LLM_MODEL,
            "protocolo": datos.get("protocolo"),
            "protocolos_recuperados": codigos,
            "datos_clave": datos.get("datos_clave") if isinstance(datos.get("datos_clave"), dict) else {},
            "tokens": resp.get("usage", {}),
        },
    }
