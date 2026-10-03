import boto3
import json
import psycopg2
from config import DB_CONFIG, AWS_REGION, EMBEDDING_MODEL, LLM_MODEL, TOP_K_RESULTS

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

def generar_vector(texto: str):
    respuesta = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": texto})
    )
    return json.loads(respuesta["body"].read())["embedding"]

def buscar_contexto(pregunta: str):
    vector = generar_vector(pregunta)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT codigo_protocolo, texto_completo,
               embedding <=> %s AS distancia
        FROM protocolos_vectores
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (vector, vector, TOP_K_RESULTS))
    resultados = cur.fetchall()
    cur.close()
    conn.close()
    return resultados

def responder(pregunta: str):
    protocolos = buscar_contexto(pregunta)
    if not protocolos:
        return "No cuento con información para esa consulta."
    
    contexto = "\n\n---\n\n".join([f"[{p[0]}]\n{p[1]}" for p in protocolos])
    
    prompt = f"""Eres ORION, asistente de emergencias de la UTFSM.

INSTRUCCIONES OBLIGATORIAS:
1. SIEMPRE comienza diciendo: "ORION es una guía de apoyo. No sustituye a personal de salud ni a servicios de emergencia."
2. Responde ÚNICAMENTE con información de los protocolos de abajo.
3. Si la respuesta no está en los protocolos, di: "No cuento con información suficiente para esa situación."
4. Usa lenguaje claro y directo.

PROTOCOLOS DISPONIBLES:
{contexto}

PREGUNTA: {pregunta}
"""

    respuesta = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": 800,
            "temperature": 0.1
        })
    )
    return json.loads(respuesta["body"].read())["output"]["message"]["content"][0]["text"]

if __name__ == "__main__":
    print("🤖 ORION Chatbot — Escribe 'salir' para terminar\n")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() in ["salir", "exit", "q"]:
            print("👋 Hasta luego")
            break
        print("ORION: ", end="", flush=True)
        print(responder(pregunta), "\n")