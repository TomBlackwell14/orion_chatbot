import boto3
import json
import re
import psycopg2
from config import DB_CONFIG, AWS_REGION, EMBEDDING_MODEL
from pathlib import Path

# Conexión a AWS Bedrock
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)

def generar_vector(texto: str):
    """Convierte texto a vector semántico con Titan Embeddings"""
    respuesta = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": texto.strip()})
    )
    return json.loads(respuesta["body"].read())["embedding"]

# Leer protocolos
ruta_protocolos = Path(__file__).parent.parent / "docs" / "protocolos_completos.txt"
with open(ruta_protocolos, "r", encoding="utf-8") as f:
    documento = f.read()

# Dividir por cada protocolo
patron = r'(?=\n(?:SAL|INF|SEG)-\d+ — )'
chunks = re.split(patron, documento)
chunks = [c.strip() for c in chunks if c.strip()]

print(f"📄 Se identificaron {len(chunks)} protocolos")

# Conectar a PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Crear estructura si no existe
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
CREATE TABLE IF NOT EXISTS protocolos_vectores (
    id SERIAL PRIMARY KEY,
    codigo_protocolo VARCHAR(20),
    texto_completo TEXT,
    embedding vector(1024),
    fecha_creacion TIMESTAMP DEFAULT NOW()
);
""")
cur.execute("""
CREATE INDEX IF NOT EXISTS idx_protocolos_embedding
ON protocolos_vectores USING hnsw (embedding vector_cosine_ops);
""")
conn.commit()

# Limpiar anteriores e insertar nuevos
cur.execute("DELETE FROM protocolos_vectores;")

for i, texto in enumerate(chunks, 1):
    match = re.match(r'((?:SAL|INF|SEG)-\d+)', texto)
    codigo = match.group(1) if match else f"PROTO-{i}"
    
    vector = generar_vector(texto)
    
    cur.execute("""
        INSERT INTO protocolos_vectores (codigo_protocolo, texto_completo, embedding)
        VALUES (%s, %s, %s)
    """, (codigo, texto, vector))
    
    print(f"✅ {codigo} → vector guardado")

conn.commit()
cur.close()
conn.close()
print("\n🎉 TODOS los protocolos indexados correctamente")