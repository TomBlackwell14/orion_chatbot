"""Indexa los protocolos en PostgreSQL + pgvector (1 vector por protocolo).

Uso:  python index.py
Fuente: PROTOCOLOS_S3_URI (s3://bucket/clave.txt) si está definida; si no, docs/protocolos_completos.txt
"""
import re
from pathlib import Path

import boto3
import psycopg2

from config import AWS_REGION, DB_CONFIG, EMBEDDING_DIM, PROTOCOLOS_S3_URI
from rag import generar_vector, vector_sql


def leer_protocolos() -> str:
    if PROTOCOLOS_S3_URI:
        bucket, _, clave = PROTOCOLOS_S3_URI.removeprefix("s3://").partition("/")
        s3 = boto3.client("s3", region_name=AWS_REGION)
        print(f"📥 Leyendo s3://{bucket}/{clave}")
        return s3.get_object(Bucket=bucket, Key=clave)["Body"].read().decode("utf-8")
    ruta = Path(__file__).resolve().parent.parent / "docs" / "protocolos_completos.txt"
    print(f"📥 Leyendo {ruta}")
    return ruta.read_text(encoding="utf-8")


def dividir_protocolos(documento: str) -> list[tuple[str, str]]:
    """Divide por protocolo y une los trozos con el mismo código (algunos repiten el título)."""
    partes = [p.strip() for p in re.split(r"(?=^(?:SAL|INF|SEG)-\d+ — )", documento, flags=re.MULTILINE) if p.strip()]
    protocolos: dict[str, str] = {}
    for i, texto in enumerate(partes, 1):
        match = re.match(r"((?:SAL|INF|SEG)-\d+)", texto)
        codigo = match.group(1) if match else f"PROTO-{i}"
        protocolos[codigo] = f"{protocolos[codigo]}\n\n{texto}" if codigo in protocolos else texto
    return list(protocolos.items())


def main():
    protocolos = dividir_protocolos(leer_protocolos())
    print(f"📄 Se identificaron {len(protocolos)} protocolos")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS protocolos_vectores (
        id SERIAL PRIMARY KEY,
        codigo_protocolo VARCHAR(20),
        texto_completo TEXT,
        embedding vector({EMBEDDING_DIM}),
        fecha_creacion TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_protocolos_embedding
    ON protocolos_vectores USING hnsw (embedding vector_cosine_ops);
    """)

    # Generar todos los vectores antes de borrar, para no dejar la tabla vacía si Bedrock falla
    filas = []
    for codigo, texto in protocolos:
        filas.append((codigo, texto, vector_sql(generar_vector(texto))))
        print(f"✅ {codigo} → vector generado")

    cur.execute("DELETE FROM protocolos_vectores;")
    cur.executemany(
        "INSERT INTO protocolos_vectores (codigo_protocolo, texto_completo, embedding) VALUES (%s, %s, %s::vector)",
        filas,
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"\n🎉 {len(filas)} protocolos indexados correctamente")


if __name__ == "__main__":
    main()
