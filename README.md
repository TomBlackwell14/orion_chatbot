# ORION Chatbot (RAG con Bedrock + pgvector)

```
App Flutter ──(JWT Cognito)──▶ Backend NestJS :3000 ──(orion-network + X-Internal-Token)──▶ Chatbot :8000 ──▶ Bedrock
                                                                                              └──▶ pgvector (protocolos)
```

El chatbot es **sin estado**: no guarda conversaciones. El backend le envía el contexto del incidente y el historial en cada llamada.

```
src/app.py        API (GET /health, POST /v1/chat)
src/rag.py        embeddings Titan + búsqueda pgvector + respuesta Nova (Converse)
src/index.py      indexa docs/protocolos_completos.txt (o S3) en la tabla protocolos_vectores
src/preguntar.py  prueba por consola sin API
aws/iam-policy-chatbot.json  política para el rol IAM de la EC2
```

## Contrato interno Backend → Chatbot

`POST http://chatbot:8000/v1/chat`
Headers: `X-Internal-Token: <INTERNAL_TOKEN>` (obligatorio) · `X-Request-Id: <id>` (opcional, para seguir el mensaje en los logs)

```json
{
  "conversacion_id": 12,
  "contexto": {
    "categoria": "Seguridad",
    "subcategoria": "Violencia interpersonal",
    "ubicacion": "Edificio A",
    "estado_incidente": "Recibido"
  },
  "historial": [
    {"emisor": "BOT",     "contenido": "Hola, soy el asistente de ORION... ¿Qué está pasando ahora?"},
    {"emisor": "USUARIO", "contenido": "Hay una pelea afuera y una persona sangra del brazo"}
  ]
}
```
- `historial`: solo mensajes `USUARIO` y `BOT`, en orden. El último debe ser del usuario (o enviarlo aparte en `"mensaje"`).
- Todo `contexto` es opcional, pero con `subcategoria` el chatbot encuentra mejor el protocolo.
- No enviar nombre, correo, RUT ni id_usuario.

Respuesta `200`:
```json
{
  "respuesta_texto": "Mantén distancia... ¿La persona está consciente?",
  "accion_sugerida": "PRIMEROS_AUXILIOS",
  "riesgo_vital": false,
  "derivar_a_humano": false,
  "metadata": {
    "modelo": "us.amazon.nova-lite-v1:0",
    "protocolo": "SEG-001",
    "protocolos_recuperados": ["SEG-001", "SAL-003", "SAL-001"],
    "datos_clave": {"heridos": 1},
    "tokens": {"inputTokens": 3500, "outputTokens": 120},
    "request_id": "…",
    "latencia_ms": 2400
  }
}
```
`accion_sugerida` ∈ `NINGUNA · PEDIR_MAS_INFO · EVACUAR · LLAMAR_EMERGENCIAS · PRIMEROS_AUXILIOS · ESPERAR_RESPONDEDOR`.
Si `riesgo_vital = true`, `derivar_a_humano` también es `true`. `metadata` es para auditoría (columna `metadata jsonb`); la app no la recibe.

Errores: `401` token inválido · `422` body inválido · `503` falló Bedrock/BD/modelo → **el backend responde su fallback** (131/132/133, `es_fallback: true`).
Timeout recomendado en el backend: 25 s.

`GET /health` (sin token) → `{"status":"ok","modo":"rag","modelo":"…","db":"ok"}`

## Despliegue en la EC2

```bash
cp .env.example .env      # completar DB_PASSWORD e INTERNAL_TOKEN (el mismo que el backend)
docker compose up -d --build
docker compose run --rm chatbot python index.py   # una vez, y cada vez que cambien los protocolos
```

## AWS (antes de usar modo rag)

1. **Bedrock → Model access** (región `us-east-2`): habilitar *Titan Text Embeddings V2* y *Amazon Nova Lite*.
2. **IAM → Roles → Create role** (EC2) con la política de `aws/iam-policy-chatbot.json` (cambiar `NOMBRE-DEL-BUCKET`, o borrar ese bloque si no usas S3). Asignarlo: *EC2 → Actions → Security → Modify IAM role*.
3. **Importante con Docker**: para que el contenedor reciba las credenciales del rol:
   ```
   aws ec2 modify-instance-metadata-options --instance-id <ID> --http-put-response-hop-limit 2 --http-endpoint enabled
   ```
   (o en la consola: *Actions → Instance settings → Modify instance metadata options → hop limit = 2*)
4. El security group de la BD debe permitir el puerto 5432 desde la EC2.
5. Protocolos en S3 (opcional): subir el .txt y poner `PROTOCOLOS_S3_URI=s3://bucket/protocolos_completos.txt`.

## Pruebas en la EC2

```bash
# 1. ¿Está vivo? (db debe decir "ok")
docker compose ps
docker compose exec chatbot python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"

# 2. ¿El contenedor tiene permisos de Bedrock? (debe imprimir 1024)
docker compose exec chatbot python -c "from rag import generar_vector; print(len(generar_vector('hola')))"

# 3. Chat completo desde la red interna (como lo haría el backend)
TOKEN=$(grep ^INTERNAL_TOKEN .env | cut -d= -f2)
docker run --rm --network orion-network curlimages/curl -s -X POST http://chatbot:8000/v1/chat \
  -H "Content-Type: application/json" -H "X-Internal-Token: $TOKEN" \
  -d '{"contexto":{"categoria":"Salud","subcategoria":"Urgencia","ubicacion":"Casino"},
       "historial":[{"emisor":"USUARIO","contenido":"Una persona se desmayó y no responde"}]}'
#    → esperar riesgo_vital: true, derivar_a_humano: true

# 4. Token malo → 401
docker run --rm --network orion-network curlimages/curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST http://chatbot:8000/v1/chat -H "X-Internal-Token: malo" -H "Content-Type: application/json" -d '{}'

# 5. Que NO sea accesible desde internet: desde tu PC, http://<IP-EC2>:8000/health debe fallar.

# 6. Logs (request_id, acción, latencia, errores)
docker compose logs -f chatbot
```

Para probar solo el contrato sin AWS: `CHATBOT_MODO=eco` en `.env` y `docker compose up -d`.
