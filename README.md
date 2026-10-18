# ORION Chatbot (RAG con Bedrock + índice local)

```
App Flutter ──(JWT Cognito)──▶ Backend NestJS :3000 ──(orion-network + X-Internal-Token)──▶ Chatbot :8000 ──▶ Bedrock
```

El chatbot es **sin estado y no usa la base de datos**: el backend le envía el contexto del incidente y el historial en
cada llamada. Los protocolos están en `docs/protocolos_completos.txt` dentro de la imagen y se indexan en memoria
(Titan Embeddings) la primera vez que se usan. Contrato de referencia: *ORION API 1.0 — Chatbot* (17-10-2026).

```
src/app.py        API (GET /health, POST /v1/chat, POST /v1/resumen)
src/rag.py        índice local + búsqueda + respuesta y resumen con Nova (Converse)
src/riesgo.py     red de seguridad determinística de riesgo vital
src/evaluar_riesgo_vital.py  evaluación con Bedrock real sobre docs/casos_riesgo_vital.json
src/preguntar.py  prueba por consola sin API
tests/test_api.py pruebas del contrato con Bedrock simulado
aws/iam-policy-chatbot.json  política para el rol IAM de la EC2
```

## Reglas del chatbot

Solo orienta con texto. No notifica, no llama, no da números externos, no asigna, no cambia subcategoría ni
prioridad y no inventa rutas ni tiempos de llegada. Su único efecto es `riesgo_vital: true`: con eso **NestJS** sube
el incidente a Crítica.

## Cómo se detecta el riesgo vital (defensa en capas)

1. **GEN-000 (marco general)** va siempre en el prompt. Define riesgo vital como "amenaza actual o inminente a la
   vida" y trae una evaluación sistemática (conciencia, respiración, circulación, amenaza del entorno, personas que
   no pueden ponerse a salvo). Las listas de cada protocolo son ejemplos, no una lista cerrada.
2. **RAG**: los 3 protocolos más parecidos al relato aportan las señales específicas de su tema.
3. **`riesgo.py`**: reglas de alta especificidad (no respira, inconsciente, convulsión, sangrado masivo, arma,
   explosivo, autolesión, atrapado en fuego o escombros, electrocución), con manejo de negaciones ("no hay armas").
   Si alguna se activa, `riesgo_vital` queda en `true` aunque el modelo diga `false`. Las reglas solo suben, nunca bajan.
4. Si Bedrock falla justo cuando hay una señal explícita, el chatbot responde una indicación fija con
   `riesgo_vital: true` (`modelo: "respuesta_segura"`) para que la detección no se pierda en el fallback de NestJS.

Cada cambio de protocolos, prompt o modelo se valida con `python evaluar_riesgo_vital.py` (Bedrock real). El
objetivo es 0 falsos negativos. Para agregar una regla, agrega también casos positivos y negados en
`docs/casos_riesgo_vital.json`.

## POST /v1/chat

Headers: `X-Internal-Token` (obligatorio) · `X-Request-Id`.

```json
{
  "request_id": "5f0c1c3e-7b8a-4b8e-9a51-2f0d6c1f4a10",
  "id_incidente": 33,
  "id_conversacion": 12,
  "mensaje": "Hay una pelea afuera del edificio A y una persona está sangrando del brazo",
  "historial": [{"rol": "bot", "texto": "Hola, soy el asistente de ORION. ¿Qué está pasando ahora?"}],
  "contexto": {
    "rol_usuario": "Reportante",
    "estado_incidente": "Recibido",
    "categoria": {"id_categoria": 3, "nombre": "Seguridad"},
    "subcategoria": {"id_subcategoria": 9, "nombre": "Violencia interpersonal"},
    "ubicacion_incidente": {"id_ubicacion": 1, "nombre": "Edificio A", "latitud": -33.03679, "longitud": -71.48688},
    "ubicacion_actual": null
  }
}
```
Respuesta `200`:
```json
{"request_id": "5f0c1c3e-…", "respuesta_texto": "Mantén distancia…", "riesgo_vital": false,
 "modelo": "us.amazon.nova-lite-v1:0", "protocolo": "SEG-001", "latencia_ms": 2350}
```

## POST /v1/resumen

Lo llama NestJS cuando el Respondedor asignado pulsa «Ver resumen IA». Recibe `request_id`, `id_incidente`,
`id_conversacion`, `contexto` (`estado_incidente`, `subcategoria`, `ubicacion_incidente`) e `historial` completo.

```json
{"request_id": "5f0c1c3e-…", "resumen_ia": "El reportante indica que una persona no responde.",
 "datos_clave": {"puntos_criticos": ["Persona que no responde"],
                 "orientaciones_entregadas": ["El chatbot indicó verificar si la persona respira"]}}
```
No devuelve `riesgo_vital` (lo calcula NestJS con los mensajes BOT guardados).

Errores de ambos: `401` token inválido · `422` body inválido · `503` falló Bedrock o el modelo no entregó JSON →
NestJS usa su fallback. Timeout del backend: 25 s.

`GET /health` (sin token) → `{"status":"ok","modo":"rag","modelo":"…","protocolos":12}`

## Despliegue en la EC2

```bash
cp .env.example .env      # completar INTERNAL_TOKEN (el mismo valor que CHATBOT_INTERNAL_TOKEN en NestJS)
docker compose up -d --build
```
Cuando cambien los protocolos basta con `docker compose up -d --build`: el índice se rearma solo.

## AWS (antes de usar modo rag)

1. **Bedrock → Model access** (región `us-east-2`): habilitar *Titan Text Embeddings V2* y *Amazon Nova Lite*.
2. **IAM → Roles** (EC2) con la política de `aws/iam-policy-chatbot.json`, asignado a la instancia.
3. **Con Docker**, para que el contenedor reciba las credenciales del rol: metadata hop limit = 2
   (`aws ec2 modify-instance-metadata-options --instance-id <ID> --http-put-response-hop-limit 2 --http-endpoint enabled`).

## Pruebas

```bash
# Sin AWS, en tu PC
pip install -r src/requirements.txt pytest httpx
python -m pytest -q tests

# En la EC2: evaluación de riesgo vital con Bedrock real (debe terminar con FN 0)
docker compose run --rm chatbot python evaluar_riesgo_vital.py

# En la EC2, desde la red interna (como lo hace NestJS)
TOKEN=$(grep ^INTERNAL_TOKEN .env | cut -d= -f2)
docker run --rm --network orion-network curlimages/curl -s http://chatbot:8000/health
docker run --rm --network orion-network curlimages/curl -s -X POST http://chatbot:8000/v1/chat \
  -H "Content-Type: application/json" -H "X-Internal-Token: $TOKEN" \
  -d '{"request_id":"prueba-1","id_incidente":33,"id_conversacion":12,"mensaje":"Una persona se desmayó y no responde",
       "historial":[],"contexto":{"estado_incidente":"Recibido","categoria":{"id_categoria":1,"nombre":"Salud"},
       "subcategoria":{"id_subcategoria":1,"nombre":"Urgencia"},"ubicacion_incidente":{"id_ubicacion":1,"nombre":"Casino"}}}'
#   → esperar riesgo_vital: true
```
Desde tu PC, `http://<IP-EC2>:8000/health` debe **fallar** (el puerto no se publica).
Para probar solo el contrato sin AWS: `CHATBOT_MODO=eco` en `.env`.
