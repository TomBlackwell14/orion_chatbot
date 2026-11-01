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
src/evaluar_riesgo_vital.py  evaluación de riesgo vital sobre docs/casos_riesgo_vital.json (recall, matriz de confusión)
src/evaluar_orientacion.py   evaluación de la orientación con docs/escenarios_orientacion.json y un LLM juez
src/orientacion.py           rúbrica, chequeos automáticos y juez · src/metricas.py  métricas e intervalos
src/acuerdo_juez.py          compara el juez con tu revisión manual
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

# En la EC2: evaluación de riesgo vital con Bedrock real (ver "Evaluación del chatbot")
docker compose run --rm -v "$PWD/resultados:/tmp/resultados" chatbot python evaluar_riesgo_vital.py --salida /tmp/resultados

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

## Evaluación del chatbot

Dos evaluaciones separadas, porque miden cosas distintas. Las dos dejan un `reporte.md` con tablas listas para la
tesis en `resultados/<tipo>_<fecha>/`. Ninguna se corre en los tests: usan Bedrock real y cuestan dinero.

### 1. Detección de riesgo vital (`evaluar_riesgo_vital.py`)

`docs/casos_riesgo_vital.json` tiene 144 frases etiquetadas (79 con riesgo vital y 65 sin), con un `tipo`
(coloquial, sin palabra clave, negación, ya resuelto, falsa alarma, par casi igual, multi-turno…) y la
justificación de cada etiqueta según GEN-000 o el protocolo. Los casos `multiturno` traen `historial_previo`:
el riesgo aparece en la respuesta del usuario a una pregunta del bot ("¿respira?" → "no").

```bash
python evaluar_riesgo_vital.py --solo-reglas      # solo riesgo.py: gratis, sin AWS
python evaluar_riesgo_vital.py                    # Bedrock real, 5 repeticiones por frase (~720 llamadas)
python evaluar_riesgo_vital.py -n 3 --tipo multiturno
```
Reporta tres capas por separado (solo reglas, solo modelo y sistema completo), con recall e intervalo de
confianza de Wilson, precisión, especificidad y F2. Agrega el **peor caso** (una frase grave cuenta como detectada
solo si se detectó en todas las repeticiones), la **estabilidad** entre repeticiones, y el desglose por tipo y
subcategoría. El error que importa es el **falso negativo**: termina con código 1 si hay alguno en el peor caso.

### 2. Calidad de la orientación (`evaluar_orientacion.py`)

`docs/escenarios_orientacion.json` tiene 29 conversaciones guionadas (46 turnos) que cubren los 12 protocolos.
Cada turno trae el mensaje del usuario, el `riesgo_vital` esperado y listas `debe` / `no_debe` sacadas del
protocolo. El bot responde de verdad y su respuesta entra al historial del turno siguiente.

Cada respuesta se evalúa con:
- **Chequeos automáticos** (código): riesgo_vital del turno, máximo 4 oraciones y 2 preguntas, aviso en la primera
  respuesta, sin números externos y sin prometer llegadas ni tiempos.
- **LLM juez** (Claude en Bedrock, `JUEZ_MODEL`, por defecto `anthropic.claude-opus-5-5`): recibe GEN-000, el
  protocolo esperado y la conversación, y marca cada criterio como cumple / no cumple con una cita de la respuesta.
  Criterios generales: seguridad, fidelidad al protocolo, prioridad, no repetir preguntas y tono; más los `debe` y
  `no_debe` del turno. Seguridad y `no_debe` son críticos: si fallan, el escenario no se aprueba.

```bash
pip install -r src/requirements-eval.txt
cd src
python evaluar_orientacion.py --sin-juez -n 1                 # primero, sin costo del juez
python evaluar_orientacion.py                                 # 3 repeticiones con juez
python evaluar_orientacion.py --desde ../resultados/orientacion_<fecha>   # re-juzga sin volver a llamar al bot
```

**Validar al juez**: cada corrida deja `revision_manual.csv` con una muestra de conversaciones. Completa la columna
`veredicto_humano` con "si" o "no" según tu criterio y corre `python acuerdo_juez.py <ruta del csv>`. Con un
acuerdo de 85-90% o más y kappa sobre 0.6 el juez es confiable para el resto.

**Dónde correrlo**: el juez usa el SDK de Anthropic (no está en la imagen de producción) y necesita acceso a Claude
en Bedrock (*Model access*), que el rol de la EC2 no tiene. Lo más simple es correr esta evaluación desde tu PC con
un perfil de AWS que tenga acceso a Nova Lite, Titan Embeddings y Claude (`AWS_PROFILE=... python evaluar_orientacion.py`).
Si en tu región no está disponible, cambia `JUEZ_REGION` o `JUEZ_MODEL`.
