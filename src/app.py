"""API interna del chatbot ORION (contrato Backend -> Chatbot).

GET  /health    -> estado del servicio (sin token)
POST /v1/chat   -> recibe contexto del incidente + historial y devuelve la respuesta (requiere X-Internal-Token)

El chatbot NO guarda estado: el backend envía el historial en cada llamada.
Si algo falla responde 503 y el backend entrega su mensaje de fallback (131/132/133, es_fallback = true).
"""
import hmac
import logging
import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import AliasChoices, BaseModel, Field

from config import CHATBOT_MODO, INTERNAL_TOKEN, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("orion-chatbot")

if not INTERNAL_TOKEN:
    raise RuntimeError("Falta la variable de entorno INTERNAL_TOKEN")

app = FastAPI(title="ORION Chatbot", version="1.0.0")


# ---------- Contrato ----------

class Mensaje(BaseModel):
    emisor: str = Field(..., validation_alias=AliasChoices("emisor", "rol"))  # USUARIO | BOT
    contenido: str = Field(..., max_length=4000)


class Contexto(BaseModel):
    categoria: Optional[str] = None         # Salud | Infraestructura | Seguridad
    subcategoria: Optional[str] = None      # ej. "Violencia interpersonal"
    ubicacion: Optional[str] = None         # ej. "Edificio A"
    estado_incidente: Optional[str] = None  # Recibido | En camino


class ChatRequest(BaseModel):
    conversacion_id: Optional[int | str] = None
    contexto: Contexto = Field(default_factory=Contexto)
    historial: list[Mensaje] = Field(default_factory=list, description="Mensajes USUARIO y BOT en orden cronológico")
    mensaje: Optional[str] = Field(None, max_length=4000, description="Mensaje actual del usuario, si no viene ya al final del historial")


class ChatResponse(BaseModel):
    respuesta_texto: str
    accion_sugerida: str   # NINGUNA | PEDIR_MAS_INFO | EVACUAR | LLAMAR_EMERGENCIAS | PRIMEROS_AUXILIOS | ESPERAR_RESPONDEDOR
    riesgo_vital: bool
    derivar_a_humano: bool
    metadata: dict = {}    # modelo, latencia_ms, protocolos, datos_clave, request_id (auditoría, la app no lo ve)


# ---------- Seguridad ----------

def verificar_token(x_internal_token: str = Header(default="")):
    if not hmac.compare_digest(x_internal_token.encode(), INTERNAL_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Token interno inválido")


# ---------- Endpoints ----------

@app.get("/health")
def health():
    estado = {"status": "ok", "modo": CHATBOT_MODO}
    if CHATBOT_MODO == "rag":
        from rag import ping_db
        estado["modelo"] = LLM_MODEL
        estado["db"] = "ok" if ping_db() else "error"
    return estado


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verificar_token)])
def chat(req: ChatRequest, x_request_id: str = Header(default="")):
    request_id = x_request_id or str(uuid.uuid4())
    inicio = time.monotonic()

    historial = [m.model_dump() for m in req.historial]
    if req.mensaje and req.mensaje.strip():
        historial.append({"emisor": "USUARIO", "contenido": req.mensaje})
    if not any(m["emisor"].upper() == "USUARIO" for m in historial):
        raise HTTPException(status_code=422, detail="No hay mensaje del usuario")

    log.info("request_id=%s conversacion_id=%s mensajes=%d", request_id, req.conversacion_id, len(historial))

    if CHATBOT_MODO == "eco":
        ultimo = next(m["contenido"] for m in reversed(historial) if m["emisor"].upper() == "USUARIO")
        resultado = {"respuesta_texto": f"Eco: {ultimo}", "accion_sugerida": "PEDIR_MAS_INFO",
                     "riesgo_vital": False, "derivar_a_humano": False, "metadata": {"modelo": "eco"}}
    else:
        from rag import responder
        try:
            resultado = responder(historial, req.contexto.model_dump())
        except Exception:
            log.exception("request_id=%s error generando respuesta", request_id)
            raise HTTPException(status_code=503, detail="Chatbot no disponible")

    resultado["metadata"]["request_id"] = request_id
    resultado["metadata"]["latencia_ms"] = int((time.monotonic() - inicio) * 1000)
    log.info("request_id=%s accion=%s riesgo_vital=%s latencia_ms=%s", request_id,
             resultado["accion_sugerida"], resultado["riesgo_vital"], resultado["metadata"]["latencia_ms"])
    return ChatResponse(**resultado)
