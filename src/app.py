"""API interna del chatbot ORION (contrato Backend -> Chatbot, ver ORION API 1.0 — Chatbot del 17-10-2026).

GET  /health      -> estado del servicio (sin token)
POST /v1/chat     -> mensaje + historial + contexto del incidente; responde orientación y riesgo_vital
POST /v1/resumen  -> historial completo; responde resumen_ia y datos_clave para el Respondedor asignado

Los dos POST requieren X-Internal-Token. El chatbot NO guarda estado ni usa la BD: el backend envía todo en cada
llamada. Si algo falla responde 503 y NestJS aplica su fallback (chat) o conserva el análisis previo (resumen).
"""
import hmac
import logging
import time
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config import CHATBOT_MODO, INTERNAL_TOKEN, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("orion-chatbot")

if not INTERNAL_TOKEN:
    raise RuntimeError("Falta la variable de entorno INTERNAL_TOKEN")

app = FastAPI(title="ORION Chatbot", version="2.0.0")


# ---------- Contrato: lo que manda NestJS ----------

class Turno(BaseModel):
    rol: Literal["usuario", "bot"]
    texto: str = Field(..., max_length=4000)


class Ubicacion(BaseModel):
    id_ubicacion: Optional[int] = None
    nombre: Optional[str] = None
    latitud: Optional[float] = None
    longitud: Optional[float] = None


class CategoriaRef(BaseModel):
    id_categoria: Optional[int] = None
    nombre: str


class SubcategoriaRef(BaseModel):
    id_subcategoria: Optional[int] = None
    nombre: str


class ContextoChat(BaseModel):
    rol_usuario: Optional[str] = None
    estado_incidente: Optional[str] = None  # Recibido | En_camino
    categoria: Optional[CategoriaRef] = None
    subcategoria: Optional[SubcategoriaRef] = None
    ubicacion_incidente: Optional[Ubicacion] = None
    ubicacion_actual: Optional[Ubicacion] = None


class ChatRequest(BaseModel):
    request_id: str
    id_incidente: int
    id_conversacion: int
    mensaje: str = Field(..., min_length=1, max_length=4000)
    historial: list[Turno] = Field(default_factory=list)  # hasta 10 anteriores, sin el mensaje actual
    contexto: ContextoChat


class ContextoResumen(BaseModel):
    estado_incidente: Optional[str] = None
    subcategoria: Optional[SubcategoriaRef] = None
    ubicacion_incidente: Optional[Ubicacion] = None


class ResumenRequest(BaseModel):
    request_id: str
    id_incidente: int
    id_conversacion: int
    contexto: ContextoResumen = Field(default_factory=ContextoResumen)
    historial: list[Turno]  # conversación completa USUARIO/BOT


# ---------- Contrato: lo que responde el chatbot ----------

class ChatResponse(BaseModel):
    request_id: str
    respuesta_texto: str
    riesgo_vital: bool
    modelo: Optional[str] = None
    protocolo: Optional[str] = None
    latencia_ms: int = 0


class DatosClave(BaseModel):
    puntos_criticos: list[str] = Field(default_factory=list)
    orientaciones_entregadas: list[str] = Field(default_factory=list)


class ResumenResponse(BaseModel):
    request_id: str
    resumen_ia: str
    datos_clave: DatosClave


# ---------- Seguridad ----------

def verificar_token(x_internal_token: str = Header(default="")):
    if not hmac.compare_digest(x_internal_token.encode(), INTERNAL_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Token interno inválido")


# ---------- Adaptador hacia rag.py ----------

def _nombre_ubicacion(u: Optional[Ubicacion]) -> Optional[str]:
    if u is None:
        return None
    if u.nombre:
        return u.nombre
    if u.latitud is not None and u.longitud is not None:
        return f"{u.latitud}, {u.longitud}"
    return None


def contexto_para_rag(estado: Optional[str], categoria: Optional[CategoriaRef],
                      subcategoria: Optional[SubcategoriaRef], ubicacion: Optional[Ubicacion],
                      actual: Optional[Ubicacion] = None) -> dict:
    """rag.py trabaja con textos simples; aquí se traducen los objetos del contrato."""
    texto_ubicacion = _nombre_ubicacion(ubicacion)
    texto_actual = _nombre_ubicacion(actual)
    if texto_actual:
        texto_ubicacion = f"{texto_ubicacion or 'sin dato'} (el usuario ahora está en: {texto_actual})"
    return {
        "categoria": categoria.nombre if categoria else None,
        "subcategoria": subcategoria.nombre if subcategoria else None,
        "ubicacion": texto_ubicacion,
        "estado_incidente": estado,  # se deja "En_camino" tal cual: el prompt lo reconoce
    }


# ---------- Endpoints ----------

@app.get("/health")
def health():
    estado = {"status": "ok", "modo": CHATBOT_MODO}
    if CHATBOT_MODO == "rag":
        from rag import protocolos_cargados
        estado["modelo"] = LLM_MODEL
        try:
            estado["protocolos"] = protocolos_cargados()
        except OSError:
            estado["protocolos"] = 0
    return estado


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verificar_token)])
def chat(req: ChatRequest):
    inicio = time.monotonic()
    ctx = req.contexto
    log.info("request_id=%s endpoint=chat incidente=%s conversacion=%s historial=%d", req.request_id,
             req.id_incidente, req.id_conversacion, len(req.historial))

    if CHATBOT_MODO == "eco":
        respuesta = ChatResponse(request_id=req.request_id, respuesta_texto=f"[eco] Recibí: {req.mensaje}",
                                 riesgo_vital=False, modelo="eco")
    else:
        from rag import responder
        historial = [t.model_dump() for t in req.historial] + [{"rol": "usuario", "texto": req.mensaje}]
        try:
            r = responder(historial, contexto_para_rag(ctx.estado_incidente, ctx.categoria, ctx.subcategoria,
                                                       ctx.ubicacion_incidente, ctx.ubicacion_actual))
        except Exception as e:
            log.error("request_id=%s endpoint=chat error=%s", req.request_id, type(e).__name__)
            raise HTTPException(status_code=503, detail="Chatbot no disponible")
        respuesta = ChatResponse(request_id=req.request_id, respuesta_texto=r["respuesta_texto"],
                                 riesgo_vital=r["riesgo_vital"], modelo=LLM_MODEL, protocolo=r["protocolo"])
        meta = r["metadata"]
        log.info("request_id=%s protocolos_recuperados=%s senales_regla=%s riesgo_vital_modelo=%s respuesta_segura=%s",
                 req.request_id, meta["protocolos_recuperados"], meta["senales_regla"], meta["riesgo_vital_modelo"],
                 meta.get("respuesta_segura", False))
        if meta.get("respuesta_segura"):
            respuesta.modelo = "respuesta_segura"

    respuesta.latencia_ms = int((time.monotonic() - inicio) * 1000)
    log.info("request_id=%s endpoint=chat riesgo_vital=%s protocolo=%s latencia_ms=%s", req.request_id,
             respuesta.riesgo_vital, respuesta.protocolo, respuesta.latencia_ms)
    return respuesta


@app.post("/v1/resumen", response_model=ResumenResponse, dependencies=[Depends(verificar_token)])
def resumen(req: ResumenRequest):
    inicio = time.monotonic()
    ctx = req.contexto
    log.info("request_id=%s endpoint=resumen incidente=%s conversacion=%s historial=%d", req.request_id,
             req.id_incidente, req.id_conversacion, len(req.historial))

    if not any(t.rol == "usuario" and t.texto.strip() for t in req.historial):
        raise HTTPException(status_code=422, detail="El historial no tiene mensajes del usuario")

    if CHATBOT_MODO == "eco":
        respuesta = ResumenResponse(request_id=req.request_id,
                                    resumen_ia=f"[eco] Conversación con {len(req.historial)} mensajes.",
                                    datos_clave=DatosClave())
    else:
        from rag import resumir
        try:
            r = resumir([t.model_dump() for t in req.historial],
                        contexto_para_rag(ctx.estado_incidente, None, ctx.subcategoria, ctx.ubicacion_incidente))
        except Exception as e:
            log.error("request_id=%s endpoint=resumen error=%s", req.request_id, type(e).__name__)
            raise HTTPException(status_code=503, detail="Resumen no disponible")
        respuesta = ResumenResponse(request_id=req.request_id, resumen_ia=r["resumen_ia"],
                                    datos_clave=DatosClave(**r["datos_clave"]))

    log.info("request_id=%s endpoint=resumen puntos=%d orientaciones=%d latencia_ms=%d", req.request_id,
             len(respuesta.datos_clave.puntos_criticos), len(respuesta.datos_clave.orientaciones_entregadas),
             int((time.monotonic() - inicio) * 1000))
    return respuesta
