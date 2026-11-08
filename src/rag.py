"""Lógica RAG de ORION: embeddings (Titan) + índice en memoria + respuesta con Bedrock (Converse).

Sin estado y sin base de datos: los protocolos vienen en docs/protocolos_completos.txt dentro de la
imagen y se indexan en memoria la primera vez que se necesitan. Todo lo demás llega en cada petición.

Riesgo vital en tres capas:
1. GEN-000 (marco general con el criterio de riesgo vital) va SIEMPRE en el prompt, no depende de la búsqueda.
2. Los protocolos recuperados aportan ejemplos específicos de señales.
3. riesgo.py fuerza riesgo_vital = true ante señales explícitas en el mensaje del usuario.
"""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config

from riesgo import detectar_senales
from config import (
    AWS_REGION, CONTACTOS_SEDE, DETECTOR_RIESGO, EMBEDDING_DIM, EMBEDDING_MODEL, LLM_MODEL,
    MAX_HISTORIAL, MAX_TOKENS, PROTOCOLOS_PATH, TEMPERATURE, TOP_K_RESULTS,
)

log = logging.getLogger("orion-chatbot.rag")

# El backend espera como máximo 25 s y /v1/chat hace 2 llamadas (embedding + converse): sin reintentos,
# cada una toma a lo más 2 + 10 s, así el peor caso (24 s) queda dentro del límite.
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(connect_timeout=2, read_timeout=10, retries={"total_max_attempts": 1}),
)

AVISO = "ORION es una guía de apoyo. No sustituye a personal de salud ni a servicios de emergencia."
# Lo que el código dice del estado del incidente (GEN-000 sección 6): siempre verdadero, nunca lo inventa el modelo
MENSAJES_ESTADO = {
    "Recibido": "Tu reporte ya fue notificado al equipo de la sede y puedes seguir su estado en la app.",
    "En_camino": "Un respondedor ya va en camino.",
}

# Contactos que entrega el código, con el texto exacto (GEN-000 sección 7). El modelo no los escribe: cuando la
# persona pide un teléfono, contacto_pedido() agrega el que corresponde.
CONTACTO_SALUD = "Teléfono de Salud de la sede (policlínico): +56 32 227 7888."
CONTACTO_SEGURIDAD = "Teléfono de Seguridad de la sede: anexo 999 desde un teléfono interno, o +56 32 227 7999 desde celular."
CONTACTO_INFRAESTRUCTURA = "No hay un teléfono de Mantenimiento para entregar: tu reporte ya quedó registrado y lo verá el personal."
FRASES_FIJAS = (AVISO, *MENSAJES_ESTADO.values(), CONTACTO_SALUD, CONTACTO_SEGURIDAD, CONTACTO_INFRAESTRUCTURA)

REGLAS_ORION = f"""REGLAS QUE NUNCA SE ROMPEN:
- Tú solo orientas con texto. No ejecutas acciones en el sistema.
- No digas que alguien fue notificado ni des tiempos de llegada.
- No digas que el personal va en camino, salvo que el estado del incidente sea "En_camino" (entonces puedes decir
  que un respondedor ya va en camino, sin dar tiempos).
- Tú no haces llamadas ni dices que llamarás o avisarás a alguien. Eso no impide que la persona llame: nunca le
  digas que no llame, que no es necesario llamar ni que no puedes darle un teléfono. Si pide un teléfono, el sistema
  agrega el contacto correcto al final de tu respuesta: no lo escribas tú. Si no lo pidió pero necesita comunicarse
  directamente, puedes entregar el contacto de la sede: {CONTACTOS_SEDE}. No existe un contacto de Infraestructura
  ni de Mantenimiento. No menciones números de emergencia externos (131, 132, 133 u otros).
- No cambies ni sugieras cambiar la categoría, la subcategoría, la prioridad ni el respondedor.
- No inventes rutas de evacuación, salidas, "la zona más cercana", distancias ni tiempos. Puedes recomendar alejarse
  del peligro y seguir la señalética y las indicaciones del personal o de las autoridades.
- Las secciones "Contexto operativo" de los protocolos describen lo que hace el personal: no las ofrezcas como hecho."""

SYSTEM_PROMPT_CHAT = f"""Eres ORION, asistente de emergencias de la sede. Conversas con la persona que reportó un incidente.
El reporte ya quedó registrado y la persona puede ver su seguimiento en la app. No le pidas que repita la ubicación
ni el tipo de emergencia.

{REGLAS_ORION}

CÓMO RESPONDER:
1. No escribas el aviso legal ni el estado del reporte: el sistema los agrega solo.
2. Usa ÚNICAMENTE la información de los PROTOCOLOS entregados y de las REGLAS (incluidos los contactos de la sede).
   Puedes combinar varios protocolos si el relato lo requiere.
3. Si la situación no está cubierta, dilo ("No cuento con información suficiente para esa situación.") y pide que
   se mantenga en un lugar seguro.
4. Lenguaje claro, directo y breve (máximo 4 oraciones), en segunda persona (tú). Resume con tus palabras: no copies
   listas del protocolo ni frases en infinitivo ("Anotar…", "No usar…"), y no menciones códigos como SAL-001.
   Haz como máximo 1 o 2 preguntas por mensaje. Cuando corresponda, termina con la pregunta decisiva del protocolo:
   la que confirma o descarta la señal más grave (si responde, si respira, si hay heridos, armas o personas expuestas,
   si quiere hacerse daño), sin repetir lo que la persona ya contestó.
5. Quién es el afectado casi siempre está claro en el mensaje: "mi compañero", "un señor", "hay alguien" es otra
   persona; "me duele", "estoy mareada", "me robaron" es quien escribe. Entonces NO preguntes quién es el afectado.
   Pregúntalo solo en una emergencia de salud cuando el mensaje no lo dice, y nunca como primera frase: primero va
   la indicación más urgente. No lo preguntes en robos, peleas, amenazas, objetos sospechosos, incendios, sismos ni
   fallas de infraestructura. Si quien escribe es el afectado, háblale directamente a él o ella.
6. Pide alejarse solo cuando hay un peligro en el entorno (fuego, humo, gas, electricidad, una estructura dañada,
   un arma o un agresor presente). Si la persona afectada necesita ayuda y ese peligro no existe (malestar,
   convulsión, dolor de pecho, crisis emocional, ideas de hacerse daño), pide acompañarla sin dejarla sola, siempre
   que sea seguro. Nunca pidas sentar ni levantar a alguien que no responde. Ante una amenaza, no ordenes evacuar:
   eso lo decide el personal.
7. Decide riesgo_vital con el "CRITERIO GENERAL DE RIESGO VITAL" del MARCO GENERAL. Las señales de cada protocolo
   son ejemplos, no una lista cerrada: cualquier riesgo de muerte o de secuela grave (D.S. 34/2022) cuenta, aunque no
   aparezca en ninguna lista. Una pelea, un sismo, una fractura cerrada o un delito, por sí solos, no son riesgo vital.
   Si una señal es probable pero no está descrita, deja riesgo_vital = false y pregunta por ella en este mismo mensaje.
   Si el usuario confirma una señal por la que preguntaste, o si quien escribe es el afectado y describe una señal,
   esa señal ya está descrita: riesgo_vital = true.
8. Si el mensaje no tiene relación con la emergencia, redirige amablemente al tema.

FORMATO DE SALIDA: responde SOLO con un objeto JSON válido, sin texto adicional ni bloques de código:
{{"respuesta_texto": "texto para la persona", "riesgo_vital": false, "protocolo": "SAL-001"}}
"protocolo" es el código del protocolo que más usaste, o null."""

SYSTEM_PROMPT_RIESGO = """Eres el detector de riesgo vital de ORION, un sistema de emergencias de una sede universitaria.
NO respondes al usuario: solo clasificas la conversación según el CRITERIO entregado.

Cómo decidir:
- Marca riesgo_vital = true si en la conversación hay AL MENOS UNA señal concreta de riesgo de muerte o de secuela
  grave (o una amenaza del entorno que pueda causarlo) que no haya sido descartada después.
- Basta con que se describa con duda ("creo que", "parece que"). También cuenta si el afectado es quien escribe.
- Las señales pueden estar repartidas en varios mensajes: una respuesta corta del usuario ("sí", "no", "sigue igual")
  confirma o niega lo que el asistente le preguntó justo antes.
- NO es riesgo vital: solo el tipo de incidente sin una señal concreta ("hay una pelea", "tembló", "se cayó"), una
  señal que el usuario descartó ("respira normal", "no hay armas") o que ya pasó y la persona está bien.
- Ante una duda real entre true y false, elige true: un caso grave no detectado es mucho peor que una falsa alarma.

Responde SOLO con un objeto JSON válido, sin texto adicional:
{"riesgo_vital": true, "senal": "cita breve de la señal, o null si no hay"}"""


SYSTEM_PROMPT_RESUMEN = f"""Eres el módulo de resumen de ORION. Recibes la conversación entre la persona que reportó un
incidente (usuario) y el asistente (bot). El resumen lo leerá el Respondedor asignado antes de atender.

{REGLAS_ORION}

QUÉ DEBES ENTREGAR:
- resumen_ia: 1 a 3 oraciones con los antecedentes NUEVOS o cambios que aportó el usuario durante la conversación,
  complementando el reporte (subcategoría y ubicación) sin repetirlo. Lo que el usuario solo mencionó o supone,
  preséntalo como dicho por él ("el reportante indica…"), no como hecho confirmado. Si la conversación no aporta
  antecedentes nuevos, escribe: "La conversación no aporta antecedentes adicionales al reporte."
- datos_clave.puntos_criticos: frases cortas con datos concretos que pueden influir en la atención (síntomas,
  personas afectadas, movilidad, armas, cambios de ubicación). No repitas el resumen completo. [] si no hay.
- datos_clave.orientaciones_entregadas: solo indicaciones que el BOT realmente dio, redactadas como
  "El chatbot indicó …". No agregues recomendaciones nuevas. [] si el bot no dio indicaciones.
- No inventes síntomas, diagnósticos, traslados, materiales ni gestiones. No opines sobre prioridad ni estado.

FORMATO DE SALIDA: responde SOLO con un objeto JSON válido, sin texto adicional ni bloques de código:
{{"resumen_ia": "…", "datos_clave": {{"puntos_criticos": ["…"], "orientaciones_entregadas": ["…"]}}}}"""


# ---------- Índice local de protocolos ----------

def _ruta_protocolos() -> Path:
    if PROTOCOLOS_PATH:
        return Path(PROTOCOLOS_PATH)
    return Path(__file__).resolve().parent.parent / "docs" / "protocolos_completos.txt"


def dividir_protocolos(documento: str) -> list[tuple[str, str]]:
    """Un bloque por código (SAL-001…SEG-004); une trozos con el mismo código."""
    partes = [p.strip() for p in re.split(r"(?=^(?:GEN|SAL|INF|SEG)-\d+ — )", documento, flags=re.MULTILINE) if p.strip()]
    protocolos: dict[str, str] = {}
    for i, texto in enumerate(partes, 1):
        match = re.match(r"((?:GEN|SAL|INF|SEG)-\d+)", texto)
        codigo = match.group(1) if match else f"PROTO-{i}"
        protocolos[codigo] = f"{protocolos[codigo]}\n\n{texto}" if codigo in protocolos else texto
    return list(protocolos.items())


def generar_vector(texto: str) -> list[float]:
    """Convierte texto a vector semántico con Titan Embeddings v2 (normalizado)."""
    respuesta = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": texto.strip()[:40000], "dimensions": EMBEDDING_DIM, "normalize": True}),
    )
    return json.loads(respuesta["body"].read())["embedding"]


_indice: list[tuple[str, str, list[float]]] | None = None
_lock = threading.Lock()


def cargar_indice() -> list[tuple[str, str, list[float]]]:
    """Arma el índice una vez por proceso. Si Bedrock falla, se reintenta en la siguiente petición."""
    global _indice
    if _indice is None:
        with _lock:
            if _indice is None:
                _indice = [(codigo, texto, generar_vector(texto)) for codigo, texto in protocolos_especificos()]
    return _indice


def _todos() -> dict[str, str]:
    return dict(dividir_protocolos(_ruta_protocolos().read_text(encoding="utf-8")))


def marco_general() -> str:
    """GEN-000: siempre va en el prompt. Si falta, el servicio no debe responder."""
    marco = _todos().get("GEN-000")
    if not marco:
        raise ValueError("Falta GEN-000 en el archivo de protocolos")
    return marco


def protocolos_especificos() -> list[tuple[str, str]]:
    """Los 12 protocolos que participan en la búsqueda (todo menos GEN-*)."""
    return [(c, t) for c, t in _todos().items() if not c.startswith("GEN-")]


def protocolos_cargados() -> int:
    return len(protocolos_especificos())


def _similitud(a: list[float], b: list[float]) -> float:
    # Los vectores de Titan vienen normalizados: el producto punto es la similitud coseno
    return sum(x * y for x, y in zip(a, b))


def buscar_contexto(consulta: str, k: int = TOP_K_RESULTS) -> list[tuple[str, str]]:
    """Busca en TODOS los protocolos, sin filtrar por subcategoría."""
    vector = generar_vector(consulta)
    ranking = sorted(cargar_indice(), key=lambda p: _similitud(vector, p[2]), reverse=True)
    return [(codigo, texto) for codigo, texto, _ in ranking[:k]]


# ---------- Historial ----------

def sin_frases_fijas(texto: str) -> str:
    """Quita el aviso y la frase de "Recibido" que agrega el código, para que el modelo no las copie al responder.
    La de En_camino se deja: con_aviso la busca en el historial para decirla una sola vez."""
    for frase in (AVISO, MENSAJES_ESTADO["Recibido"]):
        texto = texto.replace(frase, " ")
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_historial(historial: list[dict]) -> list[dict]:
    """Convierte [{rol, texto}] al formato Converse: empieza con user y alterna roles."""
    mensajes: list[dict] = []
    for m in historial[-MAX_HISTORIAL:]:
        texto = (m.get("texto") or "").strip()
        rol = "assistant" if m.get("rol") == "bot" else "user"
        if rol == "assistant":
            texto = sin_frases_fijas(texto) or "(orientación entregada)"
        if not texto:
            continue
        if mensajes and mensajes[-1]["role"] == rol:
            mensajes[-1]["content"][0]["text"] += "\n" + texto
        else:
            mensajes.append({"role": rol, "content": [{"text": texto}]})
    while mensajes and mensajes[0]["role"] != "user":
        mensajes.pop(0)  # Converse exige empezar con user (el saludo del bot va como contexto)
    # El backend guarda solo el texto del bot; se reenvía como JSON para que el modelo no copie el formato y
    # deje de responder en JSON desde el segundo turno.
    for m in mensajes:
        if m["role"] == "assistant":
            m["content"][0]["text"] = json.dumps({"respuesta_texto": m["content"][0]["text"]}, ensure_ascii=False)
    return mensajes


def _historial_plano(mensajes: list[dict]) -> list[dict]:
    """Deshace el JSON con que se reenvían los turnos del bot (para el detector)."""
    salida = []
    for m in mensajes:
        texto = m["content"][0]["text"]
        if m["role"] == "assistant":
            try:
                texto = json.loads(texto).get("respuesta_texto", texto)
            except (json.JSONDecodeError, AttributeError):
                pass
        salida.append({"role": m["role"], "content": [{"text": texto}]})
    return salida


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


# ---------- Llamadas al modelo ----------

def _parsear_json(texto: str) -> dict | None:
    """Primer objeto JSON válido del texto. Tolera texto o un segundo objeto antes o después (pasa en
    conversaciones de varios turnos, donde el modelo a veces imita el formato del historial)."""
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    decodificador, objetos, i = json.JSONDecoder(), [], texto.find("{")
    while i != -1:
        try:
            datos, fin = decodificador.raw_decode(texto, i)
        except json.JSONDecodeError:
            i = texto.find("{", i + 1)
            continue
        if isinstance(datos, dict):
            objetos.append(datos)
        i = texto.find("{", fin)
    for datos in objetos:  # se prefiere el objeto completo (con riesgo_vital o resumen_ia)
        if "riesgo_vital" in datos or "resumen_ia" in datos:
            return datos
    return objetos[0] if objetos else None


_CAMPO_TEXTO = re.compile(r'"respuesta_texto"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_CAMPO_RIESGO = re.compile(r'"riesgo_vital"\s*:\s*"?(true|false)', re.IGNORECASE)
_CAMPO_PROTOCOLO = re.compile(r'"protocolo"\s*:\s*"([^"]*)"')


def _rescatar_campos(texto: str) -> dict | None:
    """Último recurso cuando el JSON viene roto (comillas sin escapar, cortado): saca los campos con regex."""
    m = _CAMPO_TEXTO.search(texto)
    if not m:
        return None
    try:
        respuesta = json.loads(f'"{m.group(1)}"')
    except json.JSONDecodeError:
        respuesta = m.group(1)
    riesgo, protocolo = _CAMPO_RIESGO.search(texto), _CAMPO_PROTOCOLO.search(texto)
    return {"respuesta_texto": respuesta, "riesgo_vital": bool(riesgo) and riesgo.group(1).lower() == "true",
            "protocolo": protocolo.group(1) if protocolo else None}


def _a_bool(valor) -> bool:
    if isinstance(valor, str):
        return valor.strip().lower() in ("true", "si", "sí", "1")
    return bool(valor)


def _converse(system: str, mensajes: list[dict]) -> tuple[str, dict]:
    resp = bedrock.converse(
        modelId=LLM_MODEL,
        system=[{"text": system}],
        messages=mensajes,
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
    )
    uso = {**resp.get("usage", {}), "stop_reason": resp.get("stopReason")}
    return resp["output"]["message"]["content"][0]["text"], uso


def responder(historial: list[dict], contexto: dict | None = None) -> dict:
    """historial: [{rol, texto}] terminando en el mensaje actual del usuario.
    Devuelve respuesta_texto, riesgo_vital, protocolo y metadata. Lanza excepción si Bedrock falla o el modelo
    no entrega JSON válido (el backend aplica su fallback)."""
    contexto = contexto or {}
    mensajes = normalizar_historial(historial)
    if not mensajes or mensajes[-1]["role"] != "user":
        raise ValueError("El último mensaje del historial debe ser del usuario")

    senales = detectar_senales(mensajes[-1]["content"][0]["text"])
    try:
        return _responder_con_modelo(mensajes, contexto, senales)
    except Exception:
        if not senales:
            raise
        # Si Bedrock falla justo cuando hay una señal explícita, no se pierde la detección: se responde con una
        # indicación fija y riesgo_vital = true, en vez de dejar que el fallback de NestJS lo marque false.
        return {
            "respuesta_texto": con_aviso(RESPUESTA_SEGURA, mensajes, contexto.get("estado_incidente")),
            "riesgo_vital": True,
            "protocolo": None,
            "metadata": {"protocolos_recuperados": [], "senales_regla": senales, "riesgo_vital_modelo": None,
                         "respuesta_segura": True},
        }


def _criterio_riesgo() -> str:
    """Solo la sección 3 de GEN-000 (criterio de riesgo vital): prompt corto y barato para el detector."""
    marco = marco_general()
    m = re.search(r"^3\.1 .*?(?=^4\.|\Z)", marco, flags=re.MULTILINE | re.DOTALL)
    return m.group(0).strip() if m else marco


def detectar_riesgo_modelo(mensajes: list[dict]) -> bool | None:
    """Segunda opinión del modelo, solo sobre riesgo vital. Devuelve None si falla (no bloquea la respuesta).
    La conversación va como transcripción en un solo mensaje, así el formato del historial no influye."""
    lineas = [f"{'USUARIO' if m['role'] == 'user' else 'ASISTENTE'}: {m['content'][0]['text']}" for m in mensajes]
    pedido = f"CRITERIO (GEN-000, sección 3):\n{_criterio_riesgo()}\n\nCONVERSACIÓN:\n" + "\n".join(lineas)
    try:
        salida, _ = _converse(SYSTEM_PROMPT_RIESGO, [{"role": "user", "content": [{"text": pedido}]}])
    except Exception as e:  # noqa: BLE001
        log.warning("detector de riesgo falló: %s", e)
        return None
    datos = _parsear_json(salida)
    if datos and "riesgo_vital" in datos:
        return _a_bool(datos["riesgo_vital"])
    m = _CAMPO_RIESGO.search(salida)
    return m.group(1).lower() == "true" if m else None


_hilos = ThreadPoolExecutor(max_workers=8)


RESPUESTA_SEGURA = (f"Mantente en un lugar seguro y, si puedes hacerlo sin riesgo, quédate cerca de la persona "
                    f"afectada. Si necesitas contacto directo: {CONTACTOS_SEDE}. Cuéntame qué está pasando ahora.")


RECORDATORIO_JSON = ("\n\nFORMATO OBLIGATORIO: responde con UN solo objeto JSON válido con las claves respuesta_texto, "
                     "riesgo_vital y protocolo, sin texto antes ni después. Escapa las comillas dentro de los textos.")


def _responder_con_modelo(mensajes: list[dict], contexto: dict, senales: list[str]) -> dict:
    protocolos = buscar_contexto(consulta_para_busqueda(mensajes, contexto))
    textos = "\n\n---\n\n".join(f"[{codigo}]\n{texto}" for codigo, texto in protocolos) or "(ninguno)"
    alerta = ""
    if senales:
        alerta = ("\n\nALERTA DEL SISTEMA: el último mensaje contiene señales explícitas de riesgo vital "
                  f"({', '.join(senales)}). riesgo_vital ya quedó en true. Entrega primero la indicación de seguridad "
                  "más urgente.")
    system = (f"{SYSTEM_PROMPT_CHAT}\n\nMARCO GENERAL (aplica siempre):\n{marco_general()}"
              f"\n\nINCIDENTE REPORTADO:\n{texto_contexto(contexto)}\n\nPROTOCOLOS DISPONIBLES:\n{textos}{alerta}")

    # El historial que ve el detector usa el texto plano del bot, no el JSON que se reenvía al modelo principal
    detector = _hilos.submit(detectar_riesgo_modelo, _historial_plano(mensajes)) if DETECTOR_RIESGO else None
    datos, uso, intentos = None, {}, []
    for intento in range(2):
        # Segundo intento solo si el primero no trae JSON utilizable: se recuerda el formato al final del prompt
        extra = "" if intento == 0 else RECORDATORIO_JSON
        salida, uso = _converse(system + extra, mensajes)
        datos = _parsear_json(salida)
        if not datos or not str(datos.get("respuesta_texto", "")).strip():
            datos = _rescatar_campos(salida)
        if not datos and salida.strip() and "{" not in salida:
            # Texto plano en vez de JSON: se usa igual; riesgo_vital queda solo con las reglas
            log.warning("respuesta del modelo sin JSON (%d caracteres); se usa como texto", len(salida))
            datos = {"respuesta_texto": salida.strip(), "riesgo_vital": False, "protocolo": None}
        utilizable = datos and str(datos.get("respuesta_texto", "")).strip()
        # Sin riesgo_vital el modelo imitó el formato del historial ({"respuesta_texto": ...}): se reintenta
        # para no perder la detección; en el segundo intento se acepta lo que haya
        if utilizable and ("riesgo_vital" in datos or intento == 1):
            break
        if utilizable:
            intentos.append(f"[sin riesgo_vital] {salida[:300]!r}")
            continue
        intentos.append(f"[{uso.get('stop_reason')}] {salida[:300]!r}")
        log.warning("respuesta del modelo no es JSON válido (intento %d, %s): %r",
                    intento + 1, uso.get("stop_reason"), salida[:500])
    if not (datos and str(datos.get("respuesta_texto", "")).strip()):
        raise ValueError(f"Respuesta del modelo no es JSON válido tras 2 intentos: {' | '.join(intentos)}")

    protocolo = datos.get("protocolo")
    riesgo_modelo = _a_bool(datos.get("riesgo_vital", False))
    riesgo_detector = detector.result() if detector else None
    return {
        "respuesta_texto": con_aviso(
            con_contacto(quitar_promesas(str(datos["respuesta_texto"]).strip(), contexto.get("estado_incidente")),
                         contacto_pedido(mensajes[-1]["content"][0]["text"], contexto.get("categoria"))),
            mensajes, contexto.get("estado_incidente")),
        # Reglas y detector solo pueden subir a true; nunca bajan lo que decidió el modelo
        "riesgo_vital": riesgo_modelo or bool(riesgo_detector) or bool(senales),
        "protocolo": str(protocolo) if protocolo else None,
        "metadata": {"protocolos_recuperados": [c for c, _ in protocolos], "senales_regla": senales,
                     "riesgo_vital_modelo": riesgo_modelo, "riesgo_vital_detector": riesgo_detector, "tokens": uso,
                     "reintento_json": bool(intentos)},
    }


# ---------- Post-proceso determinista de la respuesta ----------

_PROMESAS = re.compile(
    r"\b(?:ya\s+)?(?:fue|fueron|ha sido|han sido|est[aá]n?)\s+(?:notificad|avisad|alertad)\w*|"
    r"\b(?:llam[eé]|llamar[eé]|llamamos|voy a llamar|avis[eé]|notifiqu[eé])\b|"
    r"\b(?:avisaremos|notificaremos|alertaremos|llamaremos|vamos a (?:avisar|notificar|llamar))\b|"
    r"\b(?:en|dentro de)\s+(?:unos?\s+)?\d+\s*(?:min|minutos|segundos)\b|tiempo estimado|llegar[áa]n? en", re.IGNORECASE)
_EN_CAMINO = re.compile(r"\b(?:va|van|viene|vienen|est[aá]n?)\s+(?:ya\s+)?en\s+camino\b", re.IGNORECASE)


_TEL = r"(?:tel[eé]fonos?|n[uú]meros?|fonos?|anexos?|contactos?)"
_PIDE_TELEFONO = re.compile(  # pedir un teléfono, no solo mencionarlo ("no contesta el teléfono", "me robaron el celular")
    r"\b(?:dame|deme|me\s+das|me\s+da|me\s+pasas|p[aá]same|me\s+puedes\s+dar|me\s+podr[ií]as\s+dar|tienes|tiene|"
    r"necesito|quiero|hay)\s+(?:el|la|los|un|una|alg[uú]n)?\s*" + _TEL + r"|"
    r"\b(?:cu[aá]l|qu[eé])\s+(?:es\s+)?(?:el\s+)?" + _TEL + r"|"
    r"\b" + _TEL + r"\s+(?:de|del|para)\s+(?:la\s+)?(?:seguridad|salud|emergencia|la\s+u\b|u\b|universidad|sede|"
    r"mantenci|mantenimiento|guardia|policl)|"
    r"\ba\s+(?:qui[eé]n|d[oó]nde|qu[eé]\s+n[uú]mero)\s+(?:llamo|puedo\s+llamar|tengo\s+que\s+llamar|debo\s+llamar|"
    r"llamar|aviso)\b|\bc[oó]mo\s+(?:los\s+|las\s+|lo\s+|la\s+)?(?:llamo|contacto)\b", re.IGNORECASE)
_PIDE_SALUD = re.compile(r"\b(?:salud|policl[ií]nico|m[eé]dic\w*|enfermer\w*|param[eé]dic\w*)\b", re.IGNORECASE)
_PIDE_SEGURIDAD = re.compile(r"\b(?:seguridad|guardias?|vigilancia)\b", re.IGNORECASE)
_PIDE_INFRA = re.compile(r"\b(?:mantenci[oó]n|mantenimiento|infraestructura|el[eé]ctric\w*|t[eé]cnico)\b", re.IGNORECASE)
_NIEGA_CONTACTO = re.compile(
    r"\bno\s+(?:te\s+|le\s+)?(?:puedo|podemos|tengo|cuento con)\b[^.?!]*\b(?:tel[eé]fono|n[uú]mero|contacto)|"
    r"\bno\s+(?:es\s+necesario|hace\s+falta|necesitas)\s+(?:que\s+)?llam|\bno\s+llames\b", re.IGNORECASE)
_NUMERO_SEDE = re.compile(r"227\s*7\s*(?:888|999)|\banexo\s*999\b", re.IGNORECASE)


def contacto_pedido(mensaje_usuario: str, categoria: str | None) -> tuple[str, ...]:
    """Si el usuario pide un teléfono, devuelve los contactos que corresponden (GEN-000 sección 7).
    Primero manda lo que nombra el usuario (salud, seguridad, mantención); si no nombra nada, la categoría."""
    if not _PIDE_TELEFONO.search(mensaje_usuario or ""):
        return ()
    if _PIDE_SALUD.search(mensaje_usuario):
        return (CONTACTO_SALUD,)
    if _PIDE_SEGURIDAD.search(mensaje_usuario):
        return (CONTACTO_SEGURIDAD,)
    if _PIDE_INFRA.search(mensaje_usuario) or (categoria or "").lower().startswith("infra"):
        return (CONTACTO_INFRAESTRUCTURA, CONTACTO_SEGURIDAD)
    if (categoria or "").lower() == "salud":
        return (CONTACTO_SALUD,)
    return (CONTACTO_SEGURIDAD,)


def con_contacto(texto: str, contactos: tuple[str, ...]) -> str:
    """Agrega los contactos pedidos con texto fijo. Quita las oraciones del modelo que niegan el contacto o que
    traen un número de la sede (para no duplicarlo ni dejar una versión distinta)."""
    if not contactos:
        return texto
    oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
    quedan = [o for o in oraciones if not _NIEGA_CONTACTO.search(o) and not _NUMERO_SEDE.search(o)]
    if len(quedan) != len(oraciones):
        log.warning("se reemplazaron %d oraciones sobre contactos por el contacto fijo", len(oraciones) - len(quedan))
    return " ".join([*quedan, *contactos]).strip()


def quitar_promesas(texto: str, estado_incidente: str | None) -> str:
    """Borra las oraciones que afirman notificaciones, llegadas o tiempos (regla que el modelo a veces rompe).
    'En camino' solo se permite cuando el incidente está En_camino."""
    oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
    quedan = [o for o in oraciones if not _PROMESAS.search(o)
              and (estado_incidente == "En_camino" or not _EN_CAMINO.search(o))]
    if len(quedan) != len(oraciones):
        log.warning("se quitaron %d oraciones con promesas de la respuesta", len(oraciones) - len(quedan))
    return " ".join(quedan) or "Mantente en un lugar seguro y avísame por este chat cualquier cambio."


def es_primera_respuesta(mensajes: list[dict]) -> bool:
    return not any(m["role"] == "assistant" for m in mensajes)


_AVISO_DEL_MODELO = re.compile(r"ORION es una gu[ií]a de apoyo\.\s*No sustituye[^.]*\.\s*", re.IGNORECASE)


def con_aviso(texto: str, mensajes: list[dict], estado_incidente: str | None = None) -> str:
    """Frases fijas agregadas por código (no dependen del modelo):
    - primera respuesta: aviso legal al inicio y el estado real del reporte;
    - si el incidente pasó a En_camino y aún no se dijo en la conversación, se informa una vez."""
    texto = _AVISO_DEL_MODELO.sub("", texto).strip() or texto  # el aviso lo pone solo el código
    estado = MENSAJES_ESTADO.get(estado_incidente or "")
    dicho = any(estado and estado in m["content"][0]["text"] for m in mensajes if m["role"] == "assistant")
    if es_primera_respuesta(mensajes):
        partes = [AVISO] if AVISO not in texto else []
        partes.append(texto)
        if estado and estado not in texto:
            partes.append(estado)
        return " ".join(partes)
    if estado_incidente == "En_camino" and estado and not dicho and estado not in texto:
        return f"{texto} {estado}"
    return texto


def _lista_de_textos(valor) -> list[str]:
    if not isinstance(valor, list):
        return []
    return [str(v).strip() for v in valor if isinstance(v, (str, int, float)) and str(v).strip()]


def resumir(historial: list[dict], contexto: dict | None = None) -> dict:
    """historial: conversación completa [{rol, texto}]. Devuelve resumen_ia y datos_clave."""
    contexto = contexto or {}
    lineas = [f"{'USUARIO' if m.get('rol') == 'usuario' else 'BOT'}: {(m.get('texto') or '').strip()}"
              for m in historial if (m.get("texto") or "").strip()]
    if not any(l.startswith("USUARIO:") for l in lineas):
        raise ValueError("El historial no tiene mensajes del usuario")

    pedido = (f"REPORTE ORIGINAL:\n{texto_contexto(contexto)}\n\nCONVERSACIÓN:\n" + "\n".join(lineas)
              + "\n\nGenera el JSON pedido.")
    salida, _ = _converse(SYSTEM_PROMPT_RESUMEN, [{"role": "user", "content": [{"text": pedido}]}])
    datos = _parsear_json(salida)
    if not datos or not str(datos.get("resumen_ia", "")).strip():
        raise ValueError(f"Resumen del modelo no es JSON válido ({len(salida)} caracteres)")

    clave = datos.get("datos_clave") if isinstance(datos.get("datos_clave"), dict) else {}
    return {
        "resumen_ia": str(datos["resumen_ia"]).strip(),
        "datos_clave": {
            "puntos_criticos": _lista_de_textos(clave.get("puntos_criticos")),
            "orientaciones_entregadas": _lista_de_textos(clave.get("orientaciones_entregadas")),
        },
    }
