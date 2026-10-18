"""Pruebas del contrato NestJS -> FastAPI sin AWS (Bedrock simulado).
Uso (desde la raíz del repo):  pip install -r src/requirements.txt pytest httpx  &&  python -m pytest -q tests
"""
import io
import json
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
os.environ["INTERNAL_TOKEN"] = "token-prueba"
os.environ["CHATBOT_MODO"] = "rag"
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-2")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import rag  # noqa: E402
from riesgo import detectar_senales  # noqa: E402

HEADERS = {"X-Internal-Token": "token-prueba", "X-Request-Id": "5f0c1c3e-7b8a-4b8e-9a51-2f0d6c1f4a10"}

# Payload exacto de ORION API 1.0 — Chatbot (17-10-2026), sección "Qué le envía el backend al chatbot"
CHAT = {
    "request_id": "5f0c1c3e-7b8a-4b8e-9a51-2f0d6c1f4a10",
    "id_incidente": 33,
    "id_conversacion": 12,
    "mensaje": "Hay una pelea afuera del edificio A y una persona está sangrando del brazo",
    "historial": [
        {"rol": "bot", "texto": "Hola, soy el asistente de ORION. ¿Qué está pasando ahora?"},
        {"rol": "usuario", "texto": "Hay gente gritando en el segundo piso"},
    ],
    "contexto": {
        "rol_usuario": "Reportante",
        "estado_incidente": "Recibido",
        "categoria": {"id_categoria": 3, "nombre": "Seguridad"},
        "subcategoria": {"id_subcategoria": 9, "nombre": "Violencia interpersonal"},
        "ubicacion_incidente": {"id_ubicacion": 1, "nombre": "Edificio A", "latitud": -33.03679, "longitud": -71.48688},
        "ubicacion_actual": None,
    },
}

# Payload de /v1/resumen (mismo documento, sección "ENVÍA — NestJS → FastAPI: POST /v1/resumen")
RESUMEN = {
    "request_id": "5f0c1c3e-7b8a-4b8e-9a51-2f0d6c1f4a10",
    "id_incidente": 33,
    "id_conversacion": 12,
    "contexto": {
        "estado_incidente": "En_camino",
        "subcategoria": {"id_subcategoria": 9, "nombre": "Violencia interpersonal"},
        "ubicacion_incidente": {"id_ubicacion": 1, "nombre": "Edificio A"},
    },
    "historial": [
        {"rol": "usuario", "texto": "Ahora no responde"},
        {"rol": "bot", "texto": "Indica si la persona respira."},
    ],
}


class BedrockFalso:
    """Simula invoke_model (embeddings) y converse (respuesta del LLM)."""

    def __init__(self, salida_llm: str):
        self.salida_llm = salida_llm
        self.systems: list[str] = []

    def invoke_model(self, modelId, body):
        texto = json.loads(body)["inputText"]
        vector = [0.0] * 1024
        vector[hash(texto[:12]) % 1024] = 1.0
        return {"body": io.BytesIO(json.dumps({"embedding": vector}).encode())}

    def converse(self, modelId, system, messages, inferenceConfig):
        self.systems.append(system[0]["text"])
        return {"output": {"message": {"content": [{"text": self.salida_llm}]}}, "usage": {"inputTokens": 1}}


@pytest.fixture
def cliente(monkeypatch):
    def _crear(salida_llm: str):
        falso = BedrockFalso(salida_llm)
        monkeypatch.setattr(rag, "bedrock", falso)
        monkeypatch.setattr(rag, "_indice", None)
        return TestClient(app_module.app), falso
    return _crear


def test_health(cliente):
    c, _ = cliente("{}")
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["protocolos"] == 12


def test_chat_sin_token_401(cliente):
    c, _ = cliente("{}")
    assert c.post("/v1/chat", json=CHAT).status_code == 401


def test_chat_contrato_nestjs(cliente):
    c, falso = cliente('{"respuesta_texto": "Aléjate y no intervengas. ¿La persona está consciente?", '
                       '"riesgo_vital": false, "protocolo": "SEG-001"}')
    r = c.post("/v1/chat", json=CHAT, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == CHAT["request_id"]
    assert body["respuesta_texto"].strip()
    assert body["riesgo_vital"] is False
    assert body["protocolo"] == "SEG-001"
    assert set(body) == {"request_id", "respuesta_texto", "riesgo_vital", "modelo", "protocolo", "latencia_ms"}
    assert "131 SAMU" not in falso.systems[0]
    assert "ya fue notificado" not in falso.systems[0]


def test_chat_riesgo_vital_true(cliente):
    c, _ = cliente('```json\n{"respuesta_texto": "Indica si respira.", "riesgo_vital": "true", "protocolo": null}\n```')
    body = c.post("/v1/chat", json=CHAT, headers=HEADERS).json()
    assert body["riesgo_vital"] is True
    assert body["protocolo"] is None


def test_chat_en_camino_y_ubicacion_sin_coordenadas(cliente):
    c, falso = cliente('{"respuesta_texto": "Un respondedor ya va en camino.", "riesgo_vital": false}')
    payload = json.loads(json.dumps(CHAT))
    payload["contexto"]["estado_incidente"] = "En_camino"
    payload["contexto"]["ubicacion_incidente"] = {"id_ubicacion": 1, "nombre": "Edificio A"}
    payload["contexto"]["ubicacion_actual"] = {"id_ubicacion": None, "nombre": None, "latitud": -33.0, "longitud": -71.4}
    r = c.post("/v1/chat", json=payload, headers=HEADERS)
    assert r.status_code == 200
    assert "En_camino" in falso.systems[0]
    assert "el usuario ahora está en: -33.0, -71.4" in falso.systems[0]


def test_chat_json_invalido_503(cliente):
    c, _ = cliente("Lo siento, no puedo responder en JSON")
    assert c.post("/v1/chat", json=CHAT, headers=HEADERS).status_code == 503


def test_chat_body_invalido_422(cliente):
    c, _ = cliente("{}")
    malo = {**CHAT, "historial": [{"emisor": "USUARIO", "contenido": "contrato viejo"}]}
    assert c.post("/v1/chat", json=malo, headers=HEADERS).status_code == 422


def test_resumen_contrato_nestjs(cliente):
    c, _ = cliente('{"resumen_ia": "El reportante indica que una persona no responde.", "datos_clave": '
                   '{"puntos_criticos": ["Persona que no responde"], "orientaciones_entregadas": '
                   '["El chatbot indicó verificar si la persona respira"]}}')
    r = c.post("/v1/resumen", json=RESUMEN, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"request_id", "resumen_ia", "datos_clave"}
    assert body["request_id"] == RESUMEN["request_id"]
    assert body["datos_clave"] == {"puntos_criticos": ["Persona que no responde"],
                                   "orientaciones_entregadas": ["El chatbot indicó verificar si la persona respira"]}
    assert "riesgo_vital" not in body


def test_resumen_arreglos_vacios_o_malos(cliente):
    c, _ = cliente('{"resumen_ia": "La conversación no aporta antecedentes adicionales al reporte.", '
                   '"datos_clave": {"puntos_criticos": "no es lista"}}')
    body = c.post("/v1/resumen", json=RESUMEN, headers=HEADERS).json()
    assert body["datos_clave"] == {"puntos_criticos": [], "orientaciones_entregadas": []}


def test_resumen_sin_mensajes_usuario_422(cliente):
    c, _ = cliente("{}")
    solo_bot = {**RESUMEN, "historial": [{"rol": "bot", "texto": "Hola"}]}
    assert c.post("/v1/resumen", json=solo_bot, headers=HEADERS).status_code == 422


def test_resumen_sin_token_401(cliente):
    c, _ = cliente("{}")
    assert c.post("/v1/resumen", json=RESUMEN).status_code == 401


def test_protocolos_sin_acciones_prohibidas():
    texto = (SRC.parent / "docs" / "protocolos_completos.txt").read_text(encoding="utf-8")
    codigos = [c for c, _ in rag.dividir_protocolos(texto)]
    assert codigos == ["GEN-000", "SAL-001", "SAL-002", "SAL-003", "SAL-004", "INF-001", "INF-002", "INF-003", "INF-004",
                       "SEG-001", "SEG-002", "SEG-003", "SEG-004"]
    for frase in ("ya ha sido notificado", "ya fue notificado", "tiempo estimado de llegada", "elevar a Alta",
                  "elevar a Media", "llamar a Carabineros", "llamar al Cuerpo de Bomberos", "Acciones que el sistema",
                  "llamar al 13", "llame al 13", "marcar 13"):
        assert frase.lower() not in texto.lower(), frase


CASOS = json.loads((SRC.parent / "docs" / "casos_riesgo_vital.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("caso", CASOS, ids=lambda c: f"caso{c['id']}")
def test_regla_riesgo_vital(caso):
    """La red de seguridad se activa exactamente en los casos marcados, incluidas las negaciones."""
    assert bool(detectar_senales(caso["mensaje"])) == caso["debe_activar_regla"], detectar_senales(caso["mensaje"])


def test_marco_general_siempre_en_el_prompt(cliente):
    c, falso = cliente('{"respuesta_texto": "Aléjate.", "riesgo_vital": false}')
    c.post("/v1/chat", json=CHAT, headers=HEADERS)
    assert "CRITERIO GENERAL DE RIESGO VITAL" in falso.systems[0]
    assert falso.systems[0].count("GEN-000 — MARCO GENERAL") == 1  # no se duplica por la búsqueda


def test_regla_sube_riesgo_si_el_modelo_no_lo_marca(cliente):
    c, falso = cliente('{"respuesta_texto": "Aléjate.", "riesgo_vital": false}')
    payload = {**CHAT, "mensaje": "Uno de los que pelean sacó un arma"}
    body = c.post("/v1/chat", json=payload, headers=HEADERS).json()
    assert body["riesgo_vital"] is True
    assert "ALERTA DEL SISTEMA" in falso.systems[0]


def test_regla_no_baja_lo_que_marca_el_modelo(cliente):
    c, _ = cliente('{"respuesta_texto": "Quédate con ella.", "riesgo_vital": true}')
    payload = {**CHAT, "mensaje": "Le duele mucho el pecho y suda frío"}
    assert c.post("/v1/chat", json=payload, headers=HEADERS).json()["riesgo_vital"] is True


def test_bedrock_cae_con_senal_explicita_responde_seguro(cliente):
    c, _ = cliente("no es json")
    payload = {**CHAT, "mensaje": "No respira"}
    r = c.post("/v1/chat", json=payload, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["riesgo_vital"] is True
    assert body["modelo"] == "respuesta_segura"
    assert "+56 32 227 7999" in body["respuesta_texto"]
