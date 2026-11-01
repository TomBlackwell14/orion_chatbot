"""Pruebas de las herramientas de evaluación, sin AWS (Bedrock y juez simulados).
Uso (desde la raíz del repo):  python -m pytest -q tests
"""
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
DOCS = SRC.parent / "docs"
sys.path.insert(0, str(SRC))
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-2")

import acuerdo_juez  # noqa: E402
import evaluar_orientacion  # noqa: E402
import evaluar_riesgo_vital  # noqa: E402
import rag  # noqa: E402
from metricas import Matriz, kappa_cohen, wilson  # noqa: E402
from orientacion import chequeos_deterministas, criterios_juez, parsear_veredictos  # noqa: E402

CASOS = json.loads((DOCS / "casos_riesgo_vital.json").read_text(encoding="utf-8"))
ESCENARIOS = json.loads((DOCS / "escenarios_orientacion.json").read_text(encoding="utf-8"))
TEXTO_PROTOCOLOS = (DOCS / "protocolos_completos.txt").read_text(encoding="utf-8")
# "SAL-001 — ..." seguido de "ID Subcategoría (BD): 1 — Urgencia"
SUBCATEGORIA_DE = dict(re.findall(r"^((?:SAL|INF|SEG)-\d+) — .*?ID Subcategoría \(BD\): \d+ — ([^\n]+)",
                                  TEXTO_PROTOCOLOS, flags=re.MULTILINE | re.DOTALL))


# ---------- Métricas ----------

def test_matriz_y_metricas():
    m = Matriz()
    for esperado, obtenido in [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 2 + [(False, False)] * 8:
        m.agregar(esperado, obtenido)
    assert (m.vp, m.fn, m.fp, m.vn) == (8, 2, 2, 8)
    assert m.recall == pytest.approx(0.8)
    assert m.precision == pytest.approx(0.8)
    assert m.especificidad == pytest.approx(0.8)
    assert m.f_beta(2) == pytest.approx(0.8)


def test_f2_pesa_mas_el_recall():
    alto_recall, alta_precision = Matriz(vp=9, fn=1, fp=5), Matriz(vp=5, fn=5, fp=1)
    assert alto_recall.f_beta(2) > alta_precision.f_beta(2)


def test_wilson():
    bajo, alto = wilson(95, 100)
    assert bajo == pytest.approx(0.888, abs=0.002) and alto == pytest.approx(0.978, abs=0.002)
    assert wilson(10, 10)[1] == 1.0 and wilson(10, 10)[0] < 1.0
    assert wilson(0, 0) is None


def test_kappa():
    assert kappa_cohen([(True, True), (False, False)] * 5) == pytest.approx(1.0)
    assert kappa_cohen([(True, False), (False, True)] * 5) < 0


# ---------- Datos ----------

def test_casos_riesgo_vital_bien_formados():
    ids = [c["id"] for c in CASOS]
    assert len(ids) == len(set(ids))
    subcategorias = set(SUBCATEGORIA_DE.values())
    for c in CASOS:
        assert c["subcategoria"] in subcategorias, c
        assert isinstance(c["esperado_riesgo_vital"], bool)
        if c.get("tipo") == "multiturno":
            h = c["historial_previo"]
            assert h[0]["rol"] == "bot" and h[-1]["rol"] == "bot", c["id"]
            assert all(a["rol"] != b["rol"] for a, b in zip(h, h[1:])), c["id"]
    positivos = sum(c["esperado_riesgo_vital"] for c in CASOS)
    assert 0.35 < positivos / len(CASOS) < 0.65  # set balanceado


def test_escenarios_bien_formados():
    ids = [e["id"] for e in ESCENARIOS]
    assert len(ids) == len(set(ids))
    for e in ESCENARIOS:
        assert e["protocolo"] in SUBCATEGORIA_DE, e["id"]
        assert e["contexto"]["subcategoria"] == SUBCATEGORIA_DE[e["protocolo"]], e["id"]
        assert e["turnos"] and all(t["debe"] or t["no_debe"] for t in e["turnos"]), e["id"]
        criterios = [c["id"] for c in criterios_juez(e)]
        assert len(criterios) == len(set(criterios))
    assert set(SUBCATEGORIA_DE) == {e["protocolo"] for e in ESCENARIOS}  # todos los protocolos tienen escenario


# ---------- Chequeos deterministas y juez ----------

ESC = {"id": "X", "protocolo": "SAL-001", "contexto": {"estado_incidente": "Recibido"},
       "turnos": [{"usuario": "a", "esperado_riesgo_vital": True, "debe": ["x"], "no_debe": ["y"]},
                  {"usuario": "b", "esperado_riesgo_vital": False, "debe": [], "no_debe": []}]}


def _por_id(veredictos):
    return {v["id"]: v for v in veredictos}


def test_chequeos_detectan_infracciones():
    bot = [{"respuesta": f"{rag.AVISO} Llama al 131. El personal va en camino. ¿Respira? ¿Responde? ¿Sangra?",
            "riesgo_vital": False},
           {"respuesta": "Llegarán en 5 minutos. Uno. Dos. Tres. Cuatro.", "riesgo_vital": False}]
    v = _por_id(chequeos_deterministas(ESC, bot, rag.AVISO))
    assert v["T1.aviso"]["cumple"]
    assert not v["T1.riesgo_vital"]["cumple"]
    assert not v["T1.sin_numeros_externos"]["cumple"]
    assert not v["T1.sin_promesas"]["cumple"]
    assert not v["T1.max_2_preguntas"]["cumple"]
    assert v["T2.riesgo_vital"]["cumple"]
    assert not v["T2.sin_promesas"]["cumple"]
    assert not v["T2.max_4_oraciones"]["cumple"]


def test_en_camino_permitido_solo_con_estado_en_camino():
    bot = [{"respuesta": "Un respondedor ya va en camino. Quédate con la persona.", "riesgo_vital": True},
           {"respuesta": "Bien.", "riesgo_vital": False}]
    recibido = _por_id(chequeos_deterministas(ESC, bot, rag.AVISO))
    en_camino = _por_id(chequeos_deterministas({**ESC, "contexto": {"estado_incidente": "En_camino"}}, bot, rag.AVISO))
    assert not recibido["T1.sin_promesas"]["cumple"]
    assert en_camino["T1.sin_promesas"]["cumple"]
    assert not recibido["T1.aviso"]["cumple"]


def test_parsear_veredictos_marca_los_omitidos():
    criterios = criterios_juez(ESC)
    salida = json.dumps({"veredictos": [{"id": "T1.seguridad", "cumple": False, "evidencia": "dale agua"}],
                         "comentario": "ok"})
    v = _por_id(parsear_veredictos(salida, criterios)[0])
    assert v["T1.seguridad"]["cumple"] is False and v["T1.seguridad"]["critico"]
    assert v["T1.no_debe.1"]["cumple"] is None and v["T1.no_debe.1"]["critico"]
    assert v["T1.debe.1"]["cumple"] is None and not v["T1.debe.1"]["critico"]


# ---------- Scripts de punta a punta con Bedrock simulado ----------

class BedrockFalso:
    def __init__(self, riesgo: bool):
        self.riesgo = riesgo

    def invoke_model(self, modelId, body):
        return {"body": io.BytesIO(json.dumps({"embedding": [1.0] + [0.0] * 1023}).encode())}

    def converse(self, modelId, system, messages, inferenceConfig):
        salida = json.dumps({"respuesta_texto": f"{rag.AVISO} Quédate con la persona. ¿Respira?",
                             "riesgo_vital": self.riesgo, "protocolo": "SAL-001"})
        return {"output": {"message": {"content": [{"text": salida}]}}, "usage": {}}


def _correr(monkeypatch, modulo, *argv):
    monkeypatch.setattr(sys, "argv", [modulo.__name__, *argv])
    return modulo.main()


def test_evaluar_riesgo_solo_reglas(monkeypatch, tmp_path):
    assert _correr(monkeypatch, evaluar_riesgo_vital, "--solo-reglas", "--salida", str(tmp_path)) == 0
    carpeta = next(tmp_path.iterdir())
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    reglas = resumen["capas"]["Solo reglas (riesgo.py)"]
    assert reglas["vp"] + reglas["fn"] == sum(c["esperado_riesgo_vital"] for c in CASOS)
    assert reglas["vp"] == sum(c["debe_activar_regla"] for c in CASOS if c["esperado_riesgo_vital"])


def test_evaluar_riesgo_con_modelo_que_nunca_detecta(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "bedrock", BedrockFalso(riesgo=False))
    monkeypatch.setattr(rag, "_indice", None)
    # caso 1: riesgo con regla · caso 5: riesgo sin regla (falso negativo) · caso 10: sin riesgo
    codigo = _correr(monkeypatch, evaluar_riesgo_vital, "-n", "2", "--ids", "1,5,10", "--salida", str(tmp_path))
    assert codigo == 1  # hay falsos negativos
    resumen = json.loads((next(tmp_path.iterdir()) / "resumen.json").read_text(encoding="utf-8"))
    sistema = resumen["capas"]["Sistema completo"]
    assert (sistema["vp"], sistema["fn"], sistema["vn"], sistema["fp"]) == (2, 2, 2, 0)
    assert resumen["capas"]["Solo modelo"]["vp"] == 0
    assert resumen["peor_caso"]["fn"] == 1 and resumen["estabilidad"] == 1.0


def test_evaluar_orientacion_con_juez_simulado(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "bedrock", BedrockFalso(riesgo=True))
    monkeypatch.setattr(rag, "_indice", None)

    def juez_falso(escenario, turnos_bot, protocolo, marco):
        criterios = criterios_juez(escenario)
        assert "NUNCA DEBE INDICAR" in protocolo and "CRITERIO GENERAL DE RIESGO VITAL" in marco
        return parsear_veredictos(json.dumps({"veredictos": [{"id": c["id"], "cumple": True, "evidencia": "ok"}
                                                             for c in criterios], "comentario": ""}), criterios)

    monkeypatch.setattr(evaluar_orientacion, "juzgar", juez_falso)
    assert _correr(monkeypatch, evaluar_orientacion, "--ids", "SAL-001-01", "-n", "2", "--salida", str(tmp_path)) == 0
    carpeta = next(tmp_path.iterdir())
    reporte = (carpeta / "reporte.md").read_text(encoding="utf-8")
    assert "Conversaciones aprobadas" in reporte
    with (carpeta / "revision_manual.csv").open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert filas and all(fl["veredicto_humano"] == "" for fl in filas)

    # Re-juzgar desde la carpeta no vuelve a llamar al bot
    monkeypatch.setattr(rag, "bedrock", None)
    salida2 = tmp_path / "segunda"
    assert _correr(monkeypatch, evaluar_orientacion, "--desde", str(carpeta), "--salida", str(salida2)) == 0

    # Acuerdo juez-humano con el CSV completado (en Excel español se guarda con punto y coma)
    for i, fl in enumerate(filas):
        fl["veredicto_humano"] = "no" if i == 0 else "si"
    ruta = tmp_path / "revisado.csv"
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(filas)
    assert acuerdo_juez.main(str(ruta)) == 0


def test_veredictos_de_juez_externo(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "bedrock", BedrockFalso(riesgo=True))
    monkeypatch.setattr(rag, "_indice", None)
    assert _correr(monkeypatch, evaluar_orientacion, "--ids", "SAL-001-02", "-n", "1", "--sin-juez",
                   "--salida", str(tmp_path / "a")) == 0
    carpeta = next((tmp_path / "a").iterdir())
    esc = next(e for e in ESCENARIOS if e["id"] == "SAL-001-02")
    externos = {"SAL-001-02#1": {"veredictos": [{"id": c["id"], "cumple": c["grupo"] != "no_debe", "evidencia": "cita"}
                                                for c in criterios_juez(esc)], "comentario": "juez en el chat"}}
    ruta = tmp_path / "veredictos_juez.json"
    ruta.write_text(json.dumps(externos), encoding="utf-8")
    monkeypatch.setattr(rag, "bedrock", None)  # no debe llamar al bot ni al juez
    monkeypatch.setattr(evaluar_orientacion, "juzgar", None)
    assert _correr(monkeypatch, evaluar_orientacion, "--desde", str(carpeta), "--veredictos-juez", str(ruta),
                   "--salida", str(tmp_path / "b")) == 0
    salida = next((tmp_path / "b").iterdir())
    reporte = (salida / "reporte.md").read_text(encoding="utf-8")
    assert "externo (veredictos_juez.json)" in reporte
    assert "0 de 1" in reporte  # falló los no_debe (críticos), así que no se aprueba
    assert (salida / "revision_manual.csv").exists()


# ---------- JSON robusto en conversaciones de varios turnos ----------

def test_parsear_json_tolera_dos_objetos_y_prefiere_el_completo():
    import rag
    texto = '{"respuesta_texto": "copia del historial"}\n{"respuesta_texto": "Ponla de lado.", "riesgo_vital": true, "protocolo": "SAL-001"}'
    datos = rag._parsear_json(texto)
    assert datos["riesgo_vital"] is True and datos["protocolo"] == "SAL-001"


def test_parsear_json_tolera_texto_despues_con_llaves():
    import rag
    datos = rag._parsear_json('{"respuesta_texto": "Hola", "riesgo_vital": false} nota {extra}')
    assert datos["respuesta_texto"] == "Hola"


def test_rescatar_campos_de_json_roto():
    import rag
    roto = '{"respuesta_texto": "Revisa si respira.", "riesgo_vital": true, "protocolo": "SAL-001", "nota": "dijo "hola""}'
    assert rag._parsear_json(roto) is None
    datos = rag._rescatar_campos(roto)
    assert datos == {"respuesta_texto": "Revisa si respira.", "riesgo_vital": True, "protocolo": "SAL-001"}


def test_reintenta_si_falta_riesgo_vital(monkeypatch):
    import rag
    salidas = iter(['{"respuesta_texto": "Quédate con ella."}',
                    '{"respuesta_texto": "Ponla de lado.", "riesgo_vital": true, "protocolo": "SAL-001"}'])
    monkeypatch.setattr(rag, "buscar_contexto", lambda *a, **k: [])
    monkeypatch.setattr(rag, "marco_general", lambda: "")
    monkeypatch.setattr(rag, "DETECTOR_RIESGO", False)
    monkeypatch.setattr(rag, "_converse", lambda system, mensajes: (next(salidas), {}))
    r = rag._responder_con_modelo([{"role": "user", "content": [{"text": "no"}]}], {}, [])
    assert r["riesgo_vital"] is True and r["metadata"]["reintento_json"] is True


# ---------- Set de validación ----------

from riesgo import detectar_senales  # noqa: E402

VALIDACION = json.loads((DOCS / "casos_riesgo_vital_validacion.json").read_text(encoding="utf-8"))


def test_validacion_bien_formada_y_sin_ids_repetidos():
    ids_principal = {c["id"] for c in CASOS}
    for c in VALIDACION:
        assert c["id"] not in ids_principal
        assert c["subcategoria"] in SUBCATEGORIA_DE.values()
        assert bool(detectar_senales(c["mensaje"])) == c["debe_activar_regla"], c["id"]
    assert 0.35 < sum(c["esperado_riesgo_vital"] for c in VALIDACION) / len(VALIDACION) < 0.65


def test_evaluar_validacion_solo_reglas(monkeypatch, tmp_path):
    assert _correr(monkeypatch, evaluar_riesgo_vital, "--solo-reglas", "--casos", "validacion",
                   "--salida", str(tmp_path)) == 0
    assert next(tmp_path.iterdir()).name.endswith("_validacion_reglas")


def test_reglas_nuevas_sin_falsos_positivos_comunes():
    for frase in ["tengo la garganta inflamada por un resfrío", "me duele el pecho de tanto toser",
                  "no se le hinchó la lengua, solo tiene ronchas", "no tiene la cara chueca y habla normal"]:
        assert not detectar_senales(frase), frase


# ---------- Detector dedicado de riesgo vital ----------

def _converse_por_prompt(principal: str, detector: str):
    def _c(system, mensajes):
        return (detector if system.startswith("Eres el detector") else principal), {}
    return _c


def _preparar(monkeypatch, principal, detector):
    monkeypatch.setattr(rag, "buscar_contexto", lambda *a, **k: [])
    monkeypatch.setattr(rag, "DETECTOR_RIESGO", True)
    monkeypatch.setattr(rag, "_converse", _converse_por_prompt(principal, detector))


MSJ = [{"role": "user", "content": [{"text": "creo que tiene un cuchillo"}]}]


def test_detector_sube_riesgo_cuando_el_modelo_no_lo_marca(monkeypatch):
    _preparar(monkeypatch, '{"respuesta_texto": "Aléjate.", "riesgo_vital": false, "protocolo": "SEG-003"}',
              '{"riesgo_vital": true, "senal": "cuchillo"}')
    r = rag._responder_con_modelo(MSJ, {}, [])
    assert r["riesgo_vital"] is True
    assert r["metadata"]["riesgo_vital_modelo"] is False and r["metadata"]["riesgo_vital_detector"] is True


def test_detector_no_baja_lo_que_marca_el_modelo(monkeypatch):
    _preparar(monkeypatch, '{"respuesta_texto": "Aléjate.", "riesgo_vital": true, "protocolo": "SEG-003"}',
              '{"riesgo_vital": false, "senal": null}')
    assert rag._responder_con_modelo(MSJ, {}, [])["riesgo_vital"] is True


def test_detector_roto_no_bloquea_la_respuesta(monkeypatch):
    _preparar(monkeypatch, '{"respuesta_texto": "Aléjate.", "riesgo_vital": false, "protocolo": "SEG-003"}', "???")
    r = rag._responder_con_modelo(MSJ, {}, [])
    assert r["riesgo_vital"] is False and r["metadata"]["riesgo_vital_detector"] is None


def test_detector_recibe_criterio_y_texto_plano_del_bot(monkeypatch):
    vistos = []

    def _c(system, mensajes):
        vistos.append((system, mensajes[0]["content"][0]["text"]))
        return '{"riesgo_vital": false}', {}
    monkeypatch.setattr(rag, "_converse", _c)
    mensajes = rag.normalizar_historial([{"rol": "usuario", "texto": "se desplomó"},
                                         {"rol": "bot", "texto": "¿Respira?"}, {"rol": "usuario", "texto": "no"}])
    rag.detectar_riesgo_modelo(rag._historial_plano(mensajes))
    system, pedido = vistos[0]
    assert system.startswith("Eres el detector") and "3.4 Cuándo marcar" in pedido
    assert "ASISTENTE: ¿Respira?" in pedido and "respuesta_texto" not in pedido
