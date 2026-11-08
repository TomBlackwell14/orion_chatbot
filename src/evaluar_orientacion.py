"""Evalúa la calidad de la orientación con conversaciones guionadas (docs/escenarios_orientacion.json).

Uso (desde src/, con credenciales de Bedrock):
  python evaluar_orientacion.py                       # 3 repeticiones por escenario, con juez
  python evaluar_orientacion.py --ids SAL-001 -n 1     # solo los escenarios de SAL-001, una vez
  python evaluar_orientacion.py --sin-juez             # solo chequeos automáticos (sin costo del juez)
  python evaluar_orientacion.py --desde ../resultados/orientacion_2026-11-01_1200   # re-juzga sin volver a llamar al bot
  python evaluar_orientacion.py --desde <carpeta> --veredictos-juez veredictos_juez.json   # juez externo (Claude en el chat)

Cada escenario son los mensajes del usuario turno a turno; el bot responde de verdad y su respuesta entra al
historial del turno siguiente, igual que en la app. Después cada turno se evalúa con chequeos deterministas y con
un LLM juez que aplica la rúbrica de orientacion.py. Deja en resultados/orientacion_<fecha>/:
  reporte.md            tablas para la tesis y los fallos con su evidencia
  veredictos.csv        un renglón por criterio evaluado
  conversaciones.json   las conversaciones completas (para re-juzgar con --desde)
  revision_manual.csv   muestra para validar al juez: completa la columna veredicto_humano y corre acuerdo_juez.py
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from metricas import pct
from orientacion import (JUEZ_MODEL, UMBRAL_APROBACION, chequeos_deterministas, criterios_juez, juzgar,
                         parsear_veredictos)

RAIZ = Path(__file__).resolve().parent.parent
ESCENARIOS = RAIZ / "docs" / "escenarios_orientacion.json"
SALUDO = {"rol": "bot", "texto": "Hola, soy el asistente de ORION. ¿Qué está pasando ahora?"}


def conversar(responder, escenario: dict) -> list[dict]:
    historial, turnos_bot = [SALUDO], []
    for turno in escenario["turnos"]:
        historial.append({"rol": "usuario", "texto": turno["usuario"]})
        try:
            r = responder(historial, escenario["contexto"])
            bot = {"respuesta": r["respuesta_texto"], "riesgo_vital": bool(r["riesgo_vital"]), "protocolo": r.get("protocolo"),
                   "recuperados": r.get("metadata", {}).get("protocolos_recuperados", []), "error": ""}
        except Exception as e:  # noqa: BLE001 - en la app NestJS mostraría su mensaje de respaldo
            bot = {"respuesta": "", "riesgo_vital": False, "protocolo": None, "recuperados": [],
                   "error": f"{type(e).__name__}: {e}"[:200]}
        turnos_bot.append(bot)
        historial.append({"rol": "bot", "texto": bot["respuesta"] or "Disculpa, tuve un problema. ¿Puedes repetir?"})
    return turnos_bot


def aprobado(veredictos: list[dict]) -> bool:
    evaluados = [v for v in veredictos if v["cumple"] is not None and v["grupo"] != "riesgo_vital"]
    if any(v["critico"] and not v["cumple"] for v in evaluados):
        return False
    resto = [v for v in evaluados if not v["critico"]]
    return not resto or sum(v["cumple"] for v in resto) / len(resto) >= UMBRAL_APROBACION


def tasa(vs: list[dict]) -> str:
    evaluados = [v for v in vs if v["cumple"] is not None]
    return f"{pct(sum(v['cumple'] for v in evaluados) / len(evaluados))} ({len(evaluados)})" if evaluados else "—"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-n", "--repeticiones", type=int, default=3)
    p.add_argument("--ids", help="ids o prefijos separados por coma, por ejemplo SAL-001,SEG-002-01")
    p.add_argument("--sin-juez", action="store_true", help="solo chequeos deterministas")
    p.add_argument("--desde", help="carpeta de una corrida anterior: re-juzga sus conversaciones sin llamar al bot")
    p.add_argument("--veredictos-juez", help="JSON con los veredictos de un juez externo, en vez de llamar al juez por API")
    p.add_argument("--muestra", type=int, default=30, help="conversaciones para revisión manual del juez")
    p.add_argument("--salida", default=str(RAIZ / "resultados"))
    p.add_argument("--escenarios", default=str(ESCENARIOS),
                   help="archivo de escenarios (por defecto la línea base; test 2: docs/escenarios_orientacion_test2.json)")
    args = p.parse_args()

    import rag  # importa boto3; el juez usa su propio cliente
    from config import LLM_MODEL

    archivo_escenarios = Path(args.escenarios)
    if not archivo_escenarios.is_absolute() and not archivo_escenarios.exists():
        archivo_escenarios = RAIZ / archivo_escenarios
    escenarios = json.loads(archivo_escenarios.read_text(encoding="utf-8"))
    if args.ids:
        prefijos = [i.strip() for i in args.ids.split(",")]
        escenarios = [e for e in escenarios if any(e["id"].startswith(pr) for pr in prefijos)]
    por_id = {e["id"]: e for e in escenarios}
    protocolos = dict(rag.dividir_protocolos(rag._ruta_protocolos().read_text(encoding="utf-8")))
    marco = protocolos["GEN-000"]

    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    sufijo = archivo_escenarios.stem.replace("escenarios_orientacion", "")  # "" o "_test2"
    carpeta = Path(args.salida) / f"orientacion{sufijo}_{fecha}"
    carpeta.mkdir(parents=True, exist_ok=True)
    # evidencia: copia exacta de los escenarios usados en esta corrida
    (carpeta / "escenarios_usados.json").write_text(archivo_escenarios.read_text(encoding="utf-8"), encoding="utf-8")

    if args.desde:
        conversaciones = json.loads((Path(args.desde) / "conversaciones.json").read_text(encoding="utf-8"))
        conversaciones = [c for c in conversaciones if c["escenario"] in por_id]
    else:
        conversaciones = []
        for e in escenarios:
            for rep in range(1, args.repeticiones + 1):
                inicio = time.monotonic()
                turnos_bot = conversar(rag.responder, e)
                conversaciones.append({"escenario": e["id"], "repeticion": rep, "turnos_bot": turnos_bot})
                print(f"[{e['id']} #{rep}] {len(turnos_bot)} turnos · {time.monotonic() - inicio:.1f} s")
        (carpeta / "conversaciones.json").write_text(json.dumps(conversaciones, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")

    externos = None
    if args.veredictos_juez:
        # {"SAL-001-01#1": {"veredictos": [{"id", "cumple", "evidencia"}], "comentario": ""}, ...}
        externos = json.loads(Path(args.veredictos_juez).read_text(encoding="utf-8"))
    filas, comentarios, errores_juez = [], {}, 0
    for conv in conversaciones:
        e, turnos_bot = por_id[conv["escenario"]], conv["turnos_bot"]
        veredictos = chequeos_deterministas(e, turnos_bot, rag.AVISO, getattr(rag, "FRASES_FIJAS", tuple(getattr(rag, "MENSAJES_ESTADO", {}).values())))
        if externos is not None:
            clave = f"{e['id']}#{conv['repeticion']}"
            if clave in externos:
                del_juez, comentario = parsear_veredictos(json.dumps(externos[clave]), criterios_juez(e))
                veredictos += del_juez
                comentarios[(e["id"], conv["repeticion"])] = comentario
            else:
                errores_juez += 1
                print(f"  sin veredictos externos para {clave}")
        elif not args.sin_juez:
            try:
                del_juez, comentario = juzgar(e, turnos_bot, protocolos[e["protocolo"]], marco)
                veredictos += del_juez
                comentarios[(e["id"], conv["repeticion"])] = comentario
            except Exception as ex:  # noqa: BLE001
                errores_juez += 1
                print(f"  juez falló en {e['id']} #{conv['repeticion']}: {type(ex).__name__}: {ex}")
        ok = aprobado(veredictos)
        print(f"{'OK ' if ok else 'XX '}[{e['id']} #{conv['repeticion']}] {e['titulo']}")
        for v in veredictos:
            n = v["turno"] - 1
            filas.append({"escenario": e["id"], "protocolo": e["protocolo"], "repeticion": conv["repeticion"],
                          "turno": v["turno"], "criterio": v["id"], "grupo": v["grupo"], "critico": v["critico"],
                          "fuente": v["fuente"], "descripcion": v["descripcion"], "cumple": v["cumple"],
                          "evidencia": v["evidencia"], "usuario": e["turnos"][n]["usuario"],
                          "respuesta_bot": turnos_bot[n]["respuesta"] or turnos_bot[n]["error"],
                          "escenario_aprobado": ok})

    with (carpeta / "veredictos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # Muestra para revisión humana: conversaciones completas al azar (semilla fija para poder repetirla)
    del_juez = [f for f in filas if f["fuente"] == "juez"]
    if del_juez:
        claves = sorted({(f["escenario"], f["repeticion"]) for f in del_juez})
        elegidas = set(random.Random(42).sample(claves, min(args.muestra, len(claves))))
        with (carpeta / "revision_manual.csv").open("w", newline="", encoding="utf-8") as f:
            campos = ["escenario", "repeticion", "turno", "criterio", "descripcion", "usuario", "respuesta_bot",
                      "cumple", "evidencia", "veredicto_humano"]
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows({**r, "cumple": "si" if r["cumple"] else "no", "veredicto_humano": ""}
                        for r in del_juez if (r["escenario"], r["repeticion"]) in elegidas)

    # ---------- Reporte ----------
    por_conv = defaultdict(list)
    for f in filas:
        por_conv[(f["escenario"], f["repeticion"])].append(f)
    aprob = {k: v[0]["escenario_aprobado"] for k, v in por_conv.items()}
    por_esc = defaultdict(list)
    for (eid, _), ok in aprob.items():
        por_esc[eid].append(ok)
    siempre = sum(all(v) for v in por_esc.values())
    nunca = sum(not any(v) for v in por_esc.values())

    reps = max((c["repeticion"] for c in conversaciones), default=0)
    rv = [f for f in filas if f["grupo"] == "riesgo_vital"]
    fn_rv = sum(1 for f in rv if not f["cumple"] and "True" in f["descripcion"])
    r = [f"# Evaluación de la orientación · {fecha}", "",
         f"- Escenarios: {len(por_esc)} · repeticiones: {reps} · conversaciones: {len(por_conv)} · "
         f"turnos evaluados: {len(rv)}",
         f"- Modelo evaluado: `{LLM_MODEL}` · juez: "
         f"{f'externo ({Path(args.veredictos_juez).name})' if externos is not None else 'sin juez' if args.sin_juez else f'`{JUEZ_MODEL}`'}"
         f"{f' · el juez falló en {errores_juez} conversaciones' if errores_juez else ''}",
         f"- Un escenario se aprueba con 0 criterios críticos fallidos y al menos {UMBRAL_APROBACION:.0%} del resto cumplido.",
         "", "## Resultado general", "",
         "| Indicador | Valor |", "| --- | --- |",
         f"| Conversaciones aprobadas | {pct(sum(aprob.values()) / len(aprob))} ({sum(aprob.values())} de {len(aprob)}) |",
         f"| Escenarios aprobados en todas las repeticiones | {siempre} de {len(por_esc)} |",
         f"| Escenarios que fallaron en todas las repeticiones | {nunca} de {len(por_esc)} |",
         f"| Criterios críticos cumplidos | {tasa([f for f in filas if f['critico']])} |",
         f"| riesgo_vital correcto por turno | {tasa(rv)} · falsos negativos: {fn_rv} |",
         "", "## Cumplimiento por criterio", "", "| Criterio | Fuente | Crítico | Cumple (n) |", "| --- | --- | --- | --- |"]
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["grupo"], f["fuente"], f["critico"])].append(f)
    for (g, fuente, critico), vs in sorted(grupos.items(), key=lambda x: (not x[0][2], x[0][1], x[0][0])):
        r.append(f"| {g} | {fuente} | {'sí' if critico else 'no'} | {tasa(vs)} |")
    conjuntos = {e["id"]: e.get("conjunto", "desarrollo") for e in escenarios}
    if len(set(conjuntos.values())) > 1:  # desarrollo (usados para ajustar el bot) vs validación (nuevos)
        r += ["", "## Por conjunto", "", "| Conjunto | Conversaciones aprobadas | Criterios críticos cumplidos |",
              "| --- | --- | --- |"]
        for conj in sorted(set(conjuntos.values())):
            ap = [ok for (eid, _), ok in aprob.items() if conjuntos.get(eid) == conj]
            crit = [f for f in filas if f["critico"] and conjuntos.get(f["escenario"]) == conj]
            r.append(f"| {conj} | {pct(sum(ap) / len(ap)) if ap else '—'} ({sum(ap)} de {len(ap)}) | {tasa(crit)} |")
    r += ["", "## Por protocolo", "", "| Protocolo | Conversaciones aprobadas | Debe | No debe | Seguridad | riesgo_vital |",
          "| --- | --- | --- | --- | --- | --- |"]
    por_proto = defaultdict(list)
    for f in filas:
        por_proto[f["protocolo"]].append(f)
    for proto, vs in sorted(por_proto.items()):
        convs = {(f["escenario"], f["repeticion"]) for f in vs}
        r.append(f"| {proto} | {sum(aprob[c] for c in convs)} de {len(convs)} | {tasa([f for f in vs if f['grupo'] == 'debe'])} "
                 f"| {tasa([f for f in vs if f['grupo'] == 'no_debe'])} | {tasa([f for f in vs if f['grupo'] == 'seguridad'])} "
                 f"| {tasa([f for f in vs if f['grupo'] == 'riesgo_vital'])} |")
    r += ["", "## Fallos (críticos primero)", ""]
    fallos = sorted((f for f in filas if f["cumple"] is False), key=lambda f: (not f["critico"], f["escenario"], f["repeticion"]))
    if not fallos:
        r.append("Ninguno.")
    for f in fallos[:150]:
        r += [f"- **{f['escenario']} #{f['repeticion']} · {f['criterio']}**{' (crítico)' if f['critico'] else ''}: "
              f"{f['descripcion']}", f"  - Usuario: {f['usuario']}", f"  - Bot: {f['respuesta_bot']}",
              f"  - Evidencia: {f['evidencia']}"]
    if len(fallos) > 150:
        r.append(f"\n… y {len(fallos) - 150} más en veredictos.csv")
    (carpeta / "reporte.md").write_text("\n".join(r), encoding="utf-8")
    print(f"\nConversaciones aprobadas: {sum(aprob.values())} de {len(aprob)} · reporte: {carpeta / 'reporte.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
