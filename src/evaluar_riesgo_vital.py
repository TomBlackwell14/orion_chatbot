"""Evalúa la detección de riesgo vital sobre docs/casos_riesgo_vital.json.

Uso en la EC2:
  docker compose run --rm chatbot python evaluar_riesgo_vital.py                 # Bedrock real, 5 repeticiones
  docker compose run --rm chatbot python evaluar_riesgo_vital.py -n 3 --tipo multiturno
  python evaluar_riesgo_vital.py --solo-reglas                                  # sin Bedrock, gratis

Cada caso es el mensaje actual del usuario (con historial previo opcional para los casos multi-turno) y su
etiqueta esperada. Cada caso se repite N veces porque el modelo no responde siempre igual. Se miden tres capas
por separado: solo reglas (riesgo.py), solo modelo, solo detector dedicado y el sistema completo (lo que recibe NestJS).

Deja en resultados/riesgo_<fecha>/ un reporte.md con las tablas para la tesis, un observaciones.csv con cada
llamada y un resumen.json. Termina con código 1 si hay falsos negativos en el peor caso o errores.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from metricas import Matriz, pct, pct_ic
from riesgo import detectar_senales

RAIZ = Path(__file__).resolve().parent.parent
CASOS = RAIZ / "docs" / "casos_riesgo_vital.json"
# Set de validación: frases nuevas escritas después de la primera corrida y que NO se usaron para ajustar reglas ni
# prompt. Sirve para mostrar que las mejoras generalizan y no solo memorizan los casos del set principal.
CASOS_VALIDACION = RAIZ / "docs" / "casos_riesgo_vital_validacion.json"
SALUDO = {"rol": "bot", "texto": "Hola, soy el asistente de ORION. ¿Qué está pasando ahora?"}


def cargar_casos(args) -> list[dict]:
    ruta = CASOS_VALIDACION if args.casos == "validacion" else CASOS
    casos = json.loads(ruta.read_text(encoding="utf-8"))
    if args.ids:
        ids = {int(i) for i in args.ids.split(",")}
        casos = [c for c in casos if c["id"] in ids]
    if args.tipo:
        casos = [c for c in casos if c.get("tipo") == args.tipo]
    if args.categoria:
        casos = [c for c in casos if c["categoria"].lower() == args.categoria.lower()]
    return casos


def historial_de(caso: dict) -> list[dict]:
    return [*(caso.get("historial_previo") or [SALUDO]), {"rol": "usuario", "texto": caso["mensaje"]}]


def llamar(responder, caso: dict, pausa: float) -> dict:
    """Una llamada al chatbot. Si falla sin señal de regla, en producción NestJS marca riesgo_vital = false."""
    contexto = {"categoria": caso["categoria"], "subcategoria": caso["subcategoria"],
                "ubicacion": "Edificio A", "estado_incidente": "Recibido"}
    inicio = time.monotonic()
    for intento in range(3):
        try:
            r = responder(historial_de(caso), contexto)
            break
        except Exception as e:  # noqa: BLE001 - se reporta y se sigue con el resto
            if "Throttling" in type(e).__name__ + str(e) and intento < 2:
                time.sleep(5 * (intento + 1))
                continue
            return {"sistema": False, "modelo": False, "detector": False, "error": f"{type(e).__name__}: {e}"[:800],
                    "ms": int((time.monotonic() - inicio) * 1000), "protocolo": None, "recuperados": "", "respuesta": ""}
    time.sleep(pausa)
    meta = r.get("metadata", {})
    return {"sistema": bool(r["riesgo_vital"]), "modelo": bool(meta.get("riesgo_vital_modelo")),
            "detector": bool(meta.get("riesgo_vital_detector")), "error": "",
            "ms": int((time.monotonic() - inicio) * 1000), "protocolo": r.get("protocolo"),
            "recuperados": ",".join(meta.get("protocolos_recuperados", [])), "respuesta": r["respuesta_texto"]}


def tabla_capas(capas: dict[str, Matriz]) -> list[str]:
    filas = ["| Capa | VP | FN | FP | VN | Recall (IC 95%) | Precisión | Especificidad | F2 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nombre, m in capas.items():
        filas.append(f"| {nombre} | {m.vp} | {m.fn} | {m.fp} | {m.vn} | {pct(m.recall)} ({pct_ic(m.resumen()['recall_ic95'])}) "
                     f"| {pct(m.precision)} | {pct(m.especificidad)} | {pct(m.f_beta())} |")
    return filas


def tabla_grupos(titulo: str, grupos: dict[str, Matriz]) -> list[str]:
    filas = [f"| {titulo} | Casos·repeticiones | Recall | Precisión | FN | FP |", "| --- | --- | --- | --- | --- | --- |"]
    for nombre, m in sorted(grupos.items()):
        filas.append(f"| {nombre} | {m.total} | {pct(m.recall)} | {pct(m.precision)} | {m.fn} | {m.fp} |")
    return filas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-n", "--repeticiones", type=int, default=5)
    p.add_argument("--solo-reglas", action="store_true", help="no llama a Bedrock; evalúa solo riesgo.py")
    p.add_argument("--ids", help="ids separados por coma")
    p.add_argument("--tipo", help="filtra por tipo: multiturno, sin_palabra_clave, falsa_alarma, ...")
    p.add_argument("--categoria", help="Salud, Infraestructura o Seguridad")
    p.add_argument("--pausa", type=float, default=0.0, help="segundos entre llamadas, por si Bedrock limita")
    p.add_argument("--casos", choices=["principal", "validacion"], default="principal",
                   help="set de casos: principal (152) o validacion (frases nuevas)")
    p.add_argument("--salida", default=str(RAIZ / "resultados"))
    args = p.parse_args()

    casos = cargar_casos(args)
    if not casos:
        print("No hay casos con esos filtros")
        return 1
    reps = 1 if args.solo_reglas else args.repeticiones
    responder = None
    if not args.solo_reglas:
        from rag import responder  # importa boto3 solo cuando se usa Bedrock
        from config import DETECTOR_RIESGO, LLM_MODEL, TEMPERATURE, TOP_K_RESULTS

    capas = {"Solo reglas (riesgo.py)": Matriz()} if args.solo_reglas else \
        {"Solo reglas (riesgo.py)": Matriz(), "Solo modelo": Matriz(), "Solo detector": Matriz(),
         "Sistema completo": Matriz()}
    if not args.solo_reglas and not DETECTOR_RIESGO:
        del capas["Solo detector"]
    peor_caso, por_tipo, por_categoria = Matriz(), defaultdict(Matriz), defaultdict(Matriz)
    estables, errores, observaciones, problemas = 0, 0, [], []

    for c in casos:
        esperado = c["esperado_riesgo_vital"]
        regla = bool(detectar_senales(c["mensaje"]))
        resultados = []
        for rep in range(1, reps + 1):
            if args.solo_reglas:
                obs = {"sistema": regla, "modelo": None, "detector": None, "error": "", "ms": 0, "protocolo": None,
                       "recuperados": "", "respuesta": ""}
            else:
                obs = llamar(responder, c, args.pausa)
                capas["Solo modelo"].agregar(esperado, obs["modelo"])
                if "Solo detector" in capas:
                    capas["Solo detector"].agregar(esperado, obs["detector"])
                capas["Sistema completo"].agregar(esperado, obs["sistema"])
            capas["Solo reglas (riesgo.py)"].agregar(esperado, regla)
            por_tipo[c.get("tipo", "base")].agregar(esperado, obs["sistema"])
            por_categoria[c["subcategoria"]].agregar(esperado, obs["sistema"])
            errores += bool(obs["error"])
            resultados.append(obs)
            observaciones.append({"id": c["id"], "repeticion": rep, "tipo": c.get("tipo", ""),
                                  "categoria": c["categoria"], "subcategoria": c["subcategoria"],
                                  "mensaje": c["mensaje"], "esperado": esperado, "regla": regla, **obs})
        detecciones = sum(o["sistema"] for o in resultados)
        estables += detecciones in (0, reps)
        # Peor caso: un positivo cuenta como detectado solo si se detectó en TODAS las repeticiones;
        # un negativo cuenta como falsa alarma si alguna repetición lo marcó.
        peor_caso.agregar(esperado, detecciones == reps if esperado else detecciones > 0)
        marca = "OK " if (detecciones == reps if esperado else detecciones == 0) else "XX "
        print(f"{marca}[{c['id']:>3}] esperado={esperado!s:<5} detectado {detecciones}/{reps} "
              f"regla={'sí' if regla else 'no'} · {c['mensaje'][:70]}")
        if marca == "XX ":
            problemas.append((c, detecciones, resultados))

    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    sufijo = ("_validacion" if args.casos == "validacion" else "") + ("_reglas" if args.solo_reglas else "")
    carpeta = Path(args.salida) / f"riesgo_{fecha}{sufijo}"
    carpeta.mkdir(parents=True, exist_ok=True)
    with (carpeta / "observaciones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(observaciones[0].keys()))
        w.writeheader()
        w.writerows(observaciones)

    n_pos = sum(c["esperado_riesgo_vital"] for c in casos)
    reporte = [f"# Evaluación de riesgo vital · {fecha}", "",
               f"- Set: {args.casos} · Casos: {len(casos)} ({n_pos} con riesgo vital, {len(casos) - n_pos} sin) · repeticiones por caso: {reps}"]
    if not args.solo_reglas:
        reporte += [f"- Modelo: `{LLM_MODEL}` · temperatura {TEMPERATURE} · protocolos recuperados (top-k): {TOP_K_RESULTS}",
                    f"- Detector dedicado de riesgo vital: {'activado' if DETECTOR_RIESGO else 'desactivado'}",
                    f"- Errores de llamada: {errores} (cuentan como riesgo_vital = false, igual que el fallback de NestJS)"]
    reporte += ["", "## Resultado por capa (cada repetición cuenta como una observación)", "", *tabla_capas(capas), ""]
    if not args.solo_reglas:
        reporte += ["## Peor caso (un caso grave cuenta como detectado solo si se detectó en todas las repeticiones)", "",
                    *tabla_capas({"Sistema completo, peor caso": peor_caso}), "",
                    f"Estabilidad: {estables} de {len(casos)} casos ({pct(estables / len(casos))}) dieron el mismo "
                    f"resultado en las {reps} repeticiones.", ""]
    capa = "solo reglas" if args.solo_reglas else "sistema completo"
    reporte += [f"## Por tipo de frase ({capa})", "", *tabla_grupos("Tipo", por_tipo), "",
                f"## Por subcategoría ({capa})", "", *tabla_grupos("Subcategoría", por_categoria), "",
                "## Casos con fallos", ""]
    if not problemas:
        reporte.append("Ninguno.")
    for c, detecciones, resultados in problemas:
        tipo_error = "FALSO NEGATIVO" if c["esperado_riesgo_vital"] else "FALSO POSITIVO"
        ultimo = resultados[-1]
        reporte += [f"### {tipo_error} · caso {c['id']} · {c['subcategoria']} · detectado {detecciones}/{reps}", "",
                    f"- Usuario: {c['mensaje']}"]
        if c.get("historial_previo"):
            reporte.append(f"- Historial previo: {' / '.join(m['texto'] for m in c['historial_previo'][1:])}")
        if c.get("justificacion"):
            reporte.append(f"- Etiqueta: {c['justificacion']}")
        if not args.solo_reglas:
            reporte += [f"- Protocolos recuperados: {ultimo['recuperados'] or '—'} · protocolo usado: {ultimo['protocolo']}",
                        f"- Bot (última repetición): {ultimo['respuesta'] or ultimo['error']}"]
        reporte.append("")
    (carpeta / "reporte.md").write_text("\n".join(reporte), encoding="utf-8")
    (carpeta / "resumen.json").write_text(json.dumps(
        {"fecha": fecha, "casos": len(casos), "repeticiones": reps, "errores": errores,
         "capas": {k: m.resumen() for k, m in capas.items()},
         "peor_caso": None if args.solo_reglas else peor_caso.resumen(),
         "estabilidad": None if args.solo_reglas else estables / len(casos)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "\n".join(tabla_capas(capas)))
    print(f"\nReporte: {carpeta / 'reporte.md'}")
    fn = capas["Solo reglas (riesgo.py)"].fn if args.solo_reglas else peor_caso.fn
    return 1 if (fn and not args.solo_reglas) or errores else 0


if __name__ == "__main__":
    sys.exit(main())
