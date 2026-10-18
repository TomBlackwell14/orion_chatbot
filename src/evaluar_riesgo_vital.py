"""Evalúa la detección de riesgo vital con Bedrock REAL sobre docs/casos_riesgo_vital.json.

Uso en la EC2:  docker compose run --rm chatbot python evaluar_riesgo_vital.py
Cada caso es un primer mensaje del usuario con la subcategoría del incidente. Muestra la matriz de confusión
y los casos fallidos. Termina con código 1 si hay falsos negativos (riesgo vital no detectado), que son el
error más grave en una app de emergencias. Correrlo cada vez que cambien los protocolos, el prompt o el modelo.
"""
import json
import sys
import time
from pathlib import Path

from rag import responder

CASOS = Path(__file__).resolve().parent.parent / "docs" / "casos_riesgo_vital.json"


def main() -> int:
    casos = json.loads(CASOS.read_text(encoding="utf-8"))
    vp = vn = fp = fn = errores = 0
    fallidos = []
    for c in casos:
        contexto = {"categoria": c["categoria"], "subcategoria": c["subcategoria"],
                    "ubicacion": "Edificio A", "estado_incidente": "Recibido"}
        historial = [{"rol": "bot", "texto": "Hola, soy el asistente de ORION. ¿Qué está pasando ahora?"},
                     {"rol": "usuario", "texto": c["mensaje"]}]
        inicio = time.monotonic()
        try:
            r = responder(historial, contexto)
        except Exception as e:  # noqa: BLE001 - se reporta y se sigue con el resto
            errores += 1
            fallidos.append((c, f"ERROR {type(e).__name__}", ""))
            continue
        ms = int((time.monotonic() - inicio) * 1000)
        obtenido, esperado = r["riesgo_vital"], c["esperado_riesgo_vital"]
        if obtenido and esperado:
            vp += 1
        elif not obtenido and not esperado:
            vn += 1
        elif obtenido:
            fp += 1
            fallidos.append((c, "FALSO POSITIVO", r["respuesta_texto"]))
        else:
            fn += 1
            fallidos.append((c, "FALSO NEGATIVO", r["respuesta_texto"]))
        print(f"[{c['id']:>2}] esperado={esperado!s:<5} obtenido={obtenido!s:<5} "
              f"regla={r['metadata']['senales_regla']} {ms} ms")

    total = len(casos)
    print(f"\nCasos: {total} · VP {vp} · VN {vn} · FP {fp} · FN {fn} · errores {errores}")
    if vp + fn:
        print(f"Sensibilidad (riesgo vital detectado): {vp / (vp + fn):.0%}")
    if vn + fp:
        print(f"Especificidad (sin falsas alarmas): {vn / (vn + fp):.0%}")
    for c, tipo, texto in fallidos:
        print(f"\n{tipo} · caso {c['id']} · {c['subcategoria']}\n  Usuario: {c['mensaje']}\n  Bot: {texto}")
    return 1 if fn or errores else 0


if __name__ == "__main__":
    sys.exit(main())
