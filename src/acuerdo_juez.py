"""Mide cuánto coincide el LLM juez con tu propia revisión, para saber si se puede confiar en él.

1. Corre evaluar_orientacion.py; deja resultados/orientacion_<fecha>/revision_manual.csv.
2. Abre el CSV (Excel o LibreOffice), lee usuario y respuesta_bot, y escribe en veredicto_humano "si" o "no"
   según TU criterio, sin mirar la columna cumple del juez si puedes. Las filas que dejes vacías no se cuentan.
3. python acuerdo_juez.py ../resultados/orientacion_<fecha>/revision_manual.csv

Regla práctica: con un acuerdo de 85-90% o más y kappa sobre 0.6 el juez es usable; si no, ajusta la redacción
de los criterios en orientacion.py o de los debe / no_debe del escenario y vuelve a juzgar con --desde.
"""
import csv
import io
import sys
from collections import defaultdict

from metricas import kappa_cohen, pct


def _si_no(valor: str) -> bool | None:
    v = (valor or "").strip().lower()
    if v in ("si", "sí", "s", "1", "true", "cumple"):
        return True
    if v in ("no", "n", "0", "false", "no cumple"):
        return False
    return None


def main(ruta: str) -> int:
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        texto = f.read()
    # Excel en español guarda los CSV con punto y coma
    delimitador = ";" if texto.split("\n", 1)[0].count(";") > texto.split("\n", 1)[0].count(",") else ","
    filas = list(csv.DictReader(io.StringIO(texto), delimiter=delimitador))
    pares, por_grupo, desacuerdos = [], defaultdict(list), []
    for fila in filas:
        humano, juez = _si_no(fila.get("veredicto_humano")), _si_no(fila.get("cumple"))
        if humano is None or juez is None:
            continue
        pares.append((juez, humano))
        grupo = fila["criterio"].split(".", 1)[1].split(".")[0]
        por_grupo[grupo].append((juez, humano))
        if juez != humano:
            desacuerdos.append(fila)
    if not pares:
        print("No hay filas con veredicto_humano completado")
        return 1
    acuerdo = sum(a == b for a, b in pares) / len(pares)
    print(f"Criterios revisados: {len(pares)}")
    print(f"Acuerdo juez-humano: {pct(acuerdo)} · kappa de Cohen: {kappa_cohen(pares):.2f}" if kappa_cohen(pares) is not None
          else f"Acuerdo juez-humano: {pct(acuerdo)}")
    print("\n| Grupo | n | Acuerdo |\n| --- | --- | --- |")
    for g, ps in sorted(por_grupo.items()):
        print(f"| {g} | {len(ps)} | {pct(sum(a == b for a, b in ps) / len(ps))} |")
    print("\nDesacuerdos (juez / tú):")
    for d in desacuerdos:
        print(f"- {d['escenario']} #{d['repeticion']} {d['criterio']}: juez={d['cumple']} tú={d['veredicto_humano']}"
              f"\n  {d['descripcion']}\n  Bot: {d['respuesta_bot'][:200]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
