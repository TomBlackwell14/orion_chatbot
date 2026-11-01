"""Métricas de clasificación para la evaluación de riesgo vital y de orientación. Sin dependencias externas.

Positivo = riesgo vital. El error que más importa es el falso negativo (había riesgo y no se detectó), por eso la
métrica principal es el recall (sensibilidad) y la métrica resumen es F2, que pesa el recall el doble que la precisión.
"""
import math
from dataclasses import dataclass


@dataclass
class Matriz:
    vp: int = 0
    vn: int = 0
    fp: int = 0
    fn: int = 0

    def agregar(self, esperado: bool, obtenido: bool) -> None:
        if esperado and obtenido:
            self.vp += 1
        elif not esperado and not obtenido:
            self.vn += 1
        elif obtenido:
            self.fp += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.vp + self.vn + self.fp + self.fn

    @property
    def recall(self) -> float | None:
        return _division(self.vp, self.vp + self.fn)

    @property
    def precision(self) -> float | None:
        return _division(self.vp, self.vp + self.fp)

    @property
    def especificidad(self) -> float | None:
        return _division(self.vn, self.vn + self.fp)

    @property
    def exactitud(self) -> float | None:
        return _division(self.vp + self.vn, self.total)

    def f_beta(self, beta: float = 2.0) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p == 0 and r == 0):
            return None
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    def resumen(self) -> dict:
        return {"vp": self.vp, "vn": self.vn, "fp": self.fp, "fn": self.fn,
                "recall": self.recall, "recall_ic95": wilson(self.vp, self.vp + self.fn),
                "precision": self.precision, "especificidad": self.especificidad,
                "f2": self.f_beta(2.0), "exactitud": self.exactitud}


def _division(a: int, b: int) -> float | None:
    return a / b if b else None


def wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de confianza de Wilson al 95% para una proporción. Se porta bien con n chico y con 0% o 100%."""
    if n == 0:
        return None
    p = exitos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - margen), min(1.0, centro + margen)


def kappa_cohen(pares: list[tuple[bool, bool]]) -> float | None:
    """Acuerdo entre dos evaluadores (juez LLM y revisión humana) corregido por azar. 1 = acuerdo total."""
    n = len(pares)
    if n == 0:
        return None
    observado = sum(a == b for a, b in pares) / n
    pa = sum(a for a, _ in pares) / n
    pb = sum(b for _, b in pares) / n
    esperado = pa * pb + (1 - pa) * (1 - pb)
    if esperado == 1:
        return 1.0 if observado == 1 else None
    return (observado - esperado) / (1 - esperado)


def pct(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.1%}"


def pct_ic(ic: tuple[float, float] | None) -> str:
    return "—" if ic is None else f"{ic[0]:.1%} – {ic[1]:.1%}"
