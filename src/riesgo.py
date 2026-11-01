"""Red de seguridad determinística para riesgo vital.

El modelo decide riesgo_vital con el criterio general de GEN-000, pero las reglas de seguridad no se delegan
solo al LLM: si el mensaje del usuario contiene una señal explícita y no negada de amenaza a la vida, el código
fuerza riesgo_vital = true aunque el modelo no lo haya marcado. Las reglas solo pueden subir a true, nunca bajar.

Las expresiones son de alta especificidad a propósito: un falso positivo sube el incidente a Crítica sin motivo.
Cada regla nueva debe agregarse junto con casos positivos y negados en docs/casos_riesgo_vital.json.
"""
import re
import unicodedata

# Palabras que, justo antes de la señal, la niegan ("no hay armas", "sin sangrado", "nadie está atrapado")
_NEGACIONES = r"(?:no|sin|ningun|ninguna|ninguno|nadie|nada|tampoco|nunca|ni|descarta|descartado)"
_VENTANA_NEGACION = 3  # cantidad de palabras previas que se revisan

REGLAS: dict[str, str] = {
    "no_respira": r"\b(?:no respira(?! (?:con dificultad|mal|raro|agitad\w*|con esfuerzo))|dejo de respirar|no puede respirar|no esta respirando|se (?:esta )?ahog\w*|"
                  r"se asfixi\w*|no le entra (?:el )?aire)\b",
    "inconsciente": r"\b(?:inconsciente|no responde|no reacciona|perdio el conocimiento|no despierta|"
                    r"se desmayo y no)\b",
    "convulsion": r"\bconvulsion\w*\b",
    "sangrado_masivo": r"\b(?:sangra (?:mucho|demasiado|a chorros)|mucha sangre|no (?:para|deja) de sangrar|"
                       r"sangrado (?:abundante|masivo)|desangr\w*)\b",
    "cianosis": r"\b(?:labios|cara|piel) (?:morad\w*|azul\w*)\b",
    "arma": r"\b(?:arma de fuego|arma blanca|pistola|revolver|escopeta|disparo\w*|balazo\w*|baleado|"
            r"apunal\w*|acuchill\w*|armad[oa]s?|(?:tiene|tienen|lleva|llevan|porta|saco|muestra|con) (?:un |una )?armas?)\b",
    "explosivo": r"\b(?:bomba(?! de (?:agua|bencina|incendio|calor))|explosivo\w*|artefacto explosivo)\b",
    "autolesion": r"\b(?:suicid\w*|quitarse la vida|quitarme la vida|(?:me|se) quiere matar|me quiero matar|"
                  r"(?:me quiero|se quiere) morir(?! de (?:la )?(?:verguenza|risa|sueno|hambre|calor|frio|ganas))|(?:se|me) corto las venas|me corte las venas|"
                  r"(?:se|me) tomo (?:muchas|todas las) pastillas)\b",
    "atrapado_peligro": r"\batrapad\w* (?:en|por|bajo|entre) (?:el |la |los |las )?"
                        r"(?:fuego|humo|llamas|incendio|escombros|derrumbe)\b",
    "electrocucion": r"\b(?:electrocut\w*|recibio una descarga|le dio la corriente|pegada a la corriente)\b",
    # D.S. 34/2022: obstrucción de la vía aérea por reacción alérgica grave
    # (la garganta sola no: "garganta inflamada" suele ser un resfrío)
    "via_aerea_alergia": r"\b(?:hinch\w*|inflam\w*)(?: \w+){0,4} (?:lengua|glotis)\b|"
                         r"\blengua (?:hinchad\w*|inflamad\w*)\b|\bse (?:le|me) (?:esta )?cerrando la garganta\b|"
                         r"\bse (?:le|me) cierra la garganta\b",
    # Signos de ACV de inicio súbito: alteración del habla o cara desviada
    "acv": r"\b(?:habla|hablando) (?:raro|rara|enredad\w*|arrastrad\w*|traposo|incoherente)\b|"
           r"\b(?:cara|boca) (?:chueca|torcida|caida|desviada)\b|\bse le (?:torcio|cayo|chueco) la (?:cara|boca)\b",
    # Dolor torácico con dificultad respiratoria (posible infarto o embolía)
    "dolor_pecho_disnea": r"\b(?:dolor|duele|apret\w*|opresion)(?: \w+){0,4} pecho\b(?: \w+){0,6} "
                          r"(?:falta (?:el |de )?aire|ahog\w*|dificultad para respirar|no (?:puede|puedo) respirar bien)\b",
    # Pérdida súbita de la visión (secuela grave)
    "perdida_vision": r"\b(?:dejo de ver|deje de ver|perdio la vista|perdi la vista|se quedo ciego|me quede ciego|"
                      r"quedo ciega|no ve nada de (?:un|ningun) ojo)\b",
}

_COMPILADAS = {nombre: re.compile(patron) for nombre, patron in REGLAS.items()}
_NEGACION = re.compile(rf"\b{_NEGACIONES}\b")


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que 'convulsión' y 'convulsion' calcen igual."""
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def _esta_negada(texto: str, inicio: int, patron_negativo: bool) -> bool:
    if patron_negativo:
        return False  # "no respira" ya contiene su propia negación
    previas = texto[:inicio].split()[-_VENTANA_NEGACION:]
    return any(_NEGACION.fullmatch(p.strip(",.;:¿?¡!")) for p in previas)


def detectar_senales(mensaje: str) -> list[str]:
    """Devuelve los nombres de las reglas que se activan con el mensaje actual del usuario."""
    texto = _normalizar(mensaje)
    activadas = []
    for nombre, regex in _COMPILADAS.items():
        for m in regex.finditer(texto):
            if not _esta_negada(texto, m.start(), m.group(0).startswith("no ")):
                activadas.append(nombre)
                break
    return activadas
