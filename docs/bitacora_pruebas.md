# Bitácora de pruebas del chatbot ORION

Registro de cada prueba, qué se cambió antes de correrla y dónde quedó la evidencia. Las pruebas anteriores no se
borran: los resultados quedan en carpetas con fecha y los archivos de escenarios de cada test se conservan.

## Riesgo vital (terminado)

Meta fijada antes de la corrida final: recall ≥ 90 %.

| Versión | Set | Recall sistema | Precisión |
|---|---|---|---|
| V1 | principal, 152 casos | 88,2 % | |
| V2 (lista de ejemplos en el prompt) | principal | 86,3 % (regresión, se revirtió) | |
| V3 (detector paralelo + 4 reglas nuevas) | principal, -n 3 | 100 % (IC 98,5–100) | 93,4 % |
| V3 | validación, 300 casos, -n 3 | 99,5 % (IC 98,2–99,9) | 95,6 % |

Archivos: `docs/casos_riesgo_vital.json` (desarrollo) y `docs/casos_riesgo_vital_validacion.json` (validación,
ids 1001–1300, no se usó para ajustar).

## Orientación — test 1 (línea base, 2026-11-01)

- Escenarios: `docs/escenarios_orientacion.json` (32 conversaciones × 3 repeticiones, 147 turnos).
- Bot sin cambios de orientación. Juez: Claude (en el chat), con la misma rúbrica de `orientacion.py`.
- Resultado: 36/96 conversaciones aprobadas (56/96 sin contar aviso y largo). Aviso en el primer turno 14,6 %,
  máximo 4 oraciones 65,3 %, `debe` 63 %, `no_debe` 94,9 %, seguridad 95,2 %. 16 conversaciones con fallo crítico.
- Problemas principales: "No llames a nadie" al pedir un teléfono, inventar un contacto de Mantenimiento, decir que
  alguien "va en camino" sin que fuera cierto, no hacer la pregunta clave del protocolo, respuestas largas.

## Orientación — test 2 (cambios del 2026-11-08)

Escenarios: `docs/escenarios_orientacion_test2.json` (los 32 del test 1 sin cambios, más SAL-002-04 y SAL-001-05).

Cambios en el bot (`src/rag.py`):
1. El aviso legal lo agrega el código al inicio de la primera respuesta, siempre.
2. El estado del reporte lo agrega el código, según el estado real: con "Recibido" dice que el reporte fue
   notificado al equipo de la sede; con "En_camino" dice que un respondedor va en camino. El modelo no puede
   inventar avisos ni tiempos (se filtran).
3. Teléfonos: el bot no hace llamadas, pero si el usuario pide un teléfono le da el de la sede según el protocolo.
   Nunca dice "no llames". No existe contacto de Infraestructura ni de Mantenimiento.
4. Respuestas en segunda persona, resumidas, sin códigos de protocolo, terminando con la pregunta clave.
5. Si no está claro, pregunta si quien escribe es la persona afectada o está ayudando a otra.

Cambios en la evaluación: `--escenarios` para elegir el set, copia `escenarios_usados.json` en cada corrida, y las
frases fijas del sistema no cuentan como promesa ni para el largo.

Resultado (corrida `orientacion_test2_2026-11-08_1337`, juez: Claude con el mismo método del test 1):
- Aprobadas en los mismos 32 escenarios: 48/96 (antes 36/96). En los 34: 51/102.
- Conversaciones con fallo crítico: 9 (antes 16). Aviso 100 % (antes 14,6 %). Máximo 4 oraciones 82,1 % (antes 65,3 %).
- Se arreglaron los teléfonos (SEG-003-03, INF-002-04) y las promesas de "va en camino".
- Empeoró la prioridad (97 % → 67 %): el bot pregunta "¿eres tú quien se siente mal?" aunque ya se sabe, y lo pone
  primero. También copia la respuesta anterior en el turno 2.
- Riesgo vital sin regresión: recall 100 % en el set principal (`riesgo_2026-11-08_1344`, -n 1).
- Análisis completo: `analisis_test2.md`, junto a los resultados de la corrida.

## Orientación — test 3 (cambios del 2026-11-08, después del test 2)

Escenarios: `docs/escenarios_orientacion_test3.json`: los 34 del test 2 sin cambios (conjunto de desarrollo) más 8
escenarios nuevos de validación (`SAL-001-V1`, `SAL-002-V1`, `SAL-003-V1`, `SAL-004-V1`, `INF-001-V1`, `SEG-001-V1`,
`SEG-002-V1`, `SEG-003-V1`). Los de validación se escribieron antes de correr el test 3 y no se usaron para
ajustar el prompt: miden si los arreglos generalizan o solo se ajustaron a los escenarios conocidos.

Cambios en el bot (`src/rag.py`), cada uno ligado a un problema del test 2:
1. El aviso y la frase "Recibido" se quitan del historial antes de reenviarlo al modelo (`sin_frases_fijas`).
   Problema: el modelo copiaba la respuesta anterior completa, con aviso incluido.
2. Si el modelo escribe su propia versión del aviso, se borra y queda solo el oficial (`_AVISO_DEL_MODELO`).
3. Prompt, punto 5: preguntar "¿eres tú quien se siente mal?" solo en salud cuando el mensaje no dice quién es el
   afectado, y nunca como primera frase. Problema: la preguntaba casi siempre y la prioridad bajó a 67 %.
4. Prompt, punto 6 nuevo: pedir alejarse solo ante un peligro del entorno; si el afectado necesita compañía,
   acompañarlo. No sentar ni levantar a quien no responde. No ordenar evacuar ante una amenaza.
   Problema: "aléjate" ante ideación suicida, dolor de pecho y convulsión; sentar a alguien que no responde;
   evacuación ordenada por el bot ante el correo de bomba.
5. El filtro de promesas también corta "avisaremos", "notificaremos", "llamaremos".

Cambios en la evaluación: el reporte muestra los resultados por conjunto (desarrollo y validación). El chequeo
`sin_promesas` también detecta "avisaremos". Riesgo vital se vuelve a medir porque el prompt cambió.

Resultado: pendiente.
