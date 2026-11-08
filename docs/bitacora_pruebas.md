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

Resultado: pendiente.
