"""Prueba por consola (sin API). Mantiene el historial en memoria como lo haría el backend.
Uso: python preguntar.py "Violencia interpersonal" "Edificio A"
"""
import json
import sys

from rag import responder

if __name__ == "__main__":
    contexto = {
        "subcategoria": sys.argv[1] if len(sys.argv) > 1 else None,
        "ubicacion": sys.argv[2] if len(sys.argv) > 2 else None,
        "estado_incidente": "Recibido",
    }
    print("🤖 ORION Chatbot — Escribe 'salir' para terminar\n")
    historial = []
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() in ["salir", "exit", "q"]:
            print("👋 Hasta luego")
            break
        historial.append({"emisor": "USUARIO", "contenido": pregunta})
        r = responder(historial, contexto)
        historial.append({"emisor": "BOT", "contenido": r["respuesta_texto"]})
        print(f"ORION: {r['respuesta_texto']}")
        print(f"   {json.dumps({k: v for k, v in r.items() if k != 'respuesta_texto'}, ensure_ascii=False)}\n")
