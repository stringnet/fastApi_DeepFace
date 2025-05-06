# main.py (del Face Detector) - MODIFICADO

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
import uvicorn
import shutil
import os
import httpx # Necesario para hacer la petición HTTP POST
import logging # Añadido para logs

app = FastAPI()

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL del endpoint de la API del Espectro que recibe el texto
ESPECTRO_API_ENDPOINT = "https://espectroapi.scanmee.io/ws-message" # Asegúrate que esta URL sea correcta

# Montamos el frontend (sin cambios)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Servimos el index.html (sin cambios)
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    # Asegúrate que la ruta 'static/index.html' es correcta en tu estructura
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
         logger.error("Error: No se encontró static/index.html")
         return HTMLResponse(content="<html><body>Error: index.html no encontrado.</body></html>", status_code=404)


# Endpoint para análisis facial (MODIFICADO)
@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...), actions: str = Form("emotion")):
    temp_file_path = f"temp_{file.filename}"
    analysis_result = None # Para guardar el resultado del análisis
    dominant_emotion = None # Para guardar la emoción detectada

    try:
        # Guardar archivo temporalmente (igual que antes)
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        # Analizar con DeepFace (igual que antes)
        # Asegúrate que 'actions' incluya 'emotion' si quieres la emoción
        action_list = [action.strip() for action in actions.split(",") if action.strip()]
        if not action_list:
            action_list = ['emotion'] # Default a emoción si no se especifica nada válido

        logger.info(f"Analizando imagen: {file.filename} con acciones: {action_list}")
        result = DeepFace.analyze(img_path=temp_file_path, actions=action_list, enforce_detection=False) # enforce=False por si la imagen es un frame directo
        analysis_result = result # Guardar resultado para devolverlo al final

        # --- INICIO: Lógica de Integración con Espectro ---

        # DeepFace.analyze devuelve una lista, incluso si solo hay una cara
        if isinstance(result, list) and len(result) > 0:
            # Extraer emoción del primer resultado (asumimos una cara)
             # Verificar que 'dominant_emotion' existe en el diccionario
            if 'dominant_emotion' in result[0]:
                dominant_emotion = result[0].get("dominant_emotion", "desconocida").lower()
                logger.info(f"Emoción detectada: {dominant_emotion}")

                # Crear el mensaje/prompt para el espectro basado en la emoción
                if dominant_emotion == "happy":
                    prompt_text = "He detectado un usuario que parece estar feliz. Salúdalo con entusiasmo."
                elif dominant_emotion == "sad":
                    prompt_text = "He detectado un usuario que parece estar triste. Salúdalo con empatía y ofrécele ánimo."
                elif dominant_emotion == "neutral":
                    prompt_text = "He detectado un usuario con expresión neutral. Inicia una conversación amable y abierta."
                # Puedes añadir más casos para 'angry', 'surprise', 'fear', etc.
                else:
                    prompt_text = f"He detectado un usuario (emoción: {dominant_emotion}). Inicia una conversación."

                # Preparar el payload JSON para espectroapi
                payload = {"text": prompt_text}

                # Enviar la petición POST a espectroapi usando httpx
                logger.info(f"Enviando prompt a Espectro API: {prompt_text}")
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(ESPECTRO_API_ENDPOINT, json=payload, timeout=10.0) # Timeout de 10s
                        response.raise_for_status() # Lanza excepción para errores HTTP 4xx/5xx
                        logger.info(f"Prompt enviado exitosamente a Espectro API. Respuesta: {response.status_code}")
                        # Podrías incluso leer response.json() si esperas algo de vuelta de espectroapi
                        # espectro_api_response = response.json()
                        # logger.info(f"Respuesta de Espectro API: {espectro_api_response}")
                    except httpx.RequestError as exc:
                        logger.error(f"Error de red al contactar Espectro API ({exc.request.url!r}): {exc}")
                    except httpx.HTTPStatusError as exc:
                        logger.error(f"Error de estado desde Espectro API ({exc.request.url!r}): {exc.response.status_code} - Respuesta: {exc.response.text}")
                    except Exception as e_http:
                         logger.error(f"Error inesperado enviando a Espectro API: {e_http}", exc_info=True)
            else:
                 logger.warning("No se encontró 'dominant_emotion' en el resultado de DeepFace.")

        else:
            logger.warning("DeepFace no devolvió resultados válidos o no detectó caras.")

        # --- FIN: Lógica de Integración con Espectro ---

    except Exception as e:
        logger.error(f"Error en /analyze: {str(e)}", exc_info=True)
        # Asegurarse de borrar el archivo temporal incluso si hay error
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_remove:
                 logger.error(f"Error eliminando archivo temporal {temp_file_path}: {e_remove}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # Asegurar la eliminación del archivo temporal en todos los casos
         if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Archivo temporal eliminado: {temp_file_path}")
            except Exception as e_remove:
                 logger.error(f"Error eliminando archivo temporal {temp_file_path} en finally: {e_remove}")


    # Devolver el resultado original del análisis facial al cliente que subió la imagen
    logger.info(f"Devolviendo resultado del análisis: {analysis_result}")
    # Usar JSONResponse para asegurar formato correcto si 'analysis_result' es complejo
    return JSONResponse(content={"status": "success", "data": analysis_result})


# Solo si corres localmente (sin cambios)
def run_local():
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)

if __name__ == "__main__":
    run_local()
