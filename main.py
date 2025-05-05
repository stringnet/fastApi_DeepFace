from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
import io
import uvicorn
import numpy as np
import base64

app = FastAPI()

# Configurar CORS para permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://facedetector.scanmee.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        np_image = np.array(image)

        analysis = DeepFace.analyze(np_image, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']

        return {"emotion": dominant_emotion}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
