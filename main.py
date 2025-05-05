from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import shutil
import os

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = DeepFace.verify(
            img1_path=file_location,
            img2_path=file_location,  # Se usa la misma imagen para detectar si hay rostro
            enforce_detection=True
        )

        os.remove(file_location)

        if result.get("verified") is True:
            return JSONResponse(content={"detected": True})
        else:
            return JSONResponse(content={"detected": False})
    except Exception as e:
        return JSONResponse(content={"detected": False, "error": str(e)}, status_code=500)

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        analysis = DeepFace.analyze(
            img_path=file_location,
            actions=["emotion", "age", "gender", "race"],
            enforce_detection=False
        )

        os.remove(file_location)
        return JSONResponse(content=analysis[0])  # Retorna solo el primer resultado

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
