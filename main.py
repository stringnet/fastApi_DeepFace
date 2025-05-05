from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import shutil
import os

app = FastAPI()

# Permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Reemplaza con tu frontend si deseas restringirlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return JSONResponse(content=analysis[0])  # DeepFace.analyze devuelve una lista

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
