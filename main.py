from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
import uvicorn
import shutil
import os
import httpx

app = FastAPI()

# Montamos el frontend desde la carpeta "static" en /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Servimos el index.html manualmente en /
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
        
# Nuevo endpoint para análisis facial completo
@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...), actions: str = Form("emotion")):
    try:
        contents = await file.read()
        temp_file_path = f"temp_{file.filename}"

        with open(temp_file_path, "wb") as f:
            f.write(contents)

        result = DeepFace.analyze(img_path=temp_file_path, actions=actions.split(","))

        os.remove(temp_file_path)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Solo si corres localmente
def run_local():
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)

if __name__ == "__main__":
    run_local()
