FROM python:3.11-slim

# Evitar errores de red en sistemas Alpine o slim
ENV DEBIAN_FRONTEND=noninteractive

# Instala librerías del sistema necesarias para OpenCV y DeepFace
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Establece directorio de trabajo
WORKDIR /app

# Copia los archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer el puerto 3001
EXPOSE 3001

# Ejecutar FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]
