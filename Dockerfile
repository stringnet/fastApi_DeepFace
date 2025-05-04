FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libgl1 && \
    pip install --no-cache-dir fastapi uvicorn deepface pillow numpy python-multipart

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
