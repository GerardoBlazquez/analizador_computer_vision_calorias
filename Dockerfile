# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# copiar requerimientos
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copiar la app (modelos y datos deben añadirse al build o montarse en runtime)
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
