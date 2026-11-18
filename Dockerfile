FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEM0_TELEMETRY=False

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/documents ./data/documents
COPY pytest.ini .

RUN mkdir -p data/qdrant_storage data/mem0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Secrets are never baked into the image: pass them with --env-file / compose env_file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
