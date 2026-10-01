# QuantLab backend container image.
#
# This image only installs the core FastAPI service (see requirements.txt).
# Optional model runtimes (llama-cpp-python, torch/transformers, onnxruntime,
# ai-edge-litert) are NOT installed by default because they are large,
# platform-specific, and not every deployment needs every format. If your
# deployed models need one of these runtimes, add the corresponding
# `pip install ...` line to requirements.txt (see the comments there) before
# building this image, or extend this Dockerfile with an extra RUN step.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Keep Python output unbuffered so logs stream immediately in container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QNT_MODELS_DIR=/data/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY evaluation_dataset.json ./

# Persistent volume mount point for uploaded/imported model files. Map this to
# durable storage (Azure Files, an EBS/Persistent Volume, a bind mount, etc.)
# in production — model files can be large and must survive container restarts.
RUN mkdir -p /data/models
VOLUME ["/data/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
