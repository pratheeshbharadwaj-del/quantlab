# All-in-one QuantLab image: builds the React UI and serves it from the FastAPI
# backend, so a single container (Render, Railway, Fly.io, Hugging Face Spaces,
# any Docker host) runs the whole app. Build from the repository root:
#   docker build -t quantlab .
#   docker run -p 8000:8000 --env-file .env -v quantlab-models:/data/models quantlab
# Then open http://localhost:8000

FROM node:20-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty API URL = the UI calls the API on the same origin.
ENV VITE_API_URL=""
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QNT_MODELS_DIR=/data/models \
    QNT_STATIC_DIR=/app/static \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/evaluation_dataset.json ./
COPY --from=frontend /app/dist ./static

RUN mkdir -p /data/models
VOLUME ["/data/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
