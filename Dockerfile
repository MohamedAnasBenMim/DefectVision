# Image de service : ONNX Runtime uniquement (pas de PyTorch), modèles inclus.
FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DV_MODELS_DIR=/srv/models DV_WEB_DIR=/srv/web
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home --uid 1000 dv
COPY models ./models
COPY web ./web
USER dv
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "uvicorn defectvision.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
