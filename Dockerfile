# CPU-first image. Torch comes from the CPU wheel index so the image stays
# lean (~300 MB instead of ~2 GB with the default CUDA wheels).
# If no checkpoint is mounted, the API still runs on heuristic scorers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    TIER1_CHECKPOINT=/app/checkpoints/tier1_mlaad.pt

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.3" "numpy>=1.26" \
 && pip install --no-cache-dir \
      "fastapi>=0.115,<1.0" "uvicorn[standard]>=0.30,<1.0" "cryptography>=42,<46"

COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/
COPY checkpoints/ ./checkpoints/

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/audit && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "uvicorn voice_detection.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
