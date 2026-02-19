FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (maximise layer cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py categories.py ./

# Pre-create model cache directory with correct permissions
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker — model lives in one process; multiple workers = N copies of the model in RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--log-level", "info", "--timeout-keep-alive", "30"]