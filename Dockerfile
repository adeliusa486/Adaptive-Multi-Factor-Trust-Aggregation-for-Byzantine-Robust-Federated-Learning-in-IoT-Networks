# =============================================================================
# AMFTA — Multi-stage Docker Build
# =============================================================================
# Build:  docker build -t amfta-fl:latest .
# Run:    docker run --rm -it amfta-fl:latest
# GPU:    docker run --gpus all amfta-fl:latest
# API:    docker run -p 8000:8000 amfta-fl:latest uvicorn api.main:app --host 0.0.0.0 --port 8000

# ─── Stage 1: Base Python environment ────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Python dependencies ────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ─── Stage 3: Application ────────────────────────────────────────────────────
FROM deps AS app

COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/server data/partitions \
             results figures checkpoints

# Install package in editable mode
RUN pip install -e . --no-deps

# ─── Stage 4: Production API ─────────────────────────────────────────────────
FROM app AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV MODEL_CHECKPOINT_PATH=/app/checkpoints/amfta_best.pt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ─── Stage 5: Training ────────────────────────────────────────────────────────
FROM app AS train

CMD ["python", "experiments/run_main.py", "--use_synthetic", "--num_rounds", "10"]
