# ─────────────────────────────────────────────────────────────
# TradIA Backend — Dockerfile
# Multi-stage build: keeps final image lean (~500MB)
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate layer
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/
COPY .env.example ./.env.example

# Create directories needed at runtime
RUN mkdir -p backend/logs backend/data backend/models

# Non-root user for security
RUN useradd -m -u 1001 tradia && chown -R tradia:tradia /app
USER tradia

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.api.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
