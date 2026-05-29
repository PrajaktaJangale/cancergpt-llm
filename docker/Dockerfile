# ─── CancerGPT Dockerfile ────────────────────────────────────────────────────
# Multi-stage build for production-ready CancerGPT container

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ─── Install Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ────────────────────────────────────────────────────
COPY . .

# Create necessary directories
RUN mkdir -p /app/processed \
             /app/features \
             /app/rag_index \
             /app/reports \
             /app/logs \
             /app/database

# Set permissions
RUN chmod +x /app/docker/entrypoint.sh 2>/dev/null || true

# ─── Expose port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Run ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
