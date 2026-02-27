# ════════════════════════════════════════════════════════════════════════════
# Dockerfile – Rural Health-Camp Logistics Agent
# Base image : python:3.10-slim  (Debian Bullseye, ~130 MB compressed)
# Exposes    : 8501  (Streamlit default)
# Build      : docker build -t rural-health-agent .
# Run        : docker run -p 8501:8501 rural-health-agent
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# (critical for Streamlit logs to appear in Docker output)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ────────────────────────────────────────────────────────
# OR-Tools ships as a Python wheel but needs libprotobuf at runtime.
# We also install curl for Docker HEALTHCHECK and procps for debugging.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libprotobuf-dev \
        protobuf-compiler \
        libssl-dev \
        curl \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security best practice) ────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy only requirements first so Docker layer-caches the pip install step.
# The heavy OR-Tools / scikit-learn layers won't re-run unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
# Flat-directory layout:  all .py files + data/ + models/ + .streamlit/
COPY --chown=appuser:appgroup . .

# ── Create writable directories the app needs at runtime ──────────────────────
RUN mkdir -p data models \
 && chown -R appuser:appgroup data models

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck: Streamlit serves a /_stcore/health endpoint ─────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
