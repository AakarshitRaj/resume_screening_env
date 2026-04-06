# ─── Resume Screening OpenEnv — Dockerfile ────────────────────────────────
# Builds a lightweight FastAPI server that exposes the OpenEnv HTTP interface.
#
# Build:
#   docker build -t resume-screening-env .
#
# Run locally:
#   docker run -p 7860:7860 resume-screening-env
#
# HF Space:
#   Set SDK to "docker" and push this repo. HF will build and expose port 7860.
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-participant"
LABEL description="Resume Screening OpenEnv — FastAPI server"
LABEL version="1.0.0"

# Non-root user for HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY env.py tasks.py data.py server.py ./

# Switch to non-root user
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the server
CMD ["python", "server.py"]
