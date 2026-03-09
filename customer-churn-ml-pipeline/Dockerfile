# Dockerfile
# ----------
# Multi-stage build for the Customer Churn Prediction API.
#
# Stage 1 (builder): Install dependencies in a venv.
# Stage 2 (runtime): Copy only the venv + app code – no build tools.
#                    Results in a smaller, more secure final image.
#
# MLOps role: Reproducible, environment-agnostic deployment unit.
#             The same image runs locally, in CI, and in production.
#
# Build:  docker build -t churn-api .
# Run  :  docker run -p 8000:8000 churn-api

# ---------------------------------------------------------------------------
# Stage 1 – Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install pip dependencies into an isolated virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 – Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security best practice: don't run as root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY src/      ./src/
COPY api/      ./api/
COPY models/   ./models/

# Activate the venv for all subsequent RUN / CMD instructions
ENV PATH="/opt/venv/bin:$PATH"

# Environment variables (overridable at runtime with -e)
ENV LOG_LEVEL=INFO \
    MODEL_NAME=best_model \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check – Docker / Kubernetes will poll this endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
