# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile  –  Market Trend Predictions API + Dashboard
#
# Two build targets:
#   api        → FastAPI / Uvicorn server
#   dashboard  → Streamlit dashboard
#
# Build the API image:
#   docker build --target api -t market-api .
#
# Build the Dashboard image:
#   docker build --target dashboard -t market-dashboard .
#
# (docker-compose.yml handles both automatically)
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps needed by psycopg2-binary and common ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# ── API target ────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Dashboard target ──────────────────────────────────────────────────────────
FROM base AS dashboard

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
