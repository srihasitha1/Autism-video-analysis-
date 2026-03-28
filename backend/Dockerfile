# AutiSense Backend - Docker Image
# Python 3.11-slim for bcrypt/passlib compatibility

FROM python:3.11-slim

# ── System dependencies ─────────────────────────────────────────────────────
# Required for psycopg2, OpenCV, and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pin bcrypt to 4.0.1 for passlib 1.7.4 compatibility
# bcrypt 5.x causes: AttributeError: module 'bcrypt' has no attribute '__about__'
RUN pip install --no-cache-dir "bcrypt==4.0.1"

# Install tf_keras for loading older Keras models
RUN pip install --no-cache-dir tf_keras

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Environment variables (defaults) ─────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=development

# ── Startup command ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
