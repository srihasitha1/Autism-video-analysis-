FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, TensorFlow, and general C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1001 autisense \
    && chown -R autisense:autisense /app

USER autisense

# Create temp video directory
RUN mkdir -p /tmp/autisense_videos

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
