# ===================================================================
# Face Clustering API - Docker Image
# 
# Build:  docker build -t face-clustering-api .
# Run:    docker run -p 8000:8000 face-clustering-api
# Test:   curl http://localhost:8000/health
# ===================================================================

# -------------------------------------------------------------------
# Stage 1: Builder (compiles dlib with all build dependencies)
# -------------------------------------------------------------------
FROM python:3.9-slim AS builder

# Install system dependencies for dlib compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libdlib-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (required for dlib compilation)
RUN pip install --no-cache-dir numpy==1.24.4

# Install dlib (takes time to compile)
RUN pip install --no-cache-dir dlib==19.24.6

# Install remaining Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# -------------------------------------------------------------------
# Stage 2: Runtime (clean image with only runtime dependencies)
# -------------------------------------------------------------------
FROM python:3.9-slim AS runtime

# Install only runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libx11-6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
