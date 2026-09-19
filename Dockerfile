FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ src/
COPY frontend/dist/ frontend/dist/

# Create directories
RUN mkdir -p outputs/phase5/visualizations

EXPOSE $PORT

CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
