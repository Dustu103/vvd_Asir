# ============================================================
# Hoichoi Problem 2 — Pipeline Test Container
# Pure code / fallback paths only (no model downloads)
# Base: python:3.11-slim + ffmpeg + minimal pip deps
# ============================================================
FROM python:3.11-slim

# System packages: ffmpeg (audio extraction) + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what we need (no model weights, no data)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy source
COPY problem_2_caption_diarization/ ./problem_2_caption_diarization/
COPY test_pipeline_docker.py .

# Output dir
RUN mkdir -p /out

# Default: run the test script
CMD ["python", "test_pipeline_docker.py", "--video", "/video/test.mp4", "--outdir", "/out"]
