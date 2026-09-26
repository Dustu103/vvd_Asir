# ============================================================
# Hoichoi Problem 2 — Production ML Server for Render / Cloud
# ============================================================
FROM python:3.11-slim

# System dependencies: ffmpeg (audio/video processing), libsndfile, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU and Transformers stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        transformers>=4.38.0 \
        peft>=0.9.0 \
        accelerate>=0.27.0 \
        soundfile>=0.12.1 \
        scipy>=1.11.0 \
        scikit-learn>=1.3.0 \
        sentencepiece \
        sacremoses \
        protobuf

# Copy application code, adapters, and deliverables
COPY problem_2_caption_diarization/ ./problem_2_caption_diarization/
COPY adapters/ ./adapters/
COPY deliverables/ ./deliverables/
COPY ml_server.py .
COPY run_problem2.py .

# Environment configuration for Render & Cloud
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ML_PORT=8000

EXPOSE 8000

# Start production ML inference server
CMD ["python", "ml_server.py"]
