# -------------------------------
# Base image with Pytorch (cpu only)
# -------------------------------
#FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime # uncomment this while using CUDA built
FROM python:3.10-slim
# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
# libgl1 is required for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt huggingface_hub gdown

# -------------------------------
# Copy entire project
# -------------------------------
# This copies:
# - app.py
# - serve.py
# - ventral/
# - lateral/
# - models/
# - utils/
# - checkpoints/
# - .github/ (harmless)
# - etc.
COPY . .
#====== download models at buildtime ========
#RUN python models/download_models.py #uncomment to download at build time

# -------------------------------
# Expose API port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Default command (ONLINE MODE)
# -------------------------------
# Can be overridden for CLI mode
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
