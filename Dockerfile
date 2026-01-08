# -------------------------------
# Base image with Pytorch (cpu only)
# -------------------------------
#FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime # uncomment this while using CUDA built

# Use an official Python image
FROM python:3.11-slim

# Set environment variables to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Copy your requirements file
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \

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

# Default command
CMD ["python", "app.py"]

