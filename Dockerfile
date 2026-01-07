# -------------------------------
# Base image with CUDA + PyTorch
# -------------------------------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
# libgl1 is required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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
RUN python models/download_models.py

# -------------------------------
# Expose API port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Default command (ONLINE MODE)
# -------------------------------
# Can be overridden for CLI mode
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
