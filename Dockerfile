FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create runtime dirs
RUN mkdir -p logs evidence

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: start both API and dashboard
# Override CMD to run batch or live
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]
