# 1. Use a stable, slim Python base image (Debian Bookworm)
# This prevents the "package not found" errors seen with older images
FROM python:3.9-slim-bookworm

# Set python to unbuffered to see logs in Cloud Run
ENV PYTHONUNBUFFERED=1

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required by OpenCV and Streamlit
# libgl1 & libglib2.0-0 are CRITICAL for cv2 to work
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file first (for better caching)
COPY requirements.txt .

# 5. Install Python dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Expose the port used by Cloud Run
EXPOSE 8080

# 8. Start Streamlit
# Railway sets $PORT dynamically - use it directly
CMD streamlit run Home.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.fileWatcherType=none \
    --server.headless=true
