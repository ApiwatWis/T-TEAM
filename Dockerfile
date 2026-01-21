# 1. Use a STABLE base image (Debian Bookworm) to prevent future breakage
FROM python:3.9-slim-bookworm

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies
# UPDATED: 'libgl1-mesa-glx' -> 'libgl1'
# REMOVED: 'software-properties-common' (not needed and caused error)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose port
EXPOSE 8080

# 8. Run command
CMD ["streamlit", "run", "Home.py", "--server.port=8080", "--server.address=0.0.0.0"]