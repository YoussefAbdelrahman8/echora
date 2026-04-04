FROM python:3.10-slim

# Install system dependencies required for OpenCV, audio support, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    espeak \
    alsa-utils \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the test script is executable
RUN chmod +x run_tests.sh

# Set default command to run tests
CMD ["./run_tests.sh"]
