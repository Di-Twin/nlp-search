FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use longer timeouts
RUN pip config set global.timeout 300

# Copy requirements first to leverage Docker cache
COPY requirements-base.txt .
COPY requirements-ml.txt .

# Install base dependencies
RUN pip install -r requirements-base.txt

# Install ML dependencies with increased timeout
RUN pip install -r requirements-ml.txt --timeout 300

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 