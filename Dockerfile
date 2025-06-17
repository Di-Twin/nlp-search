# Use Python 3.11 slim image as base with security updates
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Create a non-root user with specific UID between 10000-20000
RUN groupadd -r appuser && useradd -r -g appuser -u 10001 appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Copy requirements files
COPY requirements-base.txt .
COPY requirements-ml.txt .

# Install Python dependencies with security checks
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip install --no-cache-dir -r requirements-ml.txt --timeout 300

# Copy the rest of the application
COPY . .

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user using UID
USER 10001

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 