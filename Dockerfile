# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV DATABASE_URL=postgresql+asyncpg://nlp_search_owner:npg_e1RvVNES6GHj@ep-calm-bread-a1i8fb1i-pooler.ap-southeast-1.aws.neon.tech/nlp_search?sslmode=require
ENV REDIS_URL=rediss://default:AVNS_O95p6dowCmqCYs3Pv-Y@valkey-16df199-kalvium-4e7d.l.aivencloud.com:20093
ENV LOG_LEVEL=INFO

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 