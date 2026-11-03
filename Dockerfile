# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required by librosa (libsndfile) and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Render assigns (or default to 10000/8000; Render uses $PORT dynamically)
EXPOSE 10000

# Command to run the application using Uvicorn, binding to $PORT
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}