# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better for Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg


# Copy only the project files you need
COPY Coughometer.py inference.py evaluate.py dnn_optimization.py ingestion.py ./ 
COPY saved_models ./saved_models
COPY static ./static

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "Coughometer:app", "--host", "0.0.0.0", "--port", "8000"]
