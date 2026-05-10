# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data
COPY src/ ./src/
COPY data/ ./data/

# Tell Python to look in the src directory for the 'sehat' package
ENV PYTHONPATH=/app/src

# Create a simple entry point for HuggingFace

RUN echo "import uvicorn\nfrom sehat.api.server import app\n\nif __name__ == '__main__':\n    uvicorn.run(app, host='0.0.0.0', port=7860)" > hf_app.py

# Expose the port HuggingFace expects
EXPOSE 7860

# Command to run the app
CMD ["python", "hf_app.py"]
