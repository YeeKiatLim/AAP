# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Update and install necessary dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# cv2 Error Fix
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Tesseract Verification
RUN tesseract --version
RUN which tesseract

# Copy the application code to the working directory
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port on which the Flask app will run
EXPOSE 5000

# Run the application
CMD ["flask", "run"]
