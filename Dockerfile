# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (Railway dynamically assigns a port via the $PORT environment variable)
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
