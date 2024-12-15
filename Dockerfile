# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (or any port Railway assigns via $PORT)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
