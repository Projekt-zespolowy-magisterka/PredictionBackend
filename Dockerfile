# Use a lightweight base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies with no cache to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application files
COPY . .

# Expose the application port
EXPOSE 5000

# Use Gunicorn with a configuration file
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]