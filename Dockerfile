# Use an official Python runtime as a parent image
FROM python:3.10.7

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 5000

# Use Gunicorn with a configuration file
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]