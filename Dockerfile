# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by your Python packages
# (e.g., some matplotlib backends might need them, but 'Agg' usually doesn't)
# For now, let's assume no extra system deps are needed beyond what python:3.10-slim provides.
# If you encounter "missing library" errors during build or runtime, add them here:
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 some-other-lib && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app

# Make sure the directories for plots and PDFs will be writable by the app if created at runtime
# The os.makedirs(..., exist_ok=True) in your main.py should handle this.
# If not, you might need to create and set permissions here:
# RUN mkdir -p /app/static/plots /app/static/pdfs && chmod -R 777 /app/static
# However, it's generally better if the app creates these with appropriate permissions.

# Tell Docker that the container listens on port 7860 (or whatever your Uvicorn runs on)
EXPOSE 7860

# Define the command to run your application
# This will be overridden by Hugging Face Spaces if you specify an app_file,
# but it's good practice to have a CMD.
# Note: For Spaces, it often expects the app to be run by app.py or similar.
# We will specify the app file in the Space config later.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]