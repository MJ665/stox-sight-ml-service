# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# --- Environment Variables ---
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1
# For Hugging Face Spaces, the PORT variable is often injected (usually 7860).
# We will use it in the CMD line. Uvicorn's default is 8000 if PORT isn't set.
# ENV PORT=7860 # You can set it here, but HF might override or provide it.

# --- Create a non-root user and group ---
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} appuser && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home appuser

# Set path to include user's local bin for pip packages if installed with --user
ENV PATH="/home/appuser/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# --- Install Python Dependencies ---
# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the rest of your application, ensuring the 'appuser' owns them
COPY --chown=appuser:appuser . /app

# --- Application Directories (for runtime generated files) ---
# The /data directory is provided and managed by Hugging Face Spaces for persistent storage.
# Your Python code (main.py) is already designed to check for /data and use it,
# or fall back to a local directory (/app/local_stox_sight_outputs).
# We need to ensure the *fallback* directory within /app is created and writable by 'appuser'.
RUN mkdir -p /app/local_stox_sight_outputs/plots && \
    mkdir -p /app/local_stox_sight_outputs/pdfs && \
    chown -R appuser:appuser /app/local_stox_sight_outputs

RUN mkdir -p /app/data/plots && \
    mkdir -p /app/data/pdfs && \
    chown -R appuser:appuser /app/data

RUN mkdir -p /app/local_generated_files/plots && \
    mkdir -p /app/local_generated_files/pdfs && \
    chown -R appuser:appuser /app/local_generated_files

# --- Switch to the non-root user ---
USER appuser

# --- Expose Port & Run Application ---
# Expose the port the app will run on. HF uses this as a hint.
# The actual port mapping is handled by HF based on app_port in README or PORT env var.
EXPOSE 7860

# Command to run your application.
# Uvicorn will listen on 0.0.0.0 to be accessible from outside the container (within HF network).
# The port should ideally be taken from the PORT env var set by Hugging Face.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-7860}"]










# # Dockerfile

# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app



# # Set MPLCONFIGDIR to a writable temporary directory
# ENV MPLCONFIGDIR=/tmp/matplotlib

# ENV PYTHONUNBUFFERED=1 
# # For better logging in HF Spaces

# # Install system dependencies that might be needed by your Python packages
# # (e.g., some matplotlib backends might need them, but 'Agg' usually doesn't)
# # For now, let's assume no extra system deps are needed beyond what python:3.10-slim provides.
# # If you encounter "missing library" errors during build or runtime, add them here:
# # RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 some-other-lib && rm -rf /var/lib/apt/lists/*

# # Copy the requirements file into the container
# COPY ./requirements.txt /app/requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of your application code into the container
# COPY . /app

# RUN mkdir -p /app/static/plots && \
#     mkdir -p /app/static/pdfs

    
    

# # Create the local fallback directory structure within the image,
# # in case /data is not available/writable at runtime (e.g., during local Docker runs without /data mount)
# RUN mkdir -p /app/data/plots && \
#     mkdir -p /app/datat/pdfs
# RUN mkdir -p /app/local_generated_files_runtime/plots && \
#     mkdir -p /app/local_generated_files_runtime/pdfs


# # Make sure the directories for plots and PDFs will be writable by the app if created at runtime
# # The os.makedirs(..., exist_ok=True) in your main.py should handle this.
# # If not, you might need to create and set permissions here:
# # RUN mkdir -p /app/static/plots /app/static/pdfs && chmod -R 777 /app/static
# # However, it's generally better if the app creates these with appropriate permissions.

# # Tell Docker that the container listens on port 7860 (or whatever your Uvicorn runs on)
# EXPOSE 7860

# # Define the command to run your application
# # This will be overridden by Hugging Face Spaces if you specify an app_file,
# # but it's good practice to have a CMD.
# # Note: For Spaces, it often expects the app to be run by app.py or similar.
# # We will specify the app file in the Space config later.
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]












