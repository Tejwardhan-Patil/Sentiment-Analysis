# Use an official base image with Python and R pre-installed
FROM rocker/r-ver:4.2.0

# Set environment variables for Python
ENV PYTHON_VERSION=3.9
ENV VIRTUAL_ENV=/opt/venv

# Install Python
RUN apt-get update && apt-get install -y \
    python$PYTHON_VERSION \
    python$PYTHON_VERSION-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for Python
RUN python$PYTHON_VERSION -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install R dependencies
COPY deployment/api/packages.R .
RUN Rscript packages.R

# Copy application files
COPY . /app
WORKDIR /app

# Expose port for the API
EXPOSE 8080

# Command to run the Flask or FastAPI app for Python models
CMD ["python", "deployment/api/app.py"]