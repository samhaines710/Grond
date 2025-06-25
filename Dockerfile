# 1) Use a slim Python base image
FROM python:3.10-slim

# 2) Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 3) Install system libraries for QuantLib, linear algebra, Zipline, CVX/Ecos, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libquantlib0-dev \
        libblas-dev \
        liblapack-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        gfortran \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 4) Create and switch to the app directory
WORKDIR /app

# 5) Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# 6) Copy the rest of your application code
COPY . /app

# 7) Expose your main HTTP port (Flask + Prometheus)
EXPOSE 10000 8000

# 8) Default command: run the main orchestrator loop
CMD ["python", "grond_orchestrator.py"]
