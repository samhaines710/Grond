# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Set an env var for overrideable model path
ENV ML_MODEL_PATH=/app/models/xgb_classifier.pipeline.joblib

# Create and switch to app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Copy the pre-trained model artifact into the image
# (Make sure your local repo has models/xgb_classifier.pipeline.joblib)
COPY models/xgb_classifier.pipeline.joblib /app/models/xgb_classifier.pipeline.joblib

# (Optional) Expose Prometheus and HTTP ports
EXPOSE 8000
EXPOSE 10000

# Launch the orchestrator
CMD ["python", "grond_orchestrator.py"]