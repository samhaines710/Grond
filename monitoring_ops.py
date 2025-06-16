import logging
from datetime import datetime
from flask import Flask, jsonify, request
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from config import METRICS_PORT, HTTP_PORT, SERVICE_NAME

# ── Structured Logging ────────────────────────────────────────────────────────
def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    ))
    root.handlers.clear()
    root.addHandler(handler)

# ── Prometheus Metrics ───────────────────────────────────────────────────────
REQUEST_COUNT   = Counter(f"{SERVICE_NAME}_http_requests_total", "Total HTTP requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram(f"{SERVICE_NAME}_http_request_latency_seconds", "HTTP request latency", ["endpoint"])
IN_PROGRESS     = Gauge(f"{SERVICE_NAME}_inprogress_requests", "In-flight HTTP requests")

app = Flask(__name__)

@app.before_request
def before_request():
    IN_PROGRESS.inc()
    request._start_time = datetime.utcnow()

@app.after_request
def after_request(response):
    # Record latency
    elapsed = (datetime.utcnow() - request._start_time).total_seconds()
    REQUEST_LATENCY.labels(endpoint=request.path).observe(elapsed)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path).inc()
    IN_PROGRESS.dec()
    return response

@app.route("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health")
def health():
    """
    Simple health-check endpoint.
    """
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z"), 200

def start_monitoring_server(port: int = METRICS_PORT):
    """
    Launch the Prometheus metrics HTTP server on the given port.
    """
    setup_logging()
    start_http_server(port)
    logging.getLogger("monitoring_ops").info(f"Prometheus metrics server started on port {port}")
