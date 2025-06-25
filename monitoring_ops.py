# monitoring_ops.py

import errno
import logging
import threading
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

# ── Module-level flags to enforce idempotency ─────────────────────────────────
_metrics_server_started = False
_http_server_started   = False

# ── Structured Logging ─────────────────────────────────────────────────────────
def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
    ))
    root.handlers.clear()
    root.addHandler(handler)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter(
    f"{SERVICE_NAME}_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    f"{SERVICE_NAME}_http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"]
)
IN_PROGRESS     = Gauge(
    f"{SERVICE_NAME}_inprogress_requests",
    "In-flight HTTP requests"
)

# ── Flask App ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.before_request
def before_request():
    IN_PROGRESS.inc()
    request._start_time = datetime.utcnow()

@app.after_request
def after_request(response):
    elapsed = (datetime.utcnow() - request._start_time).total_seconds()
    REQUEST_LATENCY.labels(endpoint=request.path).observe(elapsed)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.path).inc()
    IN_PROGRESS.dec()
    return response

@app.route("/metrics")
def metrics():
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health")
def health():
    return jsonify(status="ok", time=datetime.utcnow().isoformat() + "Z"), 200

# ── Startup ─────────────────────────────────────────────────────────────────────
def start_monitoring_server():
    """
    1) launch Prometheus on METRICS_PORT (once, swallow EADDRINUSE)
    2) launch Flask on HTTP_PORT for /metrics and /health (once)
    """
    global _metrics_server_started, _http_server_started

    setup_logging()
    logger = logging.getLogger("monitoring_ops")

    # 1) Prometheus metrics endpoint
    if not _metrics_server_started:
        logger.info(f"Starting Prometheus metrics server on port {METRICS_PORT}")
        try:
            start_http_server(METRICS_PORT)
            _metrics_server_started = True
        except OSError as e:
            if getattr(e, "errno", None) == errno.EADDRINUSE:
                logger.warning(
                    f"Metrics port {METRICS_PORT} already in use; skipping metrics server start."
                )
                _metrics_server_started = True
            else:
                logger.error(
                    f"Failed to start Prometheus server on port {METRICS_PORT}: {e!r}"
                )
                raise
    else:
        logger.debug("Prometheus metrics server already started; skipping.")

    # 2) HTTP server for /metrics and /health
    if not _http_server_started:
        logger.info(f"Starting Flask health & metrics server on port {HTTP_PORT}")
        thread = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0",
                port=HTTP_PORT,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        )
        thread.start()
        _http_server_started = True
    else:
        logger.debug("Flask HTTP server already started; skipping.")