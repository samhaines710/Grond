# utils/logging_utils.py

"""
Structured logging and Prometheus counters for REST calls and 429s,
plus a simple status‐file writer.
"""

import os
import logging
from datetime import datetime
from prometheus_client import Counter
from config import STATUS_FILE, SERVICE_NAME

# ── Structured Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
# Clear existing handlers to avoid duplicates
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
    '"module":"%(module)s","message":"%(message)s"}'
))
logger.addHandler(handler)

# ── Prometheus Counters ──────────────────────────────────────────────────────────
# (server is started in monitoring_ops.start_monitoring_server)
REST_CALLS = Counter(
    f"{SERVICE_NAME}_rest_calls_total",
    "Total REST calls made"
)
REST_429 = Counter(
    f"{SERVICE_NAME}_rest_429_total",
    "Total HTTP 429 responses received"
)

# ── Status File Writer ───────────────────────────────────────────────────────────
def write_status(msg: str) -> None:
    """
    Log a status message (to stdout via structured logger) and append it to STATUS_FILE.
    """
    # Log via structured JSON logger
    logger.info(msg)

    # Append timestamped message to STATUS_FILE
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        os.makedirs(os.path.dirname(STATUS_FILE) or ".", exist_ok=True)
        with open(STATUS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")
    except Exception as e:
        logger.error(f"Failed to write status to {STATUS_FILE}: {e}")
