# utils/logging_utils.py

import os
import logging
from datetime import datetime
from prometheus_client import Counter, start_http_server
from config import STATUS_FILE

# ─── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("alladin")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
))
logger.addHandler(stream_handler)

# ─── Prometheus metrics ─────────────────────────────────────────────────────────
REST_CALLS = Counter("rest_calls_total", "Total REST calls made")
REST_429   = Counter("rest_429_total", "Total HTTP 429 responses")

try:
    start_http_server(8000)
except OSError:
    logger.info("Prometheus port 8000 already in use; skipping metrics endpoint")

# ─── Status writer ──────────────────────────────────────────────────────────────
def write_status(msg: str):
    """
    Write a timestamped status line to both stdout (via logger) and the STATUS_FILE.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(msg)
    try:
        os.makedirs(os.path.dirname(STATUS_FILE) or ".", exist_ok=True)
        with open(STATUS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")
    except Exception as e:
        logger.error(f"Failed to write status: {e}")
