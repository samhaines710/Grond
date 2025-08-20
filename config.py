"""
Grond configuration.

Changes (2025-08-20):
- Removed NIO (unavailable / no options on your platform).
- Added BABA (Alibaba) and VEX to the default trading universe.

This module intentionally keeps defaults safe, with environment-variable overrides
for production control. All values are read once at import time.
"""

from __future__ import annotations

import os
from zoneinfo import ZoneInfo

# ──────────────────────────────────────────────────────────────────────────────
# Service identity & time zone
# ──────────────────────────────────────────────────────────────────────────────

SERVICE_NAME: str = os.getenv("SERVICE_NAME", "grond")

# Market timezone used across the orchestrator (datetime.now(tz))
# Default to New York; override if your venue differs.
MARKET_TZ: str = os.getenv("MARKET_TIMEZONE", "America/New_York")
tz: ZoneInfo = ZoneInfo(MARKET_TZ)

# ──────────────────────────────────────────────────────────────────────────────
# Core trading policy knobs (read by orchestrator/strategy)
# ──────────────────────────────────────────────────────────────────────────────

# Exploration rate for BanditAllocator (kept default 0.1, override via env)
BANDIT_EPSILON: float = float(os.getenv("BANDIT_EPSILON", "0.1"))

# Per-trade notional size (manual executor). Override per environment as needed.
ORDER_SIZE: float = float(os.getenv("ORDER_SIZE", "1.0"))

# ──────────────────────────────────────────────────────────────────────────────
# Universe
# - Defaults reflect the logs you’re running, with NIO→BABA and VEX added.
# - You can override the entire list via env TICKERS="TSLA,AAPL,...".
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_TICKERS = [
    "TSLA",
    "AAPL",
    "MSFT",
    "NVDA",
    "NFLX",
    "AMZN",
    "META",
    "GOOG",
    "IBIT",
    "BABA",  # substituted for NIO
    "VEX",   # newly added
]

_env_tickers = [t.strip().upper() for t in os.getenv("TICKERS", "").split(",") if t.strip()]
TICKERS = _env_tickers if _env_tickers else _DEFAULT_TICKERS

# ──────────────────────────────────────────────────────────────────────────────
# External APIs / Credentials (read-only here; used elsewhere)
# ──────────────────────────────────────────────────────────────────────────────

# Polygon API key (if blank, upstream code may fall back to calculated greeks)
POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

# Optional: S3 model location can be overridden globally if desired.
# (ml_classifier reads MODEL_URI directly from env; kept here as a single source)
MODEL_URI: str = os.getenv("MODEL_URI", "s3://bucketbuggypie/models/xgb_classifier.pipeline.joblib")

# ──────────────────────────────────────────────────────────────────────────────
# Sanity: expose explicit __all__ for clarity
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "SERVICE_NAME",
    "MARKET_TZ",
    "tz",
    "BANDIT_EPSILON",
    "ORDER_SIZE",
    "TICKERS",
    "POLYGON_API_KEY",
    "MODEL_URI",
]
