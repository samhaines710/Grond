# config.py

import os
import pytz

# === API KEYS & TOKENS ===
POLYGON_API_KEY         = os.getenv("POLYGON_API_KEY", "")
TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")

# === SERVICE IDENTITY & MODES ===
SERVICE_NAME            = os.getenv("SERVICE_NAME", "grond")
EXECUTION_MODE          = os.getenv("EXECUTION_MODE", "SCALP")
RISK_MODE               = os.getenv("RISK_MODE", "AGGRESSIVE")

# === TRADING PARAMETERS ===
# Number of contracts/shares per order
ORDER_SIZE              = float(os.getenv("ORDER_SIZE", "1.0"))
# ε for ε–greedy bandit in signal allocation
BANDIT_EPSILON          = float(os.getenv("BANDIT_EPSILON", "0.1"))

# === UNDERLYING SYMBOLS ===
TICKERS                 = [
    "TSLA", "AAPL", "MSFT", "NVDA",
    "NFLX", "AMZN", "META", "GOOG",
    "CL",   "NG"
]
OPTIONS_TICKERS         = TICKERS.copy()

# === TIMEZONE & MARKET‐HOUR LABELS ===
tz                      = pytz.timezone("US/Eastern")
TIME_OF_DAY_LABELS      = (
    "PRE_MARKET", "MORNING", "MIDDAY",
    "AFTERNOON", "AFTER_HOURS", "OFF_HOURS"
)

# === RATE LIMITING (POLYGON API) ===
RATE_LIMIT_PER_SEC      = 5.0
BURST_CAPACITY_SEC      = 10
RATE_LIMIT_PER_MIN      = 200.0
BURST_CAPACITY_MIN      = 200

# === FILE PATHS FOR SNAPSHOTS & LOGS ===
SNAPSHOT_FILE           = "snapshots/options_oi_snapshots.json"
SIGNAL_TRACKER_FILE     = "logs/alladin_signal_performance_log.csv"
EXIT_LOG_FILE           = "logs/alladin_exit_log.csv"
STATUS_FILE             = "logs/alladin_status.txt"

# === METRICS & HTTP PORTS ===
# Port for Prometheus to scrape metrics
METRICS_PORT            = int(os.getenv("METRICS_PORT", "8000"))
# Port for HTTP health‐check / metrics endpoint
HTTP_PORT               = int(os.getenv("HTTP_PORT",    "10000"))

# === TELEGRAM NOTIFICATIONS ===
ENABLE_TELEGRAM         = True
TELEGRAM_COOLDOWN_SECONDS = 60

# === STRATEGY & CLASSIFIER PARAMETERS ===
MIN_BREAKOUT_PROBABILITY = 0.5
EXIT_BARS                = 3

MOVEMENT_CONFIG_FILE     = "movement_config.json"
MOVEMENT_LOGIC_CONFIG_FILE = "movement_logic_config.json"

# === DATA SOURCE & INGESTION SETTINGS ===
DATA_SOURCE_MODE        = "hybrid"    # "rest", "webhook", or "hybrid"
REST_POLL_INTERVAL      = 10          # seconds between REST backfills
WEBHOOK_INITIAL_DELAY   = 300         # seconds to wait before first webhook backfill

# === HISTORICAL LOOKBACKS ===
LOOKBACK_BREAKOUT       = 5   # bars to scan for breakout probability
LOOKBACK_RISK_REWARD    = 20  # bars to compute ATR‐based risk/reward
DEFAULT_CANDLE_LIMIT    = 500

# === TTL (TIME‐TO‐LIVE) MAPPINGS ===
TTL_MAP = {
    "SHORT":   3,
    "MEDIUM": 10,
    "EXPIRY": 10000
}

# === EXIT‐LEVEL PARAMETERS ===
EXIT_PROFIT_TARGET      = 0.02   # 2% profit target
EXIT_STOP_LOSS          = -0.015 # ‐1.5% stop loss
