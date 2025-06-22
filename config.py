# config.py

import os
import pytz

# === API KEYS & Tokens ===
POLYGON_API_KEY        = os.getenv("POLYGON_API_KEY", "")
TELEGRAM_BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")

# === Service Identity & Modes ===
SERVICE_NAME           = os.getenv("SERVICE_NAME", "grond")
EXECUTION_MODE         = os.getenv("EXECUTION_MODE", "SCALP")
RISK_MODE              = os.getenv("RISK_MODE", "AGGRESSIVE")

# === Trading Parameters ===
# How many contracts/shares per order
ORDER_SIZE             = float(os.getenv("ORDER_SIZE", "1.0"))
# ε for ε–greedy bandit in signal allocation
BANDIT_EPSILON         = float(os.getenv("BANDIT_EPSILON", "0.1"))

# === Underlying Symbols ===
TICKERS                = [
    "TSLA", "AAPL", "MSFT", "NVDA",
    "NFLX", "AMZN", "META", "GOOG",
    "CL",   "NG"
]
OPTIONS_TICKERS        = TICKERS.copy()

# === Timezone & Market-Hour Labels ===
tz                     = pytz.timezone("US/Eastern")
TIME_OF_DAY_LABELS     = (
    "PRE_MARKET", "MORNING", "MIDDAY",
    "AFTERNOON", "AFTER_HOURS", "OFF_HOURS"
)

# === Rate Limiting (Polygon API) ===
RATE_LIMIT_PER_SEC     = 5.0
BURST_CAPACITY_SEC     = 10
RATE_LIMIT_PER_MIN     = 200.0
BURST_CAPACITY_MIN     = 200

# === API & Logging File Paths ===
SNAPSHOT_FILE          = "snapshots/options_oi_snapshots.json"
SIGNAL_TRACKER_FILE    = "logs/alladin_signal_performance_log.csv"
EXIT_LOG_FILE          = "logs/alladin_exit_log.csv"
STATUS_FILE            = "logs/alladin_status.txt"

# === Telegram Notifications ===
ENABLE_TELEGRAM        = True
TELEGRAM_COOLDOWN_SECONDS = 60

# === Strategy & Classifier Parameters ===
MIN_BREAKOUT_PROBABILITY = 0.5
EXIT_BARS                = 3

MOVEMENT_CONFIG_FILE      = "movement_config.json"
MOVEMENT_LOGIC_CONFIG_FILE= "movement_logic_config.json"

# === Data-Source & Ingestion Settings ===
DATA_SOURCE_MODE       = "hybrid"   # "rest", "webhook", or "hybrid"
REST_POLL_INTERVAL     = 10         # seconds between REST backfills
WEBHOOK_INITIAL_DELAY  = 300        # seconds to wait before first webhook backfill

# === Historical Lookbacks ===
LOOKBACK_BREAKOUT      = 5   # bars to scan for breakout probability
LOOKBACK_RISK_REWARD   = 20  # bars to compute ATR-based R/R
DEFAULT_CANDLE_LIMIT   = 500

# === TTL (time-to-live) Mappings ===
TTL_MAP = {
    "SHORT":  3,
    "MEDIUM": 10,
    "EXPIRY": 10000
}

# === Exit-Level Parameters ===
EXIT_PROFIT_TARGET     = 0.02   # 2% profit target
EXIT_STOP_LOSS         = -0.015 # –1.5% stop loss
