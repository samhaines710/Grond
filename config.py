import os
from dotenv import load_dotenv
import pytz

# Load .env (for secrets only)
load_dotenv()

def _get_env(name: str, required: bool = True, default=None):
    val = os.getenv(name, default)
    if required and not val:
        raise EnvironmentError(f"Environment variable '{name}' is required but not set.")
    return val

# ── Required Secrets ───────────────────────────────────────────────────────────
POLYGON_API_KEY    = _get_env("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = _get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _get_env("TELEGRAM_CHAT_ID")

# ── Engine & Strategy Defaults (no env vars needed) ──────────────────────────
from pricing_engines import EngineType
PRICER_ENGINE   = EngineType.QUANTLIB.value    # always use QuantLib
BANDIT_EPSILON  = 0.1                           # exploration rate
ORDER_SIZE      = 1.0                           # default contract size
EXECUTION_MODE  = "SIM"                         # manual/paper-only

# ── Dynamic Rate & Vol Settings ───────────────────────────────────────────────
DEFAULT_RISK_FREE_TENOR           = "2year"    # yield‐curve tenor
DEFAULT_VOLATILITY_LOOKBACK_DAYS = 30         # days for realized vol

# ── Monitoring & HTTP (use Render’s $PORT) ────────────────────────────────────
SERVICE_NAME = "grond_app"
PORT         = int(os.getenv("PORT", 8000))
METRICS_PORT = PORT
HTTP_PORT    = PORT

# ── Symbols & Timezone ────────────────────────────────────────────────────────
TICKERS         = ["TSLA","AAPL","MSFT","NVDA","NFLX","AMZN","META","GOOG","CL","NG"]
OPTIONS_TICKERS = TICKERS.copy()
tz              = pytz.timezone("US/Eastern")
