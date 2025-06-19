import os
import json
import csv
import math
import time
import threading
import asyncio
from datetime import datetime, date, timedelta, time as dtime
from collections import deque
import logging

import requests
import pytz
from telegram import Bot, error as tg_error
import websocket
from functools import wraps
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prometheus_client import Counter, start_http_server
from pandas_market_calendars import get_calendar

# ── Load config dynamically ────────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location("config", "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# ── Configuration variables ────────────────────────────────────────────────────
POLYGON_API_KEY         = os.getenv("POLYGON_API_KEY", "")
TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
bot = Bot(token=TELEGRAM_BOT_TOKEN) if (
    POLYGON_API_KEY and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and getattr(config, "ENABLE_TELEGRAM", False)
) else None

TELEGRAM_COOLDOWN_SECONDS = getattr(config, "TELEGRAM_COOLDOWN_SECONDS", 60)
_last_telegram_ts = 0

tz = config.tz

SNAPSHOT_FILE            = config.SNAPSHOT_FILE
SIGNAL_TRACKER_FILE      = config.SIGNAL_TRACKER_FILE
EXIT_LOG_FILE            = config.EXIT_LOG_FILE
STATUS_FILE              = config.STATUS_FILE
TICKERS                  = config.TICKERS
OPTIONS_TICKERS          = config.OPTIONS_TICKERS
DATA_SOURCE_MODE         = config.DATA_SOURCE_MODE
REST_POLL_INTERVAL       = config.REST_POLL_INTERVAL
WEBHOOK_INITIAL_DELAY    = config.WEBHOOK_INITIAL_DELAY
RISK_FREE_RATE           = getattr(config, "RISK_FREE_RATE", 0.0)
DIVIDEND_YIELDS          = getattr(config, "DIVIDEND_YIELDS", {})
DEFAULT_VOLATILITIES     = getattr(config, "DEFAULT_VOLATILITIES", {})
LOOKBACK_BREAKOUT        = config.LOOKBACK_BREAKOUT
LOOKBACK_RISK_REWARD     = config.LOOKBACK_RISK_REWARD
DEFAULT_CANDLE_LIMIT     = config.DEFAULT_CANDLE_LIMIT
MIN_BREAKOUT_PROBABILITY = config.MIN_BREAKOUT_PROBABILITY
EXIT_BARS                = config.EXIT_BARS
EXECUTION_MODE           = config.EXECUTION_MODE
RISK_MODE                = config.RISK_MODE

RATE_LIMIT_PER_SEC       = config.RATE_LIMIT_PER_SEC
BURST_CAPACITY_SEC       = config.BURST_CAPACITY_SEC
RATE_LIMIT_PER_MIN       = config.RATE_LIMIT_PER_MIN
BURST_CAPACITY_MIN       = config.BURST_CAPACITY_MIN

TTL_MAP                  = config.TTL_MAP

ALL_GREEK_KEYS = [
    "delta", "gamma", "theta_day", "theta_5m", "vega",
    "rho", "vanna", "vomma", "charm",
    "veta", "speed", "zomma", "color"
]

# ── In-memory caches & locks ───────────────────────────────────────────────────
REALTIME_CANDLES        = {t: deque(maxlen=200) for t in TICKERS}
REALTIME_OPTIONS        = {t: deque(maxlen=100) for t in OPTIONS_TICKERS}
realtime_candles_lock   = threading.Lock()
realtime_options_lock   = threading.Lock()
greeks_cache_lock       = threading.Lock()
CACHE_TIMEOUT           = 600
GREEK_CACHE             = {}
SNAPSHOT_CACHE          = {}

# ── Logging & Prometheus metrics ───────────────────────────────────────────────
logger = logging.getLogger("grond_utils")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
))
logger.addHandler(stream_handler)

REST_CALLS = Counter("rest_calls_total", "Total REST calls made")
REST_429   = Counter("rest_429_total", "Total REST 429 responses")
try:
    start_http_server(config.METRICS_PORT)
except OSError:
    logger.info("Prometheus port already in use; skipping metrics endpoint")


def write_status(msg: str):
    """Log and append status to STATUS_FILE."""
    ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(msg)
    try:
        os.makedirs(os.path.dirname(STATUS_FILE) or ".", exist_ok=True)
        with open(STATUS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")
    except Exception as e:
        logger.error(f"Failed to write status: {e}")


# ── Market calendar helper ────────────────────────────────────────────────────
nyse_calendar = get_calendar("NYSE")
def is_market_open_today(as_of: datetime | None = None) -> bool:
    d = (as_of or datetime.now(tz)).date()
    return not nyse_calendar.valid_days(start_date=d, end_date=d).empty


# ── Circuit breaker ───────────────────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self, threshold=5, cooloff_secs=60):
        self.fail_count   = 0
        self.threshold    = threshold
        self.open_until   = None
        self.cooloff_secs = cooloff_secs

    def record_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.open_until = datetime.now(tz) + timedelta(seconds=self.cooloff_secs)
            write_status(f"Circuit open until {self.open_until.isoformat()}")

    def record_success(self):
        self.fail_count = 0
        self.open_until = None

    def is_open(self) -> bool:
        if self.open_until and datetime.now(tz) < self.open_until:
            return True
        self.open_until = None
        return False

_circuit_breaker = CircuitBreaker()


# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rate_per_sec, burst_sec, rate_per_min, burst_min):
        self.rate_sec = rate_per_sec
        self.cap_sec  = burst_sec
        self.tokens_s = burst_sec
        self.ts_s     = time.time()
        self.rate_min = rate_per_min / 60.0
        self.cap_min  = burst_min
        self.tokens_m = burst_min
        self.ts_m     = time.time()
        self.lock     = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            ds  = now - self.ts_s
            if ds > 0:
                self.tokens_s = min(self.cap_sec, self.tokens_s + ds * self.rate_sec)
                self.ts_s      = now
            dm  = now - self.ts_m
            if dm > 0:
                self.tokens_m = min(self.cap_min, self.tokens_m + dm * self.rate_min)
                self.ts_m      = now
            if self.tokens_s >= 1 and self.tokens_m >= 1:
                self.tokens_s -= 1
                self.tokens_m -= 1
                return True
            return False

    def wait(self):
        while not self.acquire():
            time.sleep(0.01)

_ratelimiter = RateLimiter(
    RATE_LIMIT_PER_SEC, BURST_CAPACITY_SEC,
    RATE_LIMIT_PER_MIN, BURST_CAPACITY_MIN
)

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _ratelimiter.wait()
        return func(*args, **kwargs)
    return wrapper


# ── Telegram helper ────────────────────────────────────────────────────────────
def send_telegram(message: str):
    global _last_telegram_ts
    now = time.time()
    if now - _last_telegram_ts < TELEGRAM_COOLDOWN_SECONDS:
        write_status("Skipping Telegram: cooldown active")
        return
    _last_telegram_ts = now

    if not bot:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
    except tg_error.TimedOut:
        write_status("Telegram timed out—skipping")
    except tg_error.TelegramError as e:
        if "flood control" not in str(e).lower():
            write_status(f"Telegram error: {e}")


# ── HTTP fetch helpers ────────────────────────────────────────────────────────
@rate_limited
def _get_json(url: str, ticker: str = "") -> dict:
    if _circuit_breaker.is_open():
        write_status(f"Skipping REST call to {url} (circuit open)")
        return {}
    REST_CALLS.inc()
    try:
        resp = requests.get(url, timeout=7)
        if resp.status_code == 429:
            REST_429.inc()
            _circuit_breaker.record_failure()
            write_status(f"429 from Polygon for {ticker}")
            return {}
        resp.raise_for_status()
        _circuit_breaker.record_success()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (500, 502, 503):
            _circuit_breaker.record_failure()
        write_status(f"HTTP error for {ticker}: {e}")
    except Exception as e:
        write_status(f"Request exception for {ticker}: {e}")
    return {}

def safe_fetch_polygon_data(url: str, ticker: str = "", retries: int = 3) -> dict:
    for attempt in range(retries):
        data = _get_json(url, ticker)
        if data:
            return data
        time.sleep(2 ** attempt)
    write_status(f"Failed to fetch data for {ticker} after {retries} attempts")
    return {}


# ── Historical & Real-time candles ─────────────────────────────────────────────
def fetch_historical_5m_candles(
    ticker: str, days: int | None = None, date_obj: date | None = None,
    limit: int = DEFAULT_CANDLE_LIMIT, premarket: bool = False,
    as_of: datetime | None = None
) -> list:
    if as_of is None:
        now = datetime.now(tz)
    else:
        now = as_of.astimezone(tz)
    if isinstance(days, date):
        date_obj = days; days = None
    if date_obj:
        start, end = date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")
    else:
        days  = min(days or 3650, 3650)
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    all_c, next_url = [], None
    while True:
        url = f"{next_url}&apiKey={POLYGON_API_KEY}" if next_url else (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
            f"{start}/{end}?adjusted=true&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
        )
        data = safe_fetch_polygon_data(url, ticker)
        res  = data.get("results", [])
        if not res:
            break
        all_c.extend(res)
        next_url = data.get("next_url")
        if not next_url or len(res) < limit:
            break
    return [c for c in all_c if datetime.fromtimestamp(c["t"]/1000, tz) <= now]

def fetch_today_5m_candles(
    ticker: str, limit: int = DEFAULT_CANDLE_LIMIT,
    premarket: bool = True, as_of: datetime | None = None
) -> list:
    return fetch_historical_5m_candles(ticker, days=1, limit=limit, premarket=premarket, as_of=as_of)

def fetch_latest_5m_candle(ticker: str, as_of: datetime | None = None) -> dict:
    if as_of is None:
        now = datetime.now(tz)
    else:
        now = as_of.astimezone(tz)
    today = now.date()
    start = today.strftime("%Y-%m-%d")
    end   = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/"
        f"{start}/{end}?adjusted=true&sort=desc&limit=5&apiKey={POLYGON_API_KEY}"
    )
    data = safe_fetch_polygon_data(url, ticker)
    for c in data.get("results", []):
        dt = datetime.fromtimestamp(c["t"]/1000, tz)
        if dt <= now:
            return {"timestamp": c["t"], "open": c["o"], "high": c["h"], "low": c["l"], "close": c["c"], "volume": c["v"]}
    return {}

def reformat_candles(raw: list) -> list:
    return [
        {"timestamp": c["t"], "open": c["o"], "high": c["h"], "low": c["l"], "close": c["c"], "volume": c["v"]}
        for c in (raw or [])
    ]


# ── CSV Signal Logging ─────────────────────────────────────────────────────────
def append_signal_log(signal: dict):
    date_str = datetime.now(tz).strftime("%Y-%m-%d")
    fname    = f"signal_log_{date_str}.csv"
    fields   = ["timestamp", "ticker", "movement_type", "action", "exit", "note", "ttl"] + ALL_GREEK_KEYS
    row      = {
        "timestamp":     signal["time"],
        "ticker":        signal["ticker"],
        "movement_type": signal["movement_type"],
        "action":        signal["action"],
        "exit":          signal["exit"],
        "note":          signal["note"],
        "ttl":           signal["ttl"]
    }
    row.update({g: signal["greeks"].get(g, 0.0) for g in ALL_GREEK_KEYS})
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    exists = os.path.isfile(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ── Black–Scholes core & Greeks ────────────────────────────────────────────────
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _bs_core(S: float, K: float, T: float, r: float, q: float, sigma: float, typ: str) -> dict:
    sqrtT = math.sqrt(max(T, 1e-6))
    d1    = (math.log(max(S / K, 1e-6)) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2    = d1 - sigma * sqrtT
    pdf   = _norm_pdf(d1)
    cdf1  = _norm_cdf(d1)
    cdf2  = _norm_cdf(d2)
    if typ == "call":
        delta = math.exp(-q * T) * cdf1
    else:
        delta = math.exp(-q * T) * (cdf1 - 1)
    gamma = math.exp(-q * T) * pdf / (S * sigma * sqrtT)
    vega  = S * math.exp(-q * T) * pdf * sqrtT
    term1 = -(S * sigma * math.exp(-q * T) * pdf) / (2 * sqrtT)
    if typ == "call":
        term2 = q * S * math.exp(-q * T) * cdf1 - r * K * math.exp(-r * T) * cdf2
    else:
        term2 = -q * S * math.exp(-q * T) * _norm_cdf(-d1) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    theta = term1 + term2
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def calculate_all_greeks(
    S: float, K: float, T: float, ticker: str, typ: str = "call", sigma_override: float | None = None
) -> dict:
    r = RISK_FREE_RATE
    q = DIVIDEND_YIELDS.get(ticker, 0.0)
    sigma = sigma_override if sigma_override is not None else DEFAULT_VOLATILITIES.get(ticker, 0.2)

    core = _bs_core(S, K, T, r, q, sigma, typ)
    d, g, th, v = core["delta"], core["gamma"], core["theta"], core["vega"]
    sqrtT = math.sqrt(max(T, 1e-6))
    d2    = (math.log(max(S / K, 1e-6)) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT) - sigma * sqrtT
    c2    = _norm_cdf(d2)
    rho   = (K * T * math.exp(-r * T) * c2) if typ == "call" else (-K * T * math.exp(-r * T) * (1 - c2))

    # Higher-order Greeks
    ev  = max(sigma * 0.01, 1e-3)
    eT  = max(T * 0.01, 1e-4)
    eS  = max(S * 0.01, 0.1)
    vanna = (_bs_core(S, K, T, r, q, sigma + ev, typ)["delta"] - _bs_core(S, K, T, r, q, sigma - ev, typ)["delta"]) / (2 * ev)
    vomma = (_bs_core(S, K, T, r, q, sigma + ev, typ)["vega"]  - _bs_core(S, K, T, r, q, sigma - ev, typ)["vega"])  / (2 * ev)
    charm = (_bs_core(S, K, T + eT, r, q, sigma, typ)["delta"] - _bs_core(S, K, T - eT, r, q, sigma, typ)["delta"]) / (2 * eT)
    veta  = (_bs_core(S, K, T + eT, r, q, sigma, typ)["vega"]   - _bs_core(S, K, T - eT, r, q, sigma, typ)["vega"])   / (2 * eT)
    speed = (_bs_core(S + eS, K, T, r, q, sigma, typ)["gamma"] - _bs_core(S - eS, K, T, r, q, sigma, typ)["gamma"]) / (2 * eS)
    zomma = (_bs_core(S, K, T, r, q, sigma + ev, typ)["gamma"] - _bs_core(S, K, T, r, q, sigma - ev, typ)["gamma"]) / (2 * ev)
    color = (_bs_core(S + eS, K, T, r, q, sigma, typ)["theta"] - _bs_core(S - eS, K, T, r, q, sigma, typ)["theta"]) / (2 * eS)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta_day": th / 365.0,
        "theta_5m": th / 365.0 * (5/60/24),
        "vega": v,
        "rho": rho,
        "vanna": vanna,
        "vomma": vomma,
        "charm": charm / 365.0,
        "veta": veta / 365.0,
        "speed": speed,
        "zomma": zomma,
        "color": color,
        "implied_volatility": sigma
    }
