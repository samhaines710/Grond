# src/utils/http_client.py

import time
import math
import threading
from typing import Any, Callable, Dict

import requests

from config import POLYGON_API_KEY, DEFAULT_VOLATILITY_FALLBACK
from utils.logging_utils import write_status, REST_CALLS, REST_429
from utils.greeks_helpers import calculate_all_greeks
from data_ingestion import REALTIME_CANDLES, REALTIME_LOCK

# ─── Circuit breaker ────────────────────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooloff_secs: int = 60):
        self.fail_count   = 0
        self.threshold    = threshold
        self.open_until   = None
        self.cooloff_secs = cooloff_secs

    def record_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.open_until = time.time() + self.cooloff_secs
            write_status(f"Circuit open until {self.open_until}")

    def record_success(self):
        self.fail_count = 0
        self.open_until = None

    def is_open(self) -> bool:
        if self.open_until and time.time() < self.open_until:
            return True
        self.open_until = None
        return False

breaker = CircuitBreaker()

# ─── Rate limiter ────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rate_per_sec: float, burst_sec: int):
        self.rate      = rate_per_sec
        self.cap       = burst_sec
        self.tokens    = burst_sec
        self.timestamp = time.time()
        self.lock      = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now   = time.time()
            delta = now - self.timestamp
            self.tokens = min(self.cap, self.tokens + delta * self.rate)
            self.timestamp = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait(self):
        while not self.acquire():
            time.sleep(0.01)

limiter = RateLimiter(rate_per_sec=5.0, burst_sec=10)

def rate_limited(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        limiter.wait()
        return func(*args, **kwargs)
    return wrapper

# ─── Safe fetch ─────────────────────────────────────────────────────────────────
@rate_limited
def safe_fetch_polygon_data(
    url: str,
    ticker: str = "",
    retries: int = 3
) -> Dict[str, Any]:
    """
    Fetch JSON from Polygon, handling rate-limit, 429s, and circuit-breaker.
    """
    if breaker.is_open():
        write_status(f"Skipping REST call to {url} (circuit open)")
        return {}

    REST_CALLS.inc()
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=7)
            if resp.status_code == 429:
                REST_429.inc()
                breaker.record_failure()
                write_status(f"429 from Polygon for {ticker}")
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            breaker.record_success()
            return resp.json()
        except Exception as e:
            write_status(f"Request error for {ticker}: {e}")
            time.sleep(2 ** attempt)

    write_status(f"Failed to fetch data after {retries} attempts: {ticker}")
    return {}

# ─── Option Greeks Loader ───────────────────────────────────────────────────────
def fetch_option_greeks(
    ticker: str,
    days_to_expiry: int = 30,
    typ: str = "call"
) -> Dict[str, Any]:
    """
    Try Polygon snapshot for `typ` near-ATM option; fallback to realized-volatility.
    """
    # 1) Polygon snapshot
    url     = f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={POLYGON_API_KEY}"
    results = safe_fetch_polygon_data(url, ticker).get("results", [])

    if results:
        same_type = [
            o for o in results
            if o.get("details", {}).get("contract_type", "").lower() == typ
        ]
        candidates = same_type or results
        opt = min(candidates, key=lambda o: abs(
            o["details"]["strike_price"] - o["underlying_asset"]["price"]
        ))
        pg = opt.get("greeks", {})
        if pg:
            write_status(f"Used Polygon {typ} Greeks for {ticker}")
            return {
                "delta":               pg.get("delta", 0.0),
                "gamma":               pg.get("gamma", 0.0),
                "theta":               pg.get("theta", 0.0),
                "vega":                pg.get("vega", 0.0),
                "rho":                 pg.get("rho", 0.0),
                "vanna":               pg.get("vanna", 0.0),
                "vomma":               pg.get("vomma", 0.0),
                "charm":               pg.get("charm", 0.0),
                "veta":                pg.get("veta", 0.0),
                "speed":               pg.get("speed", 0.0),
                "zomma":               pg.get("zomma", 0.0),
                "color":               pg.get("color", 0.0),
                "implied_volatility":  opt.get("implied_volatility", 0.0),
            }

    # 2) Fallback: realized-volatility greeks
    with REALTIME_LOCK:
        bars = list(REALTIME_CANDLES.get(ticker, []))

    sigma = DEFAULT_VOLATILITY_FALLBACK
    if len(bars) >= 2:
        rets = [
            math.log(bars[i]["close"] / bars[i - 1]["close"])
            for i in range(1, len(bars))
            if bars[i - 1]["close"] > 0
        ]
        if rets:
            m = sum(rets) / len(rets)
            v = sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)
            sigma = math.sqrt(v * 78 * 252)

    greeks = calculate_all_greeks(
        S=bars[-1]["close"] if bars else 100.0,
        K=bars[-1]["close"] if bars else 100.0,
        T=days_to_expiry / 365.0,
        ticker=ticker,
        typ=typ,
        sigma_override=sigma
    )
    greeks["implied_volatility"] = sigma
    write_status(f"Calculated fallback {typ} Greeks for {ticker} with σ={sigma:.3f}")
    return greeks
