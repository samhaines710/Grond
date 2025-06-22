# utils/http_client.py

import time
import os
import requests
from functools import wraps
from utils.logging_utils import write_status, REST_CALLS, REST_429

# Load the Polygon API key from environment
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

class CircuitBreaker:
    def __init__(self, threshold=5, cooloff_secs=60):
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
        self.fail_count  = 0
        self.open_until = None

    def is_open(self) -> bool:
        if self.open_until and time.time() < self.open_until:
            return True
        self.open_until = None
        return False

_circuit_breaker = CircuitBreaker()

class RateLimiter:
    def __init__(self, rate_per_sec, burst_sec, rate_per_min, burst_min):
        self.rate_sec  = rate_per_sec
        self.cap_sec   = burst_sec
        self.tokens_s  = burst_sec
        self.ts_s      = time.time()
        self.rate_min  = rate_per_min / 60.0
        self.cap_min   = burst_min
        self.tokens_m  = burst_min
        self.ts_m      = time.time()

    def acquire(self) -> bool:
        now = time.time()
        ds = now - self.ts_s
        if ds > 0:
            self.tokens_s = min(self.cap_sec, self.tokens_s + ds * self.rate_sec)
            self.ts_s = now
        dm = now - self.ts_m
        if dm > 0:
            self.tokens_m = min(self.cap_min, self.tokens_m + dm * self.rate_min)
            self.ts_m = now
        if self.tokens_s >= 1 and self.tokens_m >= 1:
            self.tokens_s -= 1
            self.tokens_m -= 1
            return True
        return False

    def wait(self):
        while not self.acquire():
            time.sleep(0.01)

_ratelimiter = RateLimiter(
    rate_per_sec=5.0, burst_sec=10,
    rate_per_min=200.0, burst_min=200
)

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _ratelimiter.wait()
        return func(*args, **kwargs)
    return wrapper

@rate_limited
def _get_json(url: str, ticker: str = "") -> dict:
    if _circuit_breaker.is_open():
        write_status(f"Skipping REST call to {url} (circuit open)")
        return {}
    REST_CALLS.inc()
    try:
        resp = requests.get(f"{url}&apiKey={POLYGON_API_KEY}", timeout=7)
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
    """
    Retry-backed fetch of Polygon data.
    """
    for attempt in range(retries):
        data = _get_json(url, ticker)
        if data:
            return data
        time.sleep(2 ** attempt)
    write_status(f"Failed to fetch data for {ticker} after {retries} attempts")
    return {}
