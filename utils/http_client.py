"""HTTP client helpers for REST access to the Polygon API.

This module encapsulates circuit breaker and rate limiter logic to
protect the application from excessive requests. It also provides
functions to fetch option Greeks via the Polygon snapshot API,
falling back to local calculations when necessary.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

from config import POLYGON_API_KEY, DEFAULT_VOLATILITY_FALLBACK
from utils.logging_utils import write_status, REST_CALLS, REST_429
from utils.greeks_helpers import calculate_all_greeks


class CircuitBreaker:
    """Simple circuit breaker that opens after consecutive failures."""

    def __init__(self, threshold: int = 5, cooloff_secs: int = 60) -> None:
        self.fail_count = 0
        self.threshold = threshold
        self.open_until: Optional[float] = None
        self.cooloff_secs = cooloff_secs

    def record_failure(self) -> None:
        """Increment failure count and open breaker if threshold reached."""
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.open_until = time.time() + self.cooloff_secs
            write_status(f"Circuit open until {self.open_until}")

    def record_success(self) -> None:
        """Reset the circuit breaker state on success."""
        self.fail_count = 0
        self.open_until = None

    def is_open(self) -> bool:
        """Return ``True`` if the circuit breaker is currently open."""
        if self.open_until and time.time() < self.open_until:
            return True
        self.open_until = None
        return False


breaker = CircuitBreaker()


class RateLimiter:
    """Token-bucket rate limiter used to control API call frequency."""

    def __init__(self, rate_per_sec: float, burst_sec: int) -> None:
        self.rate = rate_per_sec
        self.cap = burst_sec
        self.tokens = float(burst_sec)
        self.timestamp = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """Attempt to consume a token; return ``True`` if successful."""
        with self.lock:
            now = time.time()
            delta = now - self.timestamp
            # Refill tokens based on elapsed time
            self.tokens = min(self.cap, self.tokens + delta * self.rate)
            self.timestamp = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait(self) -> None:
        """Block until a token is available."""
        while not self.acquire():
            time.sleep(0.01)


limiter = RateLimiter(rate_per_sec=5.0, burst_sec=10)


def rate_limited(func: Callable) -> Callable:
    """Decorator to apply the rate limiter to functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        limiter.wait()
        return func(*args, **kwargs)

    return wrapper


@rate_limited
def safe_fetch_polygon_data(
    url: str,
    ticker: str = "",
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Fetch JSON from the Polygon API, handling rate limits, HTTP 429s, and circuit breaker.

    Parameters
    ----------
    url: str
        The full URL to request.
    ticker: str, optional
        A label used for logging purposes when retrying requests.
    retries: int, optional
        Number of retry attempts before giving up.

    Returns
    -------
    Dict[str, Any]
        The JSON response from the API, or an empty dict on failure.
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
            return resp.json()  # type: ignore[no-any-return]
        except Exception as exc:
            write_status(f"Request error for {ticker}: {exc}")
            time.sleep(2 ** attempt)

    write_status(f"Failed to fetch data after {retries} attempts: {ticker}")
    return {}


def fetch_option_greeks(
    ticker: str,
    days_to_expiry: int = 30,
    typ: str = "call",
) -> Dict[str, Any]:
    """
    Attempt to obtain option Greeks from the Polygon snapshot API.

    If snapshot data is unavailable, fall back to computing Greeks
    using realized volatility derived from recent price data.

    Parameters
    ----------
    ticker: str
        Underlying symbol to fetch Greeks for.
    days_to_expiry: int, optional
        Days until the option expires; used in fallback calculations.
    typ: str, optional
        Option type: ``"call"`` or ``"put"``.

    Returns
    -------
    Dict[str, Any]
        A dictionary of option Greek values.
    """
    # Attempt 1: Polygon snapshot API
    url = (
        f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        f"?apiKey={POLYGON_API_KEY}"
    )
    results = safe_fetch_polygon_data(url, ticker).get("results", [])

    if results:
        same_type = [
            o
            for o in results
            if o.get("details", {}).get("contract_type", "").lower() == typ
        ]
        candidates = same_type or results

        # Select near‑ATM (closest strike) option
        opt = min(
            candidates,
            key=lambda o: abs(
                o["details"]["strike_price"]
                - o["underlying_asset"]["price"]
            ),
        )
        pg = opt.get("greeks", {})
        if pg:
            write_status(f"Used Polygon {typ} Greeks for {ticker}")
            return {
                "delta": pg.get("delta", 0.0),
                "gamma": pg.get("gamma", 0.0),
                "theta": pg.get("theta", 0.0),
                "vega": pg.get("vega", 0.0),
                "rho": pg.get("rho", 0.0),
                "vanna": pg.get("vanna", 0.0),
                "vomma": pg.get("vomma", 0.0),
                "charm": pg.get("charm", 0.0),
                "veta": pg.get("veta", 0.0),
                "speed": pg.get("speed", 0.0),
                "zomma": pg.get("zomma", 0.0),
                "color": pg.get("color", 0.0),
                "implied_volatility": opt.get("implied_volatility", 0.0),
            }

    # Fallback: compute Greeks using realized volatility
    from utils.market_data import REALTIME_CANDLES  # local import to avoid cycles

    with threading.Lock():
        bars = list(REALTIME_CANDLES.get(ticker, []))

    sigma = DEFAULT_VOLATILITY_FALLBACK
    if len(bars) >= 2:
        rets = [
            math.log(bars[i]["close"] / bars[i - 1]["close"])
            for i in range(1, len(bars))
            if bars[i - 1]["close"] > 0
        ]
        if rets:
            mean_ret = sum(rets) / len(rets)
            var_ret = sum((r - mean_ret) ** 2 for r in rets) / max(len(rets) - 1, 1)
            sigma = math.sqrt(var_ret * 78 * 252)

    greeks = calculate_all_greeks(
        S=bars[-1]["close"] if bars else 100.0,
        K=bars[-1]["close"] if bars else 100.0,
        T=days_to_expiry / 365.0,
        ticker=ticker,
        typ=typ,
        sigma_override=sigma,
    )
    greeks["implied_volatility"] = sigma
    write_status(f"Calculated fallback {typ} Greeks for {ticker} with σ={sigma:.3f}")
    return greeks
