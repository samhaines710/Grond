"""Data ingestion utilities for historical and real-time market data.

This module defines two classes:

* ``HistoricalDataLoader`` — fetches 5-minute bars from Polygon's REST API,
  handling pagination and time filtering.
* ``RealTimeDataStreamer`` — connects to Polygon's WebSocket API to
  aggregate per-minute trades into 5-minute OHLCV bars in memory.

Both classes adhere to simple rate limiting and provide structured
access to streaming data.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

import pytz
from websocket import WebSocketApp

from config import POLYGON_API_KEY, TICKERS, tz
from utils.http_client import safe_fetch_polygon_data, rate_limited
from utils.logging_utils import write_status

# In-memory stores for real-time candles: symbol → deque of bars
REALTIME_CANDLES: Dict[str, deque] = {symbol: deque(maxlen=200) for symbol in TICKERS}
REALTIME_LOCK = threading.Lock()


class HistoricalDataLoader:
    """Load historical 5-minute bar data from Polygon's REST API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or POLYGON_API_KEY
        self.base_url: str = "https://api.polygon.io"

    @rate_limited
    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper to perform a REST GET with rate limiting."""
        query = params.copy()
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{path}?{urlencode(query)}"
        data: Dict[str, Any] = safe_fetch_polygon_data(url, ticker=path)
        return data or {}

    def fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Return a list of 5-minute bars between ``start`` and ``end`` timestamps."""
        all_bars: List[Dict[str, Any]] = []
        next_url: Optional[str] = None
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        while True:
            if next_url:
                data = self._get(next_url, {})
            else:
                day0 = start.date().strftime("%Y-%m-%d")
                day1 = end.date().strftime("%Y-%m-%d")
                path = f"/v2/aggs/ticker/{ticker}/range/5/minute/{day0}/{day1}"
                params: Dict[str, Any] = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": limit,
                }
                data = self._get(path, params)

            bars = data.get("results", [])
            if not bars:
                break

            for bar in bars:
                if start_ts <= bar["t"] <= end_ts:
                    all_bars.append(bar)

            next_url = data.get("next_url")
            if not next_url or len(bars) < limit:
                break

        write_status(f"Fetched {len(all_bars)} historical bars for {ticker}")
        return all_bars


class RealTimeDataStreamer:
    """
    Subscribe to Polygon's real-time WebSocket feed and aggregate minute trades
    into 5-minute OHLCV bars stored in ``REALTIME_CANDLES``.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key: str = api_key or POLYGON_API_KEY
        self.ws_url: str = "wss://socket.polygon.io/stocks"
        self._ws_lock = threading.Lock()
        self._ws: Optional[WebSocketApp] = None

    def on_open(self, ws: WebSocketApp) -> None:
        """Authenticate and subscribe on WebSocket open."""
        write_status("RT WS opened; authenticating…")
        ws.send(json.dumps({"action": "auth", "params": self.api_key}))
        # subscribe to aggregate‐minute streams (“AM.”)
        for ticker in TICKERS:
            ws.send(json.dumps({"action": "subscribe", "params": f"AM.{ticker}"}))

    def on_message(self, ws: WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages by aggregating into 5-minute bars."""
        try:
            payload = json.loads(message)
            items = payload if isinstance(payload, list) else [payload]
            for itm in items:
                # only handle aggregate‐minute events
                if itm.get("ev") != "AM":
                    continue

                ts_ms = itm["t"]
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.UTC).astimezone(tz)
                minute = (dt.minute // 5) * 5
                bucket = dt.replace(minute=minute, second=0, microsecond=0)
                bar_ts = int(bucket.timestamp() * 1000)

                record = {
                    "timestamp": bar_ts,
                    "open":      itm.get("o"),
                    "high":      itm.get("h"),
                    "low":       itm.get("l"),
                    "close":     itm.get("c"),
                    "volume":    itm.get("v", 0),
                }

                with REALTIME_LOCK:
                    dq = REALTIME_CANDLES.setdefault(itm["sym"], deque(maxlen=200))
                    if dq and dq[-1]["timestamp"] == bar_ts:
                        prev = dq[-1]
                        prev["high"]   = max(prev["high"], record["high"])
                        prev["low"]    = min(prev["low"], record["low"])
                        prev["close"]  = record["close"]
                        prev["volume"] += record["volume"]
                    else:
                        dq.append(record)

        except Exception as exc:
            write_status(f"RT on_message error: {exc}")

    def on_error(self, ws: WebSocketApp, err: Exception) -> None:
        """Log WebSocket errors."""
        write_status(f"RT WS error: {err}")

    def on_close(self, ws: WebSocketApp, code: int, msg: str) -> None:
        """Handle WebSocket closure and schedule a reconnect."""
        write_status(f"RT WS closed: {code}/{msg}; reconnecting in 5s")
        time.sleep(5)
        self.start()

    def start(self) -> None:
        """Start the WebSocket streaming in a background daemon thread."""
        def _run() -> None:
            ws = WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            with self._ws_lock:
                self._ws = ws  # keep a reference alive
            ws.run_forever()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        write_status("RealTimeDataStreamer thread started.")
