import os
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any, Optional

import pytz
import requests
from websocket import WebSocketApp

from config import POLYGON_API_KEY, TICKERS, OPTIONS_TICKERS, tz
from utils import rate_limited, write_status, fetch_premarket_early_data, reformat_candles

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

# Thread-safe in-memory store for 5m bars
REALTIME_CANDLES = {t: deque(maxlen=200) for t in TICKERS}
REALTIME_LOCK    = threading.Lock()


class HistoricalDataLoader:
    """
    Backfill 5-minute bars via Polygon REST, handling pagination.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"

    @rate_limited
    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        params["apiKey"] = self.api_key
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def fetch_bars(
        self,
        ticker: str,
        start:  datetime,
        end:    datetime,
        limit:  int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Fetch all 5m bars for `ticker` between `start` and `end`.
        """
        all_bars = []
        next_url = None
        start_ts = int(start.timestamp() * 1000)
        end_ts   = int(end.timestamp()   * 1000)

        while True:
            if next_url:
                data = self._get(next_url, {})
            else:
                path   = f"/v2/aggs/ticker/{ticker}/range/5/minute/{start.date()}/{end.date()}"
                params = {"adjusted": "true", "sort": "asc", "limit": limit}
                data   = self._get(path, params)

            bars = data.get("results", [])
            if not bars:
                break

            for b in bars:
                if start_ts <= b["t"] <= end_ts:
                    all_bars.append(b)

            next_url = data.get("next_url")
            if not next_url or len(bars) < limit:
                break

        write_status(f"Fetched {len(all_bars)} historical bars for {ticker}")
        return all_bars

    def fetch_option_snapshots(
        self,
        ticker: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch the latest page of option snapshots for `ticker`.
        """
        path   = f"/v3/snapshot/options/{ticker}"
        params = {"limit": limit}
        data   = self._get(path, params)
        results = data.get("results", [])
        write_status(f"Fetched {len(results)} option snapshots for {ticker}")
        return results


class RealTimeDataStreamer:
    """
    Subscribes to Polygon WebSocket for minute trades,
    aggregates into 5-minute OHLCV bars in REALTIME_CANDLES.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or POLYGON_API_KEY
        self.ws_url   = "wss://socket.polygon.io/stocks"
        self._ws_lock = threading.Lock()

    def on_open(self, ws: WebSocketApp):
        write_status("RT WS opened, sending auth…")
        ws.send(f'{{"action":"auth","params":"{self.api_key}"}}')
        for t in TICKERS:
            ws.send(f'{{"action":"subscribe","params":"AM.{t}"}}')

    def on_message(self, ws: WebSocketApp, message: str):
        try:
            payload = __import__("json").loads(message)
            items   = payload if isinstance(payload, list) else [payload]
            for itm in items:
                if itm.get("ev") != "AM":
                    continue
                ts_ms = itm["t"]
                dt    = datetime.fromtimestamp(ts_ms/1000, tz=pytz.UTC).astimezone(tz)
                minute = (dt.minute // 5) * 5
                bucket = dt.replace(minute=minute, second=0, microsecond=0)
                bar_ts = int(bucket.timestamp() * 1000)
                rec = {
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
                        prev["high"]   = max(prev["high"], rec["high"])
                        prev["low"]    = min(prev["low"], rec["low"])
                        prev["close"]  = rec["close"]
                        prev["volume"] += rec["volume"]
                    else:
                        dq.append(rec)
        except Exception as e:
            write_status(f"RT on_message error: {e}")

    def on_error(self, ws: WebSocketApp, err):
        write_status(f"RT WS error: {err}")

    def on_close(self, ws: WebSocketApp, code, msg):
        write_status(f"RT WS closed: {code}/{msg}, reconnecting in 5s")
        time.sleep(5)
        self.start()

    def start(self):
        """
        Launch the WebSocket client on a daemon thread.
        """
        def run():
            ws = WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            with self._ws_lock:
                self._ws = ws
            ws.run_forever()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        write_status("RealTimeDataStreamer thread started.")
