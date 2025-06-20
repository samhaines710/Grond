# data_ingestion.py

import time
import threading
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any, Optional

import pytz
from websocket import WebSocketApp

from config import POLYGON_API_KEY, TICKERS, tz
from utils.http_client     import safe_fetch_polygon_data, rate_limited
from utils.logging_utils   import write_status
from utils.calendar_utils  import is_market_open_today

# ─── In-Memory Stores ────────────────────────────────────────────────────────────
REALTIME_CANDLES = {t: deque(maxlen=200) for t in TICKERS}
REALTIME_LOCK    = threading.Lock()


# ─── Historical REST Loader ─────────────────────────────────────────────────────
class HistoricalDataLoader:
    """
    Backfill 5-minute bars via Polygon REST, handling pagination
    and time-window filtering.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key or POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"

    @rate_limited
    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        params["apiKey"] = self.api_key
        resp = safe_fetch_polygon_data(url, ticker="")
        return resp or {}

    def fetch_bars(
        self,
        ticker: str,
        start:  datetime,
        end:    datetime,
        limit:  int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Fetch all 5-minute bars for `ticker` between `start` and `end`.
        """
        all_bars = []
        next_url = None
        start_ts = int(start.timestamp() * 1000)
        end_ts   = int(end.timestamp()   * 1000)

        while True:
            if next_url:
                data = self._get(next_url, {})
            else:
                day0 = start.date().strftime("%Y-%m-%d")
                day1 = end.date().strftime("%Y-%m-%d")
                path   = f"/v2/aggs/ticker/{ticker}/range/5/minute/{day0}/{day1}"
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


# ─── Real-Time WebSocket Streamer ────────────────────────────────────────────────
class RealTimeDataStreamer:
    """
    Subscribes to Polygon WebSocket for minute trades,
    aggregates into 5-minute OHLCV bars in REALTIME_CANDLES.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or POLYGON_API_KEY
        self.ws_url  = "wss://socket.polygon.io/stocks"
        self._ws_lock = threading.Lock()

    def on_open(self, ws: WebSocketApp):
        write_status("RT WS opened; authenticating…")
        ws.send(json.dumps({"action":"auth","params":self.api_key}))
        for t in TICKERS:
            ws.send(json.dumps({"action":"subscribe","params":f"AM.{t}"}))

    def on_message(self, ws: WebSocketApp, message: str):
        try:
            payload = json.loads(message)
            for itm in (payload if isinstance(payload, list) else [payload]):
                if itm.get("ev") != "AM":
                    continue

                # bucket into 5-minute bars in local tz
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
        write_status(f"RT WS closed: {code}/{msg}; reconnecting in 5s")
        time.sleep(5)
        self.start()

    def start(self):
        """
        Launch the WebSocket client on a daemon thread and begin streaming.
        """
        def _run():
            ws = WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            with self._ws_lock:
                self._ws = ws
            ws.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        write_status("RealTimeDataStreamer thread started.")
