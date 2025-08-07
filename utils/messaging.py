# messaging.py

"""
Messaging and WebSocket helpers for notifications and live data.

This module handles Telegram notifications with cooldown logic and
establishes a WebSocket connection to Polygon for live data subscriptions.
Errors and status updates are logged via write_status().
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Optional

import websocket
from telegram import Bot, error as tg_error

from config import TICKERS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ENABLE_TELEGRAM, POLYGON_API_KEY, TELEGRAM_COOLDOWN_SECONDS
from utils.logging_utils import write_status

# ─── Telegram setup ────────────────────────────────────────────────────────────
_bot: Optional[Bot] = None
if ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    _bot = Bot(token=TELEGRAM_BOT_TOKEN)

_last_telegram_ts = 0.0

def send_telegram(message: str) -> None:
    """
    Send a Markdown-formatted message via Telegram, respecting cooldown.
    """
    global _last_telegram_ts
    now = time.time()
    if now - _last_telegram_ts < TELEGRAM_COOLDOWN_SECONDS:
        write_status("Skipping Telegram send: cooldown active")
        return
    _last_telegram_ts = now

    if not _bot:
        write_status("Telegram not configured; message dropped")
        return

    try:
        # Telegram Bot API is async under the hood, so run in event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(
            _bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="Markdown"
            )
        )
    except tg_error.TimedOut:
        write_status("Telegram timed out; message skipped")
    except tg_error.TelegramError as exc:
        msg = str(exc).lower()
        if "flood control" in msg:
            write_status("Telegram flood control triggered; cooling down")
        else:
            write_status(f"Telegram error: {exc}")

# ─── WebSocket setup ──────────────────────────────────────────────────────────
_websocket_instance: Optional[websocket.WebSocketApp] = None
_websocket_lock = threading.Lock()

def on_message(ws: websocket.WebSocketApp, message: str) -> None:
    """Handler for incoming WebSocket messages."""
    write_status(f"WS message received: {message[:200]}")

def on_open(ws: websocket.WebSocketApp) -> None:
    """Authenticate and subscribe when the WebSocket opens."""
    write_status("WebSocket opened; authenticating…")
    ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))
    for ticker in TICKERS:
        ws.send(json.dumps({"action": "subscribe", "params": f"AM.{ticker}"}))

def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
    """Log WebSocket errors."""
    write_status(f"WebSocket error: {err}")

def on_close(ws: websocket.WebSocketApp, code: int, msg: str) -> None:
    """Log WebSocket closure and allow reconnect logic to handle restarts."""
    write_status(f"WebSocket closed: {code} - {msg}")

def start_websocket_stream(url: str = "wss://socket.polygon.io/stocks") -> None:
    """
    Start a background thread for the Polygon WebSocket stream.
    On crash, it will reconnect after a short delay.
    """
    def _run() -> None:
        while True:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                with _websocket_lock:
                    global _websocket_instance
                    _websocket_instance = ws
                ws.run_forever()
            except Exception as exc:
                write_status(f"WebSocket crashed: {exc}; retrying in 5s")
                time.sleep(5)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    write_status("WebSocket thread started.")
