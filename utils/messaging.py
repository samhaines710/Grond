"""Messaging and WebSocket helpers for notifications and live data.

This module handles Telegram notifications and establishes a WebSocket
connection to Polygon for live data subscriptions. Cooldown logic is
implemented to prevent spamming, and errors are logged via
``write_status``.
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

from config import TICKERS
from utils.logging_utils import write_status


# Load environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "False").lower() in ("1", "true", "yes")


# Instantiate Telegram bot if enabled and credentials are provided
bot: Optional[Bot] = None
if ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

TELEGRAM_COOLDOWN_SECONDS = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "60"))
_last_telegram_ts = 0.0


def send_telegram(message: str) -> None:
    """
    Send a message via Telegram respecting a cooldown period.

    If the cooldown has not expired or the bot is not configured, the message
    will not be sent. Errors are caught and logged.
    """
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
        loop.run_until_complete(
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        )
    except tg_error.TimedOut:
        write_status("Telegram timed out—skipping")
    except tg_error.TelegramError as exc:
        if "flood control" not in str(exc).lower():
            write_status(f"Telegram error: {exc}")


# WebSocket instance and lock
_websocket_instance: Optional[websocket.WebSocketApp] = None
_websocket_lock = threading.Lock()


def on_message(ws: websocket.WebSocketApp, message: str) -> None:
    """
    Handler for incoming WebSocket messages.
    """
    write_status(f"WS message received: {message[:200]}")


def on_open(ws: websocket.WebSocketApp) -> None:
    """
    Authenticate and subscribe to symbols when the WebSocket opens.
    """
    write_status("WebSocket opened, authenticating...")
    ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))
    for ticker in TICKERS:
        ws.send(json.dumps({"action": "subscribe", "params": f"AM.{ticker}"}))


def on_error(ws: websocket.WebSocketApp, err: Exception) -> None:
    """Log WebSocket errors."""
    write_status(f"WebSocket error: {err}")


def on_close(ws: websocket.WebSocketApp, code: int, msg: str) -> None:
    """Log WebSocket closure events."""
    write_status(f"WebSocket closed: {code} - {msg}")


def start_websocket_stream(url: str = "wss://socket.polygon.io/stocks") -> None:
    """
    Start a background thread for the Polygon WebSocket stream.

    The default URL uses the real‑time feed; do not specify
    ``delayed.polygon.io`` here. On errors, the connection will attempt
    to reconnect after a short delay.
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
                write_status(f"WebSocket crashed: {exc}; reconnecting in 5s")
                time.sleep(5)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    write_status("WebSocket thread started.")
