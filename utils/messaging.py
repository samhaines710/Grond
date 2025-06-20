# utils/messaging.py

import os
import json
import asyncio
import threading
import websocket
from telegram import Bot, error as tg_error
from logging_utils import write_status
from config import (
    POLYGON_API_KEY,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    ENABLE_TELEGRAM,
    TICKERS
)

# ─── Telegram Setup ─────────────────────────────────────────────────────────────
bot = Bot(token=TELEGRAM_BOT_TOKEN) if (
    ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
) else None

_last_telegram_ts = 0
TELEGRAM_COOLDOWN_SECONDS = 60

def send_telegram(message: str):
    """
    Send to Telegram with flood and cooldown suppression.
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
        loop.run_until_complete(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
    except tg_error.TelegramError as e:
        write_status(f"Telegram error: {e}")

# ─── WebSocket Handlers ─────────────────────────────────────────────────────────
_websocket_instance = None
_websocket_lock     = threading.Lock()

def on_message(ws, message):
    write_status(f"WS msg received: {message[:100]}…")

def on_open(ws):
    write_status("WebSocket opened; subscribing to tickers")
    ws.send(json.dumps({"action":"auth","params":POLYGON_API_KEY}))
    for t in TICKERS:
        ws.send(json.dumps({"action":"subscribe","params":f"AM.{t}"}))

def on_error(ws, err):
    write_status(f"WebSocket error: {err}")

def on_close(ws, code, msg):
    write_status(f"WebSocket closed: {code}/{msg}")

def start_websocket_stream(url: str = "wss://delayed.polygon.io/stocks"):
    """
    Launch WebSocket in a background thread, auto-reconnecting.
    """
    def _run():
        nonlocal url
        while True:
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_open=on_open,
                    on_error=on_error,
                    on_close=on_close
                )
                with _websocket_lock:
                    global _websocket_instance
                    _websocket_instance = ws
                ws.run_forever()
            except Exception as e:
                write_status(f"WS crashed: {e}; reconnecting in 5s")
                time.sleep(5)
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    write_status("WebSocket thread started.")
