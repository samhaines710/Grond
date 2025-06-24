# utils/analysis.py

import math
from datetime import datetime, time as dtime
from collections import deque

from config import (
    LOOKBACK_BREAKOUT,
    LOOKBACK_RISK_REWARD,
    tz,
)
from utils.http_client import safe_fetch_polygon_data

def calculate_breakout_prob(candles: list, lookback: int = LOOKBACK_BREAKOUT) -> float:
    """
    Estimate breakout probability (%).
    Uses log-moves and volumes over the last `lookback` bars.
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves, vols = [], []
    for bar in recent:
        o, c = bar["open"], bar["close"]
        if o and o != 0:
            moves.append((c - o) / o)
            vols.append(bar["volume"])
    if not moves:
        return 0.0
    avgm = sum(abs(m) for m in moves) / len(moves)
    avgv = sum(vols) / len(vols)
    last = recent[-1]
    cm = abs(last["close"] - last["open"]) / last["open"] if last["open"] else 0
    cv = last["volume"]
    trend = sum(1 if m > 0 else -1 for m in moves) / len(moves)
    score = 50 * math.log1p(cm / max(avgm, 1e-6)) * math.log1p(cv / max(avgv, 1)) * (1 + trend)
    return round(max(0, min(100, score)), 2)

def calculate_recent_move_pct(candles: list) -> float:
    """
    % move from the first bar’s open to the last bar’s close.
    """
    if len(candles) < 2:
        return 0.0
    first = candles[0]
    last  = candles[-1]
    if first["open"] == 0:
        return 0.0
    return round((last["close"] - first["open"]) / first["open"] * 100, 4)

def calculate_signal_persistence(candles: list, lookback: int = LOOKBACK_BREAKOUT) -> float:
    """
    How “persistent” the last bar’s move was relative to recent average.
    Returns percentage [0–100].
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves = [abs((b["close"] - b["open"]) / b["open"]) for b in recent if b["open"]]
    if not moves:
        return 0.0
    avgm = sum(moves) / len(moves)
    last = moves[-1]
    return round(100 * (1 - min(last / max(avgm, 1e-6), 1)), 2)

def calculate_reversal_and_scope(
    candles: list,
    oi_bias: str,
    threshold: float = 0.005
) -> tuple[bool,bool,float]:
    """
    Based on change in the last bar vs prior:
      - reversal if OI‐bias + move exceed threshold
      - scope if abs(move) > threshold
    Returns (is_reversal, is_scope, move_pct)
    """
    if len(candles) < 2:
        return False, False, 0.0
    prev, last = candles[-2], candles[-1]
    mv = (last["close"] - prev["close"]) / prev["close"] if prev["close"] else 0.0
    rev = (oi_bias == "CALL_DOMINANT" and mv < -threshold) or (
          oi_bias == "PUT_DOMINANT" and mv > threshold)
    sc = abs(mv) > threshold
    return rev, sc, round(mv * 100, 4)

def calculate_risk_reward(candles: list, lookback: int = LOOKBACK_RISK_REWARD) -> float:
    """
    ATR‐based risk/reward ratio: (2 * ATR) / (0.5 * ATR) = 4
    """
    if len(candles) < lookback:
        return 1.0
    samples = candles[-lookback:]
    atr = sum(b["high"] - b["low"] for b in samples) / len(samples)
    return round((atr * 2) / max(atr * 0.5, 1e-6), 2)

def calculate_market_trend(ticker: str) -> float:
    """
    Placeholder market‐trend signal.
    Originally relied on VADER sentiment; now returns 0.
    """
    # TODO: plug in real market‐trend model (news, sentiment, etc.)
    return 0.0

def calculate_time_of_day(as_of: datetime | None = None) -> str:
    now = (as_of or datetime.now(tz)).astimezone(tz).time()
    if dtime(4, 0) <= now < dtime(9, 30):
        return "PRE_MARKET"
    if dtime(9, 30) <= now < dtime(11, 0):
        return "MORNING"
    if dtime(11, 0) <= now < dtime(14, 0):
        return "MIDDAY"
    if dtime(14, 0) <= now < dtime(16, 0):
        return "AFTERNOON"
    if dtime(16, 0) <= now < dtime(20, 0):
        return "AFTER_HOURS"
    return "OFF_HOURS"

def calculate_volume_ratio(candles: list) -> float:
    """
    Last bar’s volume divided by average volume.
    """
    if not candles:
        return 1.0
    vols = [b["volume"] for b in candles]
    avg = sum(vols) / len(vols)
    return round(candles[-1]["volume"] / max(avg, 1), 2)

def compute_skew_ratio(ticker: str) -> float:
    """
    (call_iv - put_iv)/avg_iv from Polygon snapshot.
    """
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={safe_fetch_polygon_data.API_KEY}"
        data = safe_fetch_polygon_data(url, ticker)
        calls = [o["implied_volatility"] for o in data.get("results", []) if o.get("details", {}).get("contract_type","").lower()=="call"]
        puts  = [o["implied_volatility"] for o in data.get("results", []) if o.get("details", {}).get("contract_type","").lower()=="put"]
        if not calls or not puts:
            return 0.0
        c_iv = sum(calls)/len(calls)
        p_iv = sum(puts)/len(puts)
        avg = (c_iv + p_iv)/2 or 1e-6
        return round((c_iv - p_iv)/avg, 3)
    except:
        return 0.0

def compute_corr_deviation(ticker: str) -> float:
    """
    Load 'corr_history.csv' and compute z-score of last value.
    """
    try:
        import pandas as pd
        df = pd.read_csv("corr_history.csv", index_col=0, parse_dates=True)
        ser = df[ticker].dropna()
        if len(ser) < 2 or ser.std() == 0:
            return 0.0
        z = (ser.iloc[-1] - ser.mean()) / ser.std()
        return round(z, 3)
    except:
        return 0.0

def compute_rsi(candles: list, period: int = 14) -> float:
    """
    Classic RSI on close prices.
    """
    if len(candles) < 2:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(candles)):
        diff = candles[i]["close"] - candles[i-1]["close"]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))
    g = sum(gains[-period:]) / max(period, len(gains))
    l = sum(losses[-period:]) / max(period, len(losses)) or 1e-6
    rs = g / l
    return round(100 - 100/(1+rs), 2)

def detect_yield_spike(tenor: str = "10year", spike_pct: float = 0.25) -> bool:
    """
    >spike_pct move in yield curve since last check.
    Stores last in 'last_yield_<tenor>.json'.
    """
    try:
        fname = f"last_yield_{tenor}.json"
        url = f"https://api.polygon.io/v1/market/yieldcurve/{tenor}?apiKey={safe_fetch_polygon_data.API_KEY}"
        data = safe_fetch_polygon_data(url, "")
        curr = float(data["results"][0]["value"])
        prev = None
        if os.path.exists(fname):
            prev = json.load(open(fname)).get("value")
        with open(fname, "w") as f:
            json.dump({"value": curr}, f)
        return prev is not None and abs(curr - prev)/prev > spike_pct
    except:
        return False
