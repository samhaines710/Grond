# src/utils/analysis.py

import math
import os
import json
from datetime import datetime, date, timedelta, time as dtime

import pandas as pd
from config import (
    POLYGON_API_KEY,
    LOOKBACK_BREAKOUT,
    LOOKBACK_RISK_REWARD,
    tz
)
from utils.http_client import safe_fetch_polygon_data

# ─── Breakout Probability ────────────────────────────────────────────────────────
def calculate_breakout_prob(candles: list, lookback: int = LOOKBACK_BREAKOUT) -> float:
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves, vols = [], []
    for b in recent:
        o, c_ = b["open"], b["close"]
        if o:
            moves.append((c_ - o)/o)
            vols.append(b["volume"])
    if not moves:
        return 0.0
    avgm = sum(abs(m) for m in moves)/len(moves)
    avgv = sum(vols)/len(vols)
    lb = recent[-1]
    cm = abs(lb["close"] - lb["open"])/lb["open"] if lb["open"] else 0
    cv = lb["volume"]
    trend = sum(1 if m>0 else -1 for m in moves)/len(moves)
    score = 50 * math.log1p(cm/max(avgm,1e-5)) * math.log1p(cv/max(avgv,1)) * (1+trend)
    return round(max(0, min(100, score)), 2)

# ─── Recent Move ────────────────────────────────────────────────────────────────
def calculate_recent_move_pct(candles: list) -> float:
    if len(candles) < 2:
        return 0.0
    # find first bar at or before 9:30am
    first = next(
        (b for b in candles
         if datetime.fromtimestamp(b["timestamp"]/1000, tz).time() <= dtime(9,30)),
        candles[0]
    )
    return round((candles[-1]["close"] - first["open"])/first["open"], 4)

# ─── Signal Persistence ─────────────────────────────────────────────────────────
def calculate_signal_persistence(candles: list, lookback: int = LOOKBACK_BREAKOUT) -> float:
    if len(candles) < lookback:
        return 0.0
    moves = [
        abs((candles[i]["close"] - candles[i]["open"])/candles[i]["open"])
        for i in range(-lookback, 0)
    ]
    if not moves:
        return 0.0
    avgm = sum(moves)/len(moves)
    last = moves[-1]
    return round(100 * (1 - min(last/max(avgm,1e-5),1)), 2)

# ─── Reversal & Scope ───────────────────────────────────────────────────────────
def calculate_reversal_and_scope(oi_bias: str, candles: list) -> tuple[bool,bool,float]:
    if len(candles) < 3:
        return False, False, 0.0
    prev, cur = candles[-2]["close"], candles[-1]["close"]
    mv = (cur - prev)/prev if prev else 0.0
    rev = (oi_bias=="CALL_DOMINANT" and mv < -0.01) or (oi_bias=="PUT_DOMINANT" and mv > 0.01)
    sc  = abs(mv) > 0.005
    return rev, sc, round(mv,4)

# ─── Risk/Reward (ATR) ──────────────────────────────────────────────────────────
def calculate_risk_reward(candles: list, lookback: int = LOOKBACK_RISK_REWARD) -> float:
    if len(candles) < lookback:
        return 1.0
    tr = [(b["high"] - b["low"]) for b in candles[-lookback:]]
    atr = sum(tr)/len(tr) if tr else 0.0
    return round((atr*2)/max(atr*0.5,1e-5), 2)

# ─── Volume Ratio ───────────────────────────────────────────────────────────────
def calculate_volume_ratio(candles: list) -> float:
    if not candles:
        return 1.0
    vols = [b["volume"] for b in candles]
    avgv = sum(vols)/len(vols) if vols else 1.0
    return round(candles[-1]["volume"]/max(avgv,1), 2)

# ─── Time‐of‐Day Label ───────────────────────────────────────────────────────────
def calculate_time_of_day(as_of: datetime | None = None) -> str:
    now = (as_of or datetime.now(tz)).astimezone(tz).time()
    if dtime(4,0) <= now < dtime(9,30):   return "PRE_MARKET"
    if dtime(9,30) <= now < dtime(11,0):  return "MORNING"
    if dtime(11,0) <= now < dtime(14,0):  return "MIDDAY"
    if dtime(14,0) <= now < dtime(16,0):  return "AFTERNOON"
    if dtime(16,0) <= now < dtime(20,0):  return "AFTER_HOURS"
    return "OFF_HOURS"

# ─── RSI ────────────────────────────────────────────────────────────────────────
def compute_rsi(candles: list, period: int = 14) -> float:
    if len(candles) < 2:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(candles)):
        diff = candles[i]["close"] - candles[i-1]["close"]
        gains.append(max(diff,0))
        losses.append(abs(min(diff,0)))
    g = gains[-period:] if len(gains)>=period else gains
    l = losses[-period:] if len(losses)>=period else losses
    avg_g = sum(g)/len(g) if g else 1e-6
    avg_l = sum(l)/len(l) or 1e-6
    rs = avg_g/avg_l
    return round(100 - 100/(1+rs), 2)

# ─── Correlation Deviation (z‐score) ─────────────────────────────────────────────
def compute_corr_deviation(ticker: str) -> float:
    try:
        df = pd.read_csv("corr_history.csv", index_col=0, parse_dates=True)
        ser = df[ticker].dropna()
        if len(ser)<2 or ser.std()==0:
            return 0.0
        z = (ser.iloc[-1] - ser.mean())/ser.std()
        return round(z, 3)
    except Exception:
        return 0.0

# ─── Skew Ratio ─────────────────────────────────────────────────────────────────
def compute_skew_ratio(ticker: str) -> float:
    results = safe_fetch_polygon_data(
        f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={POLYGON_API_KEY}",
        ticker
    ).get("results", [])
    calls, puts = [], []
    for o in results:
        iv = o.get("implied_volatility")
        typ = o.get("details",{}).get("contract_type","").lower()
        if iv is None or typ not in ("call","put"):
            continue
        (calls if typ=="call" else puts).append(iv)
    if not calls or not puts:
        return 0.0
    c_iv, p_iv = sum(calls)/len(calls), sum(puts)/len(puts)
    avg = (c_iv + p_iv)/2 or 1e-6
    return round((c_iv - p_iv)/avg, 3)

# ─── Yield Spike Detection ───────────────────────────────────────────────────────
def detect_yield_spike(tenor: str = "10year", spike_pct: float = 0.25) -> bool:
    fname = f"last_yield_{tenor}.json"
    try:
        data = safe_fetch_polygon_data(
            f"https://api.polygon.io/v1/market/yieldcurve/{tenor}?apiKey={POLYGON_API_KEY}",
            ""
        )
        curr = float(data["results"][0]["value"])
        prev = None
        if os.path.exists(fname):
            prev = json.load(open(fname)).get("value")
        with open(fname, "w") as f:
            json.dump({"value": curr}, f)
        if prev and prev!=0 and abs(curr - prev)/prev > spike_pct:
            return True
    except Exception:
        pass
    return False
