# utils/analysis.py

import os
import json
import math
from datetime import datetime, time as dtime
from collections import deque
from typing import List, Tuple

from config import (
    LOOKBACK_BREAKOUT,
    LOOKBACK_RISK_REWARD,
    tz,
)
from utils.http_client import safe_fetch_polygon_data

def calculate_breakout_prob(candles: List[dict], lookback: int = LOOKBACK_BREAKOUT) -> float:
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves = []
    vols  = []
    for b in recent:
        o, c = b["open"], b["close"]
        if o:
            moves.append((c - o) / o)
            vols.append(b["volume"])
    if not moves:
        return 0.0
    avgm = sum(abs(m) for m in moves) / len(moves)
    avgv = sum(vols) / len(vols)
    last = recent[-1]
    cm   = abs(last["close"] - last["open"]) / last["open"] if last["open"] else 0.0
    cv   = last["volume"]
    trend = sum(1 if m>0 else -1 for m in moves) / len(moves)
    score = 50 * math.log1p(cm / max(avgm, 1e-6)) * math.log1p(cv / max(avgv, 1e-6)) * (1 + trend)
    return round(max(0, min(100, score)), 2)

def calculate_recent_move_pct(candles: List[dict]) -> float:
    if len(candles) < 2:
        return 0.0
    first = candles[0]; last = candles[-1]
    if not first["open"]:
        return 0.0
    return round((last["close"] - first["open"]) / first["open"], 4)

def calculate_signal_persistence(candles: List[dict], lookback: int = LOOKBACK_BREAKOUT) -> float:
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves = [abs((b["close"] - b["open"]) / b["open"]) for b in recent if b["open"]]
    if not moves:
        return 0.0
    avgm = sum(moves) / len(moves)
    last = moves[-1]
    return round(100 * (1 - min(last / max(avgm, 1e-6), 1)), 2)

def calculate_reversal_and_scope(candles: List[dict], oi_bias: str) -> Tuple[bool, bool, float]:
    if len(candles) < 2:
        return False, False, 0.0
    prev, last = candles[-2], candles[-1]
    mv = (last["close"] - prev["close"]) / prev["close"] if prev["close"] else 0.0
    rev = (oi_bias == "CALL_DOMINANT" and mv < -0.01) or (oi_bias == "PUT_DOMINANT" and mv > 0.01)
    sc  = abs(mv) > 0.005
    return rev, sc, round(mv * 100, 4)

def calculate_risk_reward(candles: List[dict]) -> float:
    if len(candles) < LOOKBACK_RISK_REWARD:
        return 1.0
    sample = candles[-LOOKBACK_RISK_REWARD:]
    atr = sum(b["high"] - b["low"] for b in sample) / len(sample)
    return round((atr * 2) / max(atr * 0.5, 1e-6), 2)

def calculate_time_of_day(as_of: datetime | None = None) -> str:
    now = (as_of or datetime.now(tz)).astimezone(tz).time()
    if dtime(4,0) <= now < dtime(9,30):  return "PRE_MARKET"
    if dtime(9,30) <= now < dtime(11,0): return "MORNING"
    if dtime(11,0) <= now < dtime(14,0): return "MIDDAY"
    if dtime(14,0) <= now < dtime(16,0): return "AFTERNOON"
    if dtime(16,0) <= now < dtime(20,0): return "AFTER_HOURS"
    return "OFF_HOURS"

def calculate_volume_ratio(candles: List[dict]) -> float:
    if not candles:
        return 1.0
    vols = [b["volume"] for b in candles]
    avg  = sum(vols) / len(vols)
    return round(candles[-1]["volume"] / max(avg, 1), 2)

def compute_skew_ratio(ticker: str) -> float:
    try:
        data = safe_fetch_polygon_data(
            f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={os.getenv('POLYGON_API_KEY')}",
            ticker
        ).get("results", [])
        calls = [o["implied_volatility"] for o in data if o.get("details",{}).get("contract_type","").lower()=="call"]
        puts  = [o["implied_volatility"] for o in data if o.get("details",{}).get("contract_type","").lower()=="put"]
        if not calls or not puts:
            return 0.0
        c_iv = sum(calls)/len(calls)
        p_iv = sum(puts)/len(puts)
        avg = (c_iv + p_iv)/2 or 1e-6
        return round((c_iv - p_iv)/avg, 3)
    except:
        return 0.0

def compute_corr_deviation(ticker: str) -> float:
    try:
        import pandas as pd
        df = pd.read_csv("corr_history.csv", index_col=0, parse_dates=True)
        ser = df[ticker].dropna()
        if len(ser)<2 or ser.std()==0:
            return 0.0
        return round((ser.iloc[-1] - ser.mean())/ser.std(), 3)
    except:
        return 0.0

def compute_rsi(candles: List[dict], period: int = 14) -> float:
    if len(candles) < 2:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(candles)):
        diff = candles[i]["close"] - candles[i-1]["close"]
        gains.append(max(diff,0))
        losses.append(abs(min(diff,0)))
    avg_g = sum(gains[-period:]) / max(period, len(gains)) or 1e-6
    avg_l = sum(losses[-period:]) / max(period, len(losses)) or 1e-6
    rs = avg_g / avg_l
    return round(100 - 100/(1+rs), 2)

def detect_yield_spike(tenor: str = "10year", spike_pct: float = 0.25) -> bool:
    fname = f"last_yield_{tenor}.json"
    try:
        url  = f"https://api.polygon.io/v1/market/yieldcurve/{tenor}?apiKey={os.getenv('POLYGON_API_KEY')}"
        data = safe_fetch_polygon_data(url, tenor)
        curr = float(data["results"][0]["value"])
        prev = None
        if os.path.exists(fname):
            prev = json.load(open(fname)).get("value")
        with open(fname, "w") as f:
            json.dump({"value":curr}, f)
        return prev is not None and abs(curr - prev)/prev > spike_pct
    except:
        return False
