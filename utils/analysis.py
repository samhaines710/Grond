"""Analytical utilities for evaluating market data and signals.

This module contains a collection of functions used to calculate
probabilities, ratios, and statistical indicators based on price and
volume data. It also includes helpers to fetch snapshots from the
Polygon API and compute derived metrics like skew ratio, correlation
deviation, RSI, and yield spikes.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, time as dtime
from typing import List, Tuple, Optional

from config import LOOKBACK_BREAKOUT, LOOKBACK_RISK_REWARD, tz
from utils.http_client import safe_fetch_polygon_data


def calculate_breakout_prob(candles: List[dict], lookback: int = LOOKBACK_BREAKOUT) -> float:
    """
    Estimate the probability of a breakout given a list of candles.

    The function looks back over the specified number of bars and computes
    a score based on price movement, volume, and trend direction. The
    resulting value is a percentage between 0 and 100.
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves: List[float] = []
    vols: List[float] = []
    for bar in recent:
        open_price = bar["open"]
        close_price = bar["close"]
        if open_price:
            moves.append((close_price - open_price) / open_price)
            vols.append(bar["volume"])
    if not moves:
        return 0.0
    avg_move = sum(abs(m) for m in moves) / len(moves)
    avg_vol = sum(vols) / len(vols)
    last = recent[-1]
    cm = abs(last["close"] - last["open"]) / last["open"] if last["open"] else 0.0
    cv = last["volume"]
    trend = sum(1.0 if m > 0 else -1.0 for m in moves) / len(moves)
    score = (
        50
        * math.log1p(cm / max(avg_move, 1e-6))
        * math.log1p(cv / max(avg_vol, 1e-6))
        * (1 + trend)
    )
    return round(max(0.0, min(100.0, score)), 2)


def calculate_recent_move_pct(candles: List[dict]) -> float:
    """
    Compute the percentage change from the first bar's open to the last bar's close.
    """
    if len(candles) < 2:
        return 0.0
    first = candles[0]
    last = candles[-1]
    if not first["open"]:
        return 0.0
    pct = (last["close"] - first["open"]) / first["open"]
    return round(pct, 4)


def calculate_signal_persistence(
    candles: List[dict],
    lookback: int = LOOKBACK_BREAKOUT,
) -> float:
    """
    Estimate how persistent a signal is by comparing the last bar's move to average moves.
    """
    if len(candles) < lookback:
        return 0.0
    recent = candles[-lookback:]
    moves = [
        abs((bar["close"] - bar["open"]) / bar["open"])
        for bar in recent
        if bar["open"]
    ]
    if not moves:
        return 0.0
    avg_move = sum(moves) / len(moves)
    last_move = moves[-1]
    return round(100.0 * (1 - min(last_move / max(avg_move, 1e-6), 1.0)), 2)


def calculate_reversal_and_scope(
    candles: List[dict],
    oi_bias: str,
) -> Tuple[bool, bool, float]:
    """
    Determine if there is a reversal and whether the magnitude (scope) is significant.

    Returns a tuple of ``(is_reversal, is_scope, move_pct*100)`` where ``is_reversal``
    indicates whether the price move conflicts with the open interest bias and ``is_scope``
    flags whether the absolute move is above a small threshold.
    """
    if len(candles) < 2:
        return False, False, 0.0
    prev, last = candles[-2], candles[-1]
    prev_close = prev["close"]
    mv = (last["close"] - prev_close) / prev_close if prev_close else 0.0
    rev = (
        (oi_bias == "CALL_DOMINANT" and mv < -0.01)
        or (oi_bias == "PUT_DOMINANT" and mv > 0.01)
    )
    sc = abs(mv) > 0.005
    return rev, sc, round(mv * 100.0, 4)


def calculate_risk_reward(candles: List[dict]) -> float:
    """
    Compute a simple risk–reward ratio based on high–low ranges over a lookback window.
    """
    if len(candles) < LOOKBACK_RISK_REWARD:
        return 1.0
    sample = candles[-LOOKBACK_RISK_REWARD:]
    atr = sum(bar["high"] - bar["low"] for bar in sample) / len(sample)
    return round((atr * 2.0) / max(atr * 0.5, 1e-6), 2)


def calculate_time_of_day(as_of: Optional[datetime] = None) -> str:
    """
    Categorize the current time into trading session labels.
    """
    now_time = (as_of or datetime.now(tz)).astimezone(tz).time()
    if dtime(4, 0) <= now_time < dtime(9, 30):
        return "PRE_MARKET"
    if dtime(9, 30) <= now_time < dtime(11, 0):
        return "MORNING"
    if dtime(11, 0) <= now_time < dtime(14, 0):
        return "MIDDAY"
    if dtime(14, 0) <= now_time < dtime(16, 0):
        return "AFTERNOON"
    if dtime(16, 0) <= now_time < dtime(20, 0):
        return "AFTER_HOURS"
    return "OFF_HOURS"


def calculate_volume_ratio(candles: List[dict]) -> float:
    """
    Compute the ratio of the last bar's volume to the average volume.
    """
    if not candles:
        return 1.0
    vols = [bar["volume"] for bar in candles]
    avg_vol = sum(vols) / len(vols)
    last_vol = candles[-1]["volume"]
    return round(last_vol / max(avg_vol, 1.0), 2)


def compute_skew_ratio(ticker: str) -> float:
    """
    Compute the skew ratio of implied volatilities between call and put options.
    """
    try:
        data = safe_fetch_polygon_data(
            f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={os.getenv('POLYGON_API_KEY')}",
            ticker,
        ).get("results", [])
        calls = [
            o["implied_volatility"]
            for o in data
            if o.get("details", {}).get("contract_type", "").lower() == "call"
        ]
        puts = [
            o["implied_volatility"]
            for o in data
            if o.get("details", {}).get("contract_type", "").lower() == "put"
        ]
        if not calls or not puts:
            return 0.0
        c_iv = sum(calls) / len(calls)
        p_iv = sum(puts) / len(puts)
        avg = (c_iv + p_iv) / 2.0 or 1e-6
        return round((c_iv - p_iv) / avg, 3)
    except Exception:
        return 0.0


def compute_corr_deviation(ticker: str) -> float:
    """
    Compare the latest correlation value against historical mean and standard deviation
    from a CSV file.
    """
    try:
        import pandas as pd  # local import to avoid an unconditional dependency

        df = pd.read_csv("corr_history.csv", index_col=0, parse_dates=True)
        series = df[ticker].dropna()
        if len(series) < 2 or series.std() == 0:
            return 0.0
        return round((series.iloc[-1] - series.mean()) / series.std(), 3)
    except Exception:
        return 0.0


def compute_rsi(candles: List[dict], period: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) over the specified period.
    """
    if len(candles) < 2:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(candles)):
        diff = candles[i]["close"] - candles[i - 1]["close"]
        gains.append(max(diff, 0.0))
        losses.append(abs(min(diff, 0.0)))
    lookback = min(period, len(gains))
    avg_g = sum(gains[-lookback:]) / max(period, len(gains)) or 1e-6
    avg_l = sum(losses[-lookback:]) / max(period, len(losses)) or 1e-6
    rs = avg_g / avg_l
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def detect_yield_spike(tenor: str = "10year", spike_pct: float = 0.25) -> bool:
    """
    Detect large jumps in yield curve values compared to the last recorded value.
    """
    fname = f"last_yield_{tenor}.json"
    try:
        url = (
            f"https://api.polygon.io/v1/market/yieldcurve/{tenor}"
            f"?apiKey={os.getenv('POLYGON_API_KEY')}"
        )
        data = safe_fetch_polygon_data(url, tenor)
        curr = float(data["results"][0]["value"])
        prev: Optional[float] = None
        if os.path.exists(fname):
            with open(fname) as f:
                prev = json.load(f).get("value")
        with open(fname, "w") as f:
            json.dump({"value": curr}, f)
        return prev is not None and prev != 0 and abs(curr - prev) / prev > spike_pct
    except Exception:
        return False
