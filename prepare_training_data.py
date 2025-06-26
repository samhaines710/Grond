#!/usr/bin/env python3
# prepare_training_data.py

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

from config import TICKERS, tz
from data_ingestion import HistoricalDataLoader
from utils import (
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    fetch_option_greeks
)

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":%(message)s}'
)
logger = logging.getLogger(__name__)

# ─── Fetch Treasury Yields ─────────────────────────────────────────────────────
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
YIELDS_ENDPOINT = "https://api.polygon.io/fed/v1/treasury-yields"

def fetch_treasury_yields(date: str = None) -> dict:
    """
    Calls Polygon's treasury-yields endpoint once and returns a dict
    mapping 'yield_2_year', 'yield_10_year', 'yield_30_year', etc.
    """
    params = {"apiKey": POLYGON_API_KEY}
    if date:
        params["date"] = date
    resp = requests.get(YIELDS_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("results", [])
    if not data:
        logger.warning(f'"No yield data returned for date={date}"')
        return {}
    record = data[0]
    logger.info(f'"Fetched treasury yields for date={record.get("date")}"')
    return record

# Cache the yields once per run
YIELDS = fetch_treasury_yields()

# ─── Output Configuration ───────────────────────────────────────────────────────
OUTPUT_DIR  = "data"
OUTPUT_FILE = "movement_training_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# ─── Parameters (allow CI override) ────────────────────────────────────────────
HIST_DAYS      = int(os.getenv("HIST_DAYS", 30))
LOOKBACK_BARS  = 12
LOOKAHEAD_BARS = 1

def extract_features_and_label(symbol: str) -> pd.DataFrame:
    # 1) Load OHLC bars
    end   = datetime.now(tz)
    start = end - timedelta(days=HIST_DAYS)
    loader   = HistoricalDataLoader()
    raw_bars = loader.fetch_bars(symbol, start, end)
    logger.info(f'"Fetched {len(raw_bars)} bars for {symbol} over {HIST_DAYS} days"')

    # 2) Frame into DataFrame with datetime index
    df = pd.DataFrame([{
        "timestamp": b["t"],
        "open":      b["o"],
        "high":      b["h"],
        "low":       b["l"],
        "close":     b["c"],
        "volume":    b["v"]
    } for b in raw_bars])
    df["dt"] = (
        pd.to_datetime(df["timestamp"], unit="ms")
          .dt.tz_localize("UTC")
          .dt.tz_convert(tz)
    )
    df.set_index("dt", inplace=True)

    # 3) Hoist yield and Greeks outside loop
    ys2  = float(YIELDS.get("yield_2_year", 0.0))
    ys10 = float(YIELDS.get("yield_10_year", 0.0))
    ys30 = float(YIELDS.get("yield_30_year", 0.0))
    greeks = fetch_option_greeks(symbol)
    logger.info(f'"Using yields (2y={ys2},10y={ys10},30y={ys30}) and Greeks for {symbol}"')

    records = []
    # 4) Slide window to compute features + label
    for i in range(LOOKBACK_BARS, len(df) - LOOKAHEAD_BARS):
        window  = df.iloc[i-LOOKBACK_BARS : i]
        current = df.iloc[i]
        candles = window.reset_index().to_dict("records")

        feat = {
            "symbol":          symbol,
            "breakout_prob":   calculate_breakout_prob(candles),
            "recent_move_pct": calculate_recent_move_pct(candles),
            "time_of_day":     calculate_time_of_day(current.name),
            "volume_ratio":    calculate_volume_ratio(candles),
            "rsi":             compute_rsi(candles),
            "corr_dev":        compute_corr_deviation(symbol),
            "skew_ratio":      compute_skew_ratio(symbol),
            "yield_spike_2year":  ys2,
            "yield_spike_10year": ys10,
            "yield_spike_30year": ys30,
            **greeks
        }

        # 5) Label next bar’s return
        next_bar = df.iloc[i + LOOKAHEAD_BARS]
        delta = (next_bar["close"] - current["close"]) / current["close"]
        feat["movement_type"] = "CALL" if delta > 0 else ("PUT" if delta < 0 else "NEUTRAL")

        records.append(feat)

    return pd.DataFrame(records)

def main():
    all_dfs = []
    for ticker in TICKERS:
        logger.info(f'"Generating data for {ticker}"')
        df_ticker = extract_features_and_label(ticker)
        all_dfs.append(df_ticker)

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full.to_csv(OUTPUT_PATH, index=False)
    logger.info(f'"✅ Saved {len(full)} rows to {OUTPUT_PATH}"')

if __name__ == "__main__":
    main()
