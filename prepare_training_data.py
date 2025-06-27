#!/usr/bin/env python3
"""
prepare_training_data.py

Fetches historical OHLC bars, computes feature vectors (including treasury yields
from Polygon’s treasury-yields endpoint), and labels each window as CALL/PUT/NEUTRAL.
"""

import os
import pandas as pd
from datetime import datetime, timedelta

from config import TICKERS, tz
from data_ingestion import HistoricalDataLoader
from utils import (
    reformat_candles,
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    detect_yield_spike,
    fetch_option_greeks
)

# ─── Output Configuration ───────────────────────────────────────────────────────
OUTPUT_DIR  = "data"
OUTPUT_FILE = "movement_training_data.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# ─── Parameters ────────────────────────────────────────────────────────────────
HIST_DAYS      = 30
LOOKBACK_BARS  = 12
LOOKAHEAD_BARS = 1

def extract_features_and_label(symbol: str) -> pd.DataFrame:
    # 1) Fetch OHLC bars
    end   = datetime.now(tz)
    start = end - timedelta(days=HIST_DAYS)
    loader   = HistoricalDataLoader()
    raw_bars = loader.fetch_bars(symbol, start, end)

    # 2) Build DataFrame
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

    records = []
    # 3) Slide window to compute features + label
    for i in range(LOOKBACK_BARS, len(df) - LOOKAHEAD_BARS):
        window  = df.iloc[i-LOOKBACK_BARS : i]
        current = df.iloc[i]
        candles = window.reset_index().to_dict("records")

        # feature computations
        feat = {
            "symbol":            symbol,
            "breakout_prob":     calculate_breakout_prob(candles),
            "recent_move_pct":   calculate_recent_move_pct(candles),
            "time_of_day":       calculate_time_of_day(current.name),
            "volume_ratio":      calculate_volume_ratio(candles),
            "rsi":               compute_rsi(candles),
            "corr_dev":          compute_corr_deviation(symbol),
            "skew_ratio":        compute_skew_ratio(symbol),
            "yield_spike_2year": detect_yield_spike("2year"),
            "yield_spike_10year":detect_yield_spike("10year"),
            "yield_spike_30year":detect_yield_spike("30year"),
            **fetch_option_greeks(symbol)
        }

        # label next bar’s return
        next_bar = df.iloc[i + LOOKAHEAD_BARS]
        move_pct = (next_bar["close"] - current["close"]) / current["close"]
        feat["movement_type"] = (
            "CALL" if move_pct > 0
            else "PUT" if move_pct < 0
            else "NEUTRAL"
        )

        records.append(feat)

    return pd.DataFrame(records)

def main():
    all_dfs = []
    for ticker in TICKERS:
        print(f"Generating data for {ticker}…")
        all_dfs.append(extract_features_and_label(ticker))

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    full.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(full)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
